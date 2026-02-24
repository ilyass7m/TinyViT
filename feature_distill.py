# --------------------------------------------------------
# Feature Distillation Module for TinyViT
# Extension to add feature-level knowledge distillation
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProjectionHead(nn.Module):
    """
    Projects student features to teacher feature dimension.

    Simple linear projection works well in practice. Can be extended
    to MLP if needed.
    """
    def __init__(
        self,
        student_dim: int = 320,  # TinyViT-5M
        teacher_dim: int = 768,  # CLIP-ViT-L/14
        hidden_dim: Optional[int] = None,
        use_mlp: bool = False,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

        if use_mlp and hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
            )
        else:
            self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FeatureDistillationLoss(nn.Module):
    """
    Feature distillation loss using cosine similarity.

    L_feature = 1 - cosine_similarity(proj(student_feat), teacher_feat)

    Cosine similarity is preferred over L2 because:
    - Scale-invariant (CLIP features have different magnitude)
    - Works well with normalized feature spaces
    """
    def __init__(
        self,
        student_dim: int = 320,
        teacher_dim: int = 768,
        use_mlp: bool = False,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.projection = ProjectionHead(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            use_mlp=use_mlp,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        student_features: torch.Tensor,  # [B, student_dim]
        teacher_features: torch.Tensor,  # [B, teacher_dim]
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss between projected student and teacher features.

        Args:
            student_features: Student model features [B, student_dim]
            teacher_features: Teacher model features [B, teacher_dim]

        Returns:
            Scalar loss value (1 - mean cosine similarity)
        """
        # Ensure float32 for projection (handles AMP FP16 inputs)
        student_features = student_features.float()
        teacher_features = teacher_features.float()

        # Project student features to teacher dimension
        student_proj = self.projection(student_features)

        # L2 normalize both
        student_norm = F.normalize(student_proj, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)

        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)

        # Loss is 1 - similarity (so minimizing loss maximizes similarity)
        loss = 1.0 - cosine_sim.mean()

        return loss


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss for logit + feature distillation.

    L_total = (1-α)*CE(student, hard_labels) + α*KL(student, teacher_logits) + β*L_feature

    Where:
    - α (alpha): weight for logit distillation (default 0.5)
    - β (beta): weight for feature distillation (default 0.5)
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 1.0,
        student_dim: int = 320,
        teacher_dim: int = 768,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss()
        self.feature_loss = FeatureDistillationLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        )

    def forward(
        self,
        student_logits: torch.Tensor,     # [B, num_classes]
        student_features: torch.Tensor,   # [B, student_dim]
        targets: torch.Tensor,            # [B]
        teacher_probs: torch.Tensor,      # [B, num_classes] (soft labels)
        teacher_features: torch.Tensor,   # [B, teacher_dim]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined distillation loss.

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary with individual loss components
        """
        # 1. Cross-entropy with hard labels
        ce = self.ce_loss(student_logits, targets)

        # 2. KL divergence with teacher soft labels
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs_temp = teacher_probs.clamp(min=1e-8)
        kl = F.kl_div(student_log_probs, teacher_probs_temp, reduction='batchmean')
        kl = kl * (self.temperature ** 2)  # Scale by T^2

        # 3. Feature distillation loss
        feat_loss = self.feature_loss(student_features, teacher_features)

        # Combined loss
        # L = (1-α)*CE + α*KL + β*Feature
        # Note: We apply alpha to both CE and KL, then add beta*feature separately
        logit_loss = (1 - self.alpha) * ce + self.alpha * kl
        total_loss = logit_loss + self.beta * feat_loss

        loss_dict = {
            'ce_loss': ce.item(),
            'kl_loss': kl.item(),
            'feature_loss': feat_loss.item(),
            'logit_loss': logit_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict


class TeacherModelWrapper(nn.Module):
    """
    Wrapper for teacher model (CLIP) to extract both logits and features.

    The teacher is always in eval mode and doesn't compute gradients.
    """
    def __init__(self, teacher_model: nn.Module):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()

        # Freeze all parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
            features: Feature embeddings [B, feature_dim]
        """
        # Get features from visual encoder
        features = self.teacher.forward_features(x)

        # Get logits from classification head
        logits = self.teacher.head(features)

        return logits, features

    def get_feature_dim(self) -> int:
        """Return the teacher's feature dimension."""
        if hasattr(self.teacher, 'num_features'):
            return self.teacher.num_features
        elif hasattr(self.teacher, 'embed_dim'):
            return self.teacher.embed_dim
        else:
            # For CLIP wrapper
            return 768  # CLIP-ViT-L/14 default


def build_teacher_for_feature_distill(config):
    """
    Build and load the teacher model for online feature distillation.

    Args:
        config: Config with MODEL and DISTILL settings

    Returns:
        TeacherModelWrapper with loaded checkpoint
    """
    from models import build_model

    teacher_type = config.DISTILL.TEACHER_TYPE
    print(f"Building teacher model of type: {teacher_type}")

    # Store original config values
    original_type = config.MODEL.TYPE
    original_tiny_vit = None

    config.defrost()

    if teacher_type == 'tiny_vit':
        # Build TinyViT teacher with specified architecture
        config.MODEL.TYPE = 'tiny_vit'
        # Store original TinyViT config
        original_tiny_vit = {
            'EMBED_DIMS': list(config.MODEL.TINY_VIT.EMBED_DIMS),
            'DEPTHS': list(config.MODEL.TINY_VIT.DEPTHS),
            'NUM_HEADS': list(config.MODEL.TINY_VIT.NUM_HEADS),
            'WINDOW_SIZES': list(config.MODEL.TINY_VIT.WINDOW_SIZES),
            'MLP_RATIO': config.MODEL.TINY_VIT.MLP_RATIO,
            'MBCONV_EXPAND_RATIO': config.MODEL.TINY_VIT.MBCONV_EXPAND_RATIO,
            'LOCAL_CONV_SIZE': config.MODEL.TINY_VIT.LOCAL_CONV_SIZE,
        }
        # Apply teacher TinyViT architecture
        config.MODEL.TINY_VIT.EMBED_DIMS = list(config.DISTILL.TEACHER_TINY_VIT.EMBED_DIMS)
        config.MODEL.TINY_VIT.DEPTHS = list(config.DISTILL.TEACHER_TINY_VIT.DEPTHS)
        config.MODEL.TINY_VIT.NUM_HEADS = list(config.DISTILL.TEACHER_TINY_VIT.NUM_HEADS)
        config.MODEL.TINY_VIT.WINDOW_SIZES = list(config.DISTILL.TEACHER_TINY_VIT.WINDOW_SIZES)
        config.MODEL.TINY_VIT.MLP_RATIO = config.DISTILL.TEACHER_TINY_VIT.MLP_RATIO
        config.MODEL.TINY_VIT.MBCONV_EXPAND_RATIO = config.DISTILL.TEACHER_TINY_VIT.MBCONV_EXPAND_RATIO
        config.MODEL.TINY_VIT.LOCAL_CONV_SIZE = config.DISTILL.TEACHER_TINY_VIT.LOCAL_CONV_SIZE
    else:
        # For other model types (vit_base_patch16_224, clip_vit_large_patch14, resnet152, etc.)
        config.MODEL.TYPE = teacher_type

    config.freeze()

    # Build teacher model
    teacher = build_model(config)

    # Restore original config
    config.defrost()
    config.MODEL.TYPE = original_type
    if original_tiny_vit is not None:
        config.MODEL.TINY_VIT.EMBED_DIMS = original_tiny_vit['EMBED_DIMS']
        config.MODEL.TINY_VIT.DEPTHS = original_tiny_vit['DEPTHS']
        config.MODEL.TINY_VIT.NUM_HEADS = original_tiny_vit['NUM_HEADS']
        config.MODEL.TINY_VIT.WINDOW_SIZES = original_tiny_vit['WINDOW_SIZES']
        config.MODEL.TINY_VIT.MLP_RATIO = original_tiny_vit['MLP_RATIO']
        config.MODEL.TINY_VIT.MBCONV_EXPAND_RATIO = original_tiny_vit['MBCONV_EXPAND_RATIO']
        config.MODEL.TINY_VIT.LOCAL_CONV_SIZE = original_tiny_vit['LOCAL_CONV_SIZE']
    config.freeze()

    # Load teacher checkpoint (required for online distillation)
    if config.DISTILL.TEACHER_CHECKPOINT:
        checkpoint = torch.load(config.DISTILL.TEACHER_CHECKPOINT, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        teacher.load_state_dict(state_dict, strict=False)
        print(f"Loaded teacher checkpoint: {config.DISTILL.TEACHER_CHECKPOINT}")
    else:
        print("WARNING: No teacher checkpoint provided. Using randomly initialized teacher.")

    return TeacherModelWrapper(teacher)


# Utility function to add feature distillation to existing training
def get_student_features(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get both logits and features from student model.

    For TinyViT, we need to call forward_features then head separately.
    """
    if hasattr(model, 'module'):
        model = model.module  # Handle DDP

    features = model.forward_features(x)
    logits = model.head(features)

    return logits, features


class OnlineDistillationLoss(nn.Module):
    """
    Online distillation loss for training with live teacher forward passes.

    Supports two modes:
    1. Logits-only: L = KL(student_logits, teacher_logits)
    2. Logits + Features: L = KL(student_logits, teacher_logits) + β * L_feature

    This is designed to be backward-compatible with the saved-logits approach
    while adding optional feature distillation.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        feature_enabled: bool = False,
        feature_weight: float = 0.5,
        student_dim: int = 320,
        teacher_dim: int = 576,
    ):
        super().__init__()
        self.temperature = temperature
        self.feature_enabled = feature_enabled
        self.feature_weight = feature_weight

        if feature_enabled:
            self.feature_loss = FeatureDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
            )
        else:
            self.feature_loss = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute online distillation loss.

        Args:
            student_logits: Student predictions [B, num_classes]
            teacher_logits: Teacher predictions [B, num_classes]
            student_features: Optional student features [B, student_dim]
            teacher_features: Optional teacher features [B, teacher_dim]

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components for logging
        """
        # KL divergence loss on logits
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)  # Scale by T^2

        loss_dict = {
            'kl_loss': kl_loss.item(),
        }

        total_loss = kl_loss

        # Feature distillation loss (optional)
        if self.feature_enabled and student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.feature_weight * feat_loss
            loss_dict['feature_loss'] = feat_loss.item()
            loss_dict['feature_weight'] = self.feature_weight

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def build_online_distill_loss(config) -> OnlineDistillationLoss:
    """
    Build online distillation loss module from config.

    Args:
        config: Config with DISTILL settings

    Returns:
        OnlineDistillationLoss module
    """
    return OnlineDistillationLoss(
        temperature=config.DISTILL.TEMPERATURE,
        feature_enabled=config.DISTILL.FEATURE_ENABLED,
        feature_weight=config.DISTILL.FEATURE_WEIGHT,
        student_dim=config.DISTILL.FEATURE_DIM_STUDENT,
        teacher_dim=config.DISTILL.FEATURE_DIM_TEACHER,
    )
