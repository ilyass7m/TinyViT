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

    # Temporarily modify config to build teacher
    # We need to build CLIP model, not TinyViT
    original_type = config.MODEL.TYPE

    # Build CLIP model
    config.defrost()
    config.MODEL.TYPE = 'clip_vit_large_patch14'
    config.freeze()

    teacher = build_model(config)

    # Restore original model type
    config.defrost()
    config.MODEL.TYPE = original_type
    config.freeze()

    # Load teacher checkpoint if provided
    if hasattr(config.DISTILL, 'TEACHER_CHECKPOINT') and config.DISTILL.TEACHER_CHECKPOINT:
        checkpoint = torch.load(config.DISTILL.TEACHER_CHECKPOINT, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        teacher.load_state_dict(state_dict, strict=False)
        print(f"Loaded teacher checkpoint: {config.DISTILL.TEACHER_CHECKPOINT}")

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
