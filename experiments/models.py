"""
Model Definitions for TinyViT CIFAR-100 Experiments

Provides unified interface for:
- Student models: TinyViT-5M, TinyViT-11M
- Teacher models: ResNet-50, ViT-Base, TinyViT-21M
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import timm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
from .config import TeacherType, StudentType


# =============================================================================
# TinyViT Model Builders
# =============================================================================

def build_tinyvit_5m(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    Build TinyViT-5M model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        TinyViT-5M model
    """
    model = tiny_vit_5m_224(pretrained=pretrained)

    # Replace classification head if num_classes differs
    if num_classes != 1000:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model


def build_tinyvit_11m(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """Build TinyViT-11M model."""
    model = tiny_vit_11m_224(pretrained=pretrained)

    if num_classes != 1000:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model


def build_tinyvit_21m(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """Build TinyViT-21M model."""
    model = tiny_vit_21m_224(pretrained=pretrained)

    if num_classes != 1000:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model


# =============================================================================
# Teacher Model Builders (using timm)
# =============================================================================

def build_resnet50(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 teacher model using timm.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        ResNet-50 model
    """
    model = timm.create_model(
        'resnet50',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def build_vit_base(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """
    Build ViT-Base teacher model using timm.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        ViT-Base model (patch16, 224x224)
    """
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


# =============================================================================
# Unified Model Builder
# =============================================================================

def build_student(
    student_type: StudentType,
    num_classes: int = 100,
    pretrained: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Build student model.

    Args:
        student_type: Type of student model
        num_classes: Number of output classes
        pretrained: "imagenet" for pretrained weights, None for random init
        checkpoint_path: Path to local checkpoint (overrides pretrained)

    Returns:
        Student model
    """
    use_pretrained = pretrained == "imagenet"

    if student_type == StudentType.TINYVIT_5M:
        model = build_tinyvit_5m(num_classes=num_classes, pretrained=use_pretrained)
    elif student_type == StudentType.TINYVIT_11M:
        model = build_tinyvit_11m(num_classes=num_classes, pretrained=use_pretrained)
    else:
        raise ValueError(f"Unknown student type: {student_type}")

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, num_classes)

    return model


def build_teacher(
    teacher_type: TeacherType,
    num_classes: int = 100,
    pretrained: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Build teacher model.

    Args:
        teacher_type: Type of teacher model
        num_classes: Number of output classes
        pretrained: "imagenet" for pretrained weights
        checkpoint_path: Path to finetuned checkpoint

    Returns:
        Teacher model
    """
    use_pretrained = pretrained == "imagenet"

    if teacher_type == TeacherType.RESNET50:
        model = build_resnet50(num_classes=num_classes, pretrained=use_pretrained)
    elif teacher_type == TeacherType.VIT_BASE:
        model = build_vit_base(num_classes=num_classes, pretrained=use_pretrained)
    elif teacher_type == TeacherType.TINYVIT_21M:
        model = build_tinyvit_21m(num_classes=num_classes, pretrained=use_pretrained)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, num_classes)

    return model


def build_model_from_config(config) -> nn.Module:
    """
    Build model from experiment config.

    Args:
        config: ExperimentConfig instance

    Returns:
        Model (student or teacher depending on config.model_type)
    """
    num_classes = config.data.num_classes

    if config.model_type == "teacher":
        return build_teacher(
            teacher_type=config.teacher_type,
            num_classes=num_classes,
            pretrained=config.pretrained,
            checkpoint_path=config.distillation.teacher_checkpoint,
        )
    else:
        return build_student(
            student_type=config.student_type,
            num_classes=num_classes,
            pretrained=config.pretrained,
        )


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    num_classes: int = 100,
) -> nn.Module:
    """
    Load checkpoint into model, handling head size mismatch.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        num_classes: Expected number of classes

    Returns:
        Model with loaded weights
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Filter out mismatched head weights
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (from DDP)
        if k.startswith('module.'):
            k = k[7:]

        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped.append(f"{k}: {v.shape} vs {model_state[k].shape}")
        else:
            skipped.append(f"{k}: not in model")

    # Load filtered weights
    msg = model.load_state_dict(filtered_state, strict=False)

    print(f"  Loaded {len(filtered_state)} parameters")
    if skipped:
        print(f"  Skipped {len(skipped)} parameters (shape mismatch or missing)")
    if msg.missing_keys:
        head_keys = [k for k in msg.missing_keys if 'head' in k or 'fc' in k or 'classifier' in k]
        if head_keys:
            print(f"  Initialized new classification head: {head_keys}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    accuracy: float,
    path: str,
    config=None,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        accuracy: Best accuracy achieved
        path: Path to save checkpoint
        config: Optional config to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'accuracy': accuracy,
    }

    if config is not None:
        checkpoint['config'] = config.to_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


# =============================================================================
# Model Info Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_info(model: nn.Module, name: str = "Model"):
    """Print model information."""
    total, trainable = count_parameters(model)
    print(f"\n{name} Info:")
    print(f"  Total parameters: {total / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable / 1e6:.2f}M")
    print(f"  Architecture: {model.__class__.__name__}")


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all layers except classification head.

    Works with different model architectures by detecting
    common head naming patterns.
    """
    frozen = 0
    trainable = 0

    for name, param in model.named_parameters():
        # Common head patterns: head, fc, classifier
        is_head = any(h in name.lower() for h in ['head', 'fc', 'classifier'])

        if is_head:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    print(f"Frozen backbone: {frozen / 1e6:.2f}M parameters")
    print(f"Trainable head: {trainable / 1e6:.4f}M parameters")

    return model
