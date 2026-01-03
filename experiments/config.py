"""
Experiment Configuration for TinyViT CIFAR-100 Distillation Study

Naming Convention:
- B1, B2: Baselines (scratch, transfer)
- T1, T2, T3: Teacher preparation (resnet50, vit_base, tinyvit21m)
- D1, D2, D3: Distillation experiments (same-family, cnn, vit teacher)
- A1-A4: Ablations (topk sparsity, transfer+distill)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os
import json


# =============================================================================
# Enums for Model Types
# =============================================================================

class TeacherType(Enum):
    """Available teacher architectures."""
    RESNET50 = "resnet50"
    VIT_BASE = "vit_base_patch16_224"
    TINYVIT_21M = "tiny_vit_21m"


class StudentType(Enum):
    """Available student architectures."""
    TINYVIT_5M = "tiny_vit_5m"
    TINYVIT_11M = "tiny_vit_11m"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataConfig:
    """Data configuration - Fixed for all experiments."""
    dataset: str = "cifar100"
    data_path: str = "./data"
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    num_classes: int = 100


@dataclass
class AugmentationConfig:
    """Augmentation configuration - TinyViT defaults."""
    color_jitter: float = 0.4
    auto_augment: str = "rand-m9-mstd0.5-inc1"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    mixup: float = 0.8
    cutmix: float = 1.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration - TinyViT defaults."""
    epochs: int = 200
    base_lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    clip_grad: float = 5.0
    layer_lr_decay: float = 0.9
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class DistillationConfig:
    """Distillation configuration."""
    enabled: bool = False
    teacher_type: Optional[TeacherType] = None
    teacher_checkpoint: Optional[str] = None
    logits_path: Optional[str] = None
    topk: int = 100
    temperature: float = 1.0
    alpha: float = 0.5  # L = (1-alpha)*CE + alpha*KL


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    exp_id: str = "B1"
    exp_name: str = "baseline_scratch"
    description: str = ""
    seed: int = 42
    output_dir: str = "./output"

    # Model configuration
    model_type: str = "student"  # "student" or "teacher"
    student_type: StudentType = StudentType.TINYVIT_5M
    teacher_type: Optional[TeacherType] = None
    pretrained: Optional[str] = None  # None, "imagenet", or checkpoint path
    freeze_backbone: bool = False

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    # Hardware
    device: str = "cuda"
    amp_enabled: bool = True

    # Logging
    log_interval: int = 50
    save_freq: int = 20

    def get_output_path(self) -> str:
        """Get full output path for this experiment."""
        return os.path.join(self.output_dir, self.exp_id + "_" + self.exp_name)

    def get_checkpoint_path(self) -> str:
        """Get path for best checkpoint."""
        return os.path.join(self.get_output_path(), "best.pth")

    def get_logits_path(self) -> str:
        """Get path for saved logits."""
        teacher_name = self.teacher_type.value if self.teacher_type else "unknown"
        return os.path.join(self.output_dir, "logits", f"{teacher_name}_top{self.distillation.topk}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "exp_id": self.exp_id,
            "exp_name": self.exp_name,
            "description": self.description,
            "seed": self.seed,
            "model_type": self.model_type,
            "student_type": self.student_type.value if self.student_type else None,
            "teacher_type": self.teacher_type.value if self.teacher_type else None,
            "pretrained": self.pretrained,
            "data": {
                "dataset": self.data.dataset,
                "img_size": self.data.img_size,
                "batch_size": self.data.batch_size,
            },
            "training": {
                "epochs": self.training.epochs,
                "base_lr": self.training.base_lr,
                "weight_decay": self.training.weight_decay,
            },
            "distillation": {
                "enabled": self.distillation.enabled,
                "topk": self.distillation.topk,
                "alpha": self.distillation.alpha,
            },
        }

    def save(self, path: Optional[str] = None):
        """Save config to JSON file."""
        if path is None:
            path = os.path.join(self.get_output_path(), "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Pre-defined Experiment Configurations
# =============================================================================

def _create_baseline_scratch() -> ExperimentConfig:
    """B1: Train TinyViT-5M from scratch on CIFAR-100."""
    return ExperimentConfig(
        exp_id="B1",
        exp_name="baseline_scratch",
        description="Train TinyViT-5M from random initialization (lower bound)",
        student_type=StudentType.TINYVIT_5M,
        pretrained=None,
    )


def _create_baseline_transfer() -> ExperimentConfig:
    """B2: Finetune ImageNet-pretrained TinyViT-5M on CIFAR-100."""
    config = ExperimentConfig(
        exp_id="B2",
        exp_name="baseline_transfer",
        description="Finetune ImageNet-pretrained TinyViT-5M (missing baseline from paper)",
        student_type=StudentType.TINYVIT_5M,
        pretrained="imagenet",
    )
    # Adjusted for finetuning
    config.training.epochs = 100
    config.training.base_lr = 5e-4
    config.training.warmup_epochs = 5
    return config


def _create_teacher_resnet50() -> ExperimentConfig:
    """T1: Finetune ResNet-50 teacher on CIFAR-100."""
    config = ExperimentConfig(
        exp_id="T1",
        exp_name="teacher_resnet50",
        description="Finetune ImageNet-pretrained ResNet-50 on CIFAR-100",
        model_type="teacher",
        teacher_type=TeacherType.RESNET50,
        pretrained="imagenet",
    )
    config.training.epochs = 100
    config.training.base_lr = 1e-4
    config.training.warmup_epochs = 5
    config.data.batch_size = 64
    return config


def _create_teacher_vit_base() -> ExperimentConfig:
    """T2: Finetune ViT-Base teacher on CIFAR-100."""
    config = ExperimentConfig(
        exp_id="T2",
        exp_name="teacher_vit_base",
        description="Finetune ImageNet-pretrained ViT-Base on CIFAR-100",
        model_type="teacher",
        teacher_type=TeacherType.VIT_BASE,
        pretrained="imagenet",
    )
    config.training.epochs = 100
    config.training.base_lr = 1e-4
    config.training.warmup_epochs = 5
    config.data.batch_size = 32  # ViT needs smaller batch
    return config


def _create_teacher_tinyvit21m() -> ExperimentConfig:
    """T3: Finetune TinyViT-21M teacher on CIFAR-100."""
    config = ExperimentConfig(
        exp_id="T3",
        exp_name="teacher_tinyvit21m",
        description="Finetune ImageNet-pretrained TinyViT-21M on CIFAR-100",
        model_type="teacher",
        teacher_type=TeacherType.TINYVIT_21M,
        pretrained="imagenet",
    )
    config.training.epochs = 100
    config.training.base_lr = 1e-4
    config.training.warmup_epochs = 5
    config.data.batch_size = 48
    return config


def _create_distill_same_family(logits_path: str) -> ExperimentConfig:
    """D1: Distill from TinyViT-21M to TinyViT-5M (core paper reproduction)."""
    config = ExperimentConfig(
        exp_id="D1",
        exp_name="distill_same_family",
        description="TinyViT-21M -> TinyViT-5M distillation (reproduce paper)",
        student_type=StudentType.TINYVIT_5M,
        pretrained=None,  # Train from scratch with distillation
    )
    config.distillation.enabled = True
    config.distillation.teacher_type = TeacherType.TINYVIT_21M
    config.distillation.logits_path = logits_path
    config.distillation.topk = 100
    config.distillation.alpha = 0.5
    return config


def _create_distill_cnn_teacher(logits_path: str) -> ExperimentConfig:
    """D2: Distill from ResNet-50 to TinyViT-5M."""
    config = ExperimentConfig(
        exp_id="D2",
        exp_name="distill_cnn_teacher",
        description="ResNet-50 -> TinyViT-5M distillation (cross-architecture)",
        student_type=StudentType.TINYVIT_5M,
        pretrained=None,
    )
    config.distillation.enabled = True
    config.distillation.teacher_type = TeacherType.RESNET50
    config.distillation.logits_path = logits_path
    config.distillation.topk = 100
    config.distillation.alpha = 0.5
    return config


def _create_distill_vit_teacher(logits_path: str) -> ExperimentConfig:
    """D3: Distill from ViT-Base to TinyViT-5M."""
    config = ExperimentConfig(
        exp_id="D3",
        exp_name="distill_vit_teacher",
        description="ViT-Base -> TinyViT-5M distillation (different ViT family)",
        student_type=StudentType.TINYVIT_5M,
        pretrained=None,
    )
    config.distillation.enabled = True
    config.distillation.teacher_type = TeacherType.VIT_BASE
    config.distillation.logits_path = logits_path
    config.distillation.topk = 100
    config.distillation.alpha = 0.5
    return config


def _create_ablation_topk(topk: int, logits_path: str) -> ExperimentConfig:
    """A1-A3: Logit sparsity ablation."""
    config = ExperimentConfig(
        exp_id=f"A{['10', '50', '100'].index(str(topk)) + 1}" if topk in [10, 50, 100] else f"A_topk{topk}",
        exp_name=f"ablation_topk_{topk}",
        description=f"Logit sparsity ablation with K={topk}",
        student_type=StudentType.TINYVIT_5M,
        pretrained=None,
    )
    config.distillation.enabled = True
    config.distillation.teacher_type = TeacherType.TINYVIT_21M
    config.distillation.logits_path = logits_path
    config.distillation.topk = topk
    config.distillation.alpha = 0.5
    return config


def _create_ablation_transfer_distill(logits_path: str) -> ExperimentConfig:
    """A4: Transfer learning + Distillation combined."""
    config = ExperimentConfig(
        exp_id="A4",
        exp_name="ablation_transfer_distill",
        description="Pretrained TinyViT-5M + TinyViT-21M distillation (benefit stacking)",
        student_type=StudentType.TINYVIT_5M,
        pretrained="imagenet",  # Start from pretrained
    )
    config.distillation.enabled = True
    config.distillation.teacher_type = TeacherType.TINYVIT_21M
    config.distillation.logits_path = logits_path
    config.distillation.topk = 100
    config.distillation.alpha = 0.5
    # Adjusted for transfer + distill
    config.training.epochs = 100
    config.training.base_lr = 5e-4
    config.training.warmup_epochs = 5
    return config


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENTS = {
    # Baselines
    "B1": _create_baseline_scratch,
    "B2": _create_baseline_transfer,

    # Teachers
    "T1": _create_teacher_resnet50,
    "T2": _create_teacher_vit_base,
    "T3": _create_teacher_tinyvit21m,
}

# These need logits_path argument
DISTILLATION_EXPERIMENTS = {
    "D1": _create_distill_same_family,
    "D2": _create_distill_cnn_teacher,
    "D3": _create_distill_vit_teacher,
    "A4": _create_ablation_transfer_distill,
}


def get_experiment_config(exp_id: str, logits_path: Optional[str] = None, topk: Optional[int] = None) -> ExperimentConfig:
    """
    Get experiment configuration by ID.

    Args:
        exp_id: Experiment identifier (B1, B2, T1-T3, D1-D3, A1-A4)
        logits_path: Path to saved teacher logits (required for D1-D3, A1-A4)
        topk: Top-K value for ablation experiments A1-A3

    Returns:
        ExperimentConfig for the specified experiment
    """
    # Handle ablation experiments A1-A3
    if exp_id in ["A1", "A2", "A3"]:
        topk_map = {"A1": 10, "A2": 50, "A3": 100}
        if topk is None:
            topk = topk_map[exp_id]
        if logits_path is None:
            raise ValueError(f"logits_path required for experiment {exp_id}")
        return _create_ablation_topk(topk, logits_path)

    # Handle distillation experiments
    if exp_id in DISTILLATION_EXPERIMENTS:
        if logits_path is None:
            raise ValueError(f"logits_path required for experiment {exp_id}")
        return DISTILLATION_EXPERIMENTS[exp_id](logits_path)

    # Handle baseline and teacher experiments
    if exp_id in EXPERIMENTS:
        return EXPERIMENTS[exp_id]()

    raise ValueError(f"Unknown experiment ID: {exp_id}. Available: {list(EXPERIMENTS.keys()) + list(DISTILLATION_EXPERIMENTS.keys()) + ['A1', 'A2', 'A3']}")


def list_experiments() -> Dict[str, str]:
    """List all available experiments with descriptions."""
    experiments = {}

    # Baselines
    experiments["B1"] = "Baseline: Train TinyViT-5M from scratch"
    experiments["B2"] = "Baseline: Finetune pretrained TinyViT-5M"

    # Teachers
    experiments["T1"] = "Teacher: Finetune ResNet-50 on CIFAR-100"
    experiments["T2"] = "Teacher: Finetune ViT-Base on CIFAR-100"
    experiments["T3"] = "Teacher: Finetune TinyViT-21M on CIFAR-100"

    # Distillation
    experiments["D1"] = "Distill: TinyViT-21M -> TinyViT-5M (same-family)"
    experiments["D2"] = "Distill: ResNet-50 -> TinyViT-5M (CNN teacher)"
    experiments["D3"] = "Distill: ViT-Base -> TinyViT-5M (ViT teacher)"

    # Ablations
    experiments["A1"] = "Ablation: Top-K=10 logit sparsity"
    experiments["A2"] = "Ablation: Top-K=50 logit sparsity"
    experiments["A3"] = "Ablation: Top-K=100 logit sparsity"
    experiments["A4"] = "Ablation: Transfer + Distillation combined"

    return experiments
