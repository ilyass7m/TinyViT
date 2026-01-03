"""
Offline Distillation Utilities for TinyViT CIFAR-100 Experiments

Provides:
- Teacher logits saving
- Distillation dataset wrapper
- Distillation training loop with KL divergence loss
"""

import os
import time
import datetime
import json
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn

from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.data import Mixup
from torchvision import datasets, transforms

from .config import ExperimentConfig, TeacherType
from .models import build_teacher, build_student, save_checkpoint, print_model_info
from .trainer import (
    build_transform, build_dataloader, build_optimizer, build_scheduler,
    AverageMeter, MetricsLogger, validate
)


# =============================================================================
# Logits Storage
# =============================================================================

class LogitsDataset(Dataset):
    """
    Dataset wrapper that loads pre-saved teacher logits.

    The logits are stored as sparse top-k format:
    - indices: int16 array of shape (N, K)
    - values: float16 array of shape (N, K)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        logits_path: str,
        topk: int = 100,
        num_classes: int = 100,
    ):
        """
        Args:
            base_dataset: Original CIFAR-100 dataset
            logits_path: Path to saved logits directory
            topk: Number of top logits saved
            num_classes: Total number of classes
        """
        self.dataset = base_dataset
        self.logits_path = logits_path
        self.topk = topk
        self.num_classes = num_classes

        # Load logits
        self.logits_indices, self.logits_values = self._load_logits()
        print(f"Loaded logits for {len(self.logits_indices)} samples (top-{topk})")

    def _load_logits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load saved logits from disk."""
        indices_path = os.path.join(self.logits_path, 'indices.npy')
        values_path = os.path.join(self.logits_path, 'values.npy')

        if not os.path.exists(indices_path) or not os.path.exists(values_path):
            raise FileNotFoundError(
                f"Logits not found at {self.logits_path}. "
                f"Run save_teacher_logits() first."
            )

        indices = np.load(indices_path)
        values = np.load(values_path)

        return indices, values

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns:
            image: Transformed image tensor
            target: Original class label
            soft_target: Dense soft target distribution from teacher
        """
        image, target = self.dataset[idx]

        # Reconstruct dense soft target from sparse format
        indices = self.logits_indices[idx]
        values = self.logits_values[idx]

        soft_target = np.zeros(self.num_classes, dtype=np.float32)
        soft_target[indices] = values

        return image, target, torch.from_numpy(soft_target)


def build_distill_dataloader(
    config: ExperimentConfig,
    is_train: bool = True
) -> Tuple[Dataset, DataLoader]:
    """
    Build dataloader with distillation logits.

    Args:
        config: Experiment configuration
        is_train: Whether training or validation

    Returns:
        Tuple of (dataset, dataloader)
    """
    transform = build_transform(is_train, config)

    base_dataset = datasets.CIFAR100(
        root=config.data.data_path,
        train=is_train,
        transform=transform,
        download=True
    )

    if is_train and config.distillation.enabled and config.distillation.logits_path:
        dataset = LogitsDataset(
            base_dataset=base_dataset,
            logits_path=config.distillation.logits_path,
            topk=config.distillation.topk,
            num_classes=config.data.num_classes,
        )
    else:
        dataset = base_dataset

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=is_train,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=is_train
    )

    return dataset, loader


# =============================================================================
# Teacher Logits Saving
# =============================================================================

@torch.no_grad()
def save_teacher_logits(
    config: ExperimentConfig,
    teacher_checkpoint: str,
) -> str:
    """
    Save teacher logits for offline distillation.

    Args:
        config: Experiment configuration
        teacher_checkpoint: Path to finetuned teacher checkpoint

    Returns:
        Path where logits are saved
    """
    print("=" * 80)
    print(f"Saving Teacher Logits")
    print(f"Teacher: {config.teacher_type.value}")
    print(f"Top-K: {config.distillation.topk}")
    print("=" * 80)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    topk = config.distillation.topk

    # Build teacher model
    print("\nLoading teacher model...")
    teacher = build_teacher(
        teacher_type=config.teacher_type,
        num_classes=config.data.num_classes,
        pretrained=None,  # We'll load from checkpoint
    )

    # Load checkpoint
    checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        teacher.load_state_dict(checkpoint['model'])
    else:
        teacher.load_state_dict(checkpoint)

    teacher = teacher.to(device)
    teacher.eval()
    print_model_info(teacher, "Teacher")

    # Build dataloader (without augmentation for consistent logits)
    print("\nBuilding dataloader...")
    transform = transforms.Compose([
        transforms.Resize((config.data.img_size, config.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    dataset = datasets.CIFAR100(
        root=config.data.data_path,
        train=True,
        transform=transform,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size * 2,  # Larger batch for inference
        shuffle=False,  # Important: keep order for indexing
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # Collect logits
    print(f"\nProcessing {len(dataset)} samples...")
    all_indices = []
    all_values = []
    acc_meter = AverageMeter()

    start_time = time.time()

    for idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.amp_enabled):
            outputs = teacher(images)

        # Compute accuracy
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        acc_meter.update(acc1.item(), images.size(0))

        # Get soft probabilities
        probs = F.softmax(outputs, dim=-1)

        # Get top-k
        values, indices = probs.topk(k=topk, dim=-1, largest=True, sorted=True)

        # Convert to numpy and store
        all_indices.append(indices.cpu().numpy().astype(np.int16))
        all_values.append(values.cpu().numpy().astype(np.float16))

        if idx % 50 == 0:
            print(f"  Processed {idx * loader.batch_size}/{len(dataset)} samples "
                  f"| Teacher Acc: {acc_meter.avg:.2f}%")

    # Concatenate all
    all_indices = np.concatenate(all_indices, axis=0)
    all_values = np.concatenate(all_values, axis=0)

    print(f"\nTeacher accuracy on training set: {acc_meter.avg:.2f}%")
    print(f"Logits shape: indices={all_indices.shape}, values={all_values.shape}")

    # Save to disk
    output_path = config.get_logits_path()
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, 'indices.npy'), all_indices)
    np.save(os.path.join(output_path, 'values.npy'), all_values)

    # Save metadata
    metadata = {
        'teacher_type': config.teacher_type.value,
        'teacher_checkpoint': teacher_checkpoint,
        'topk': topk,
        'num_samples': len(dataset),
        'teacher_train_acc': acc_meter.avg,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nLogits saved to: {output_path}")
    print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed))}")

    return output_path


# =============================================================================
# Distillation Loss
# =============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    L = (1 - alpha) * CE(student, labels) + alpha * KL(student, teacher)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 1.0,
        num_classes: int = 100,
    ):
        """
        Args:
            alpha: Weight for distillation loss (0 = CE only, 1 = KL only)
            temperature: Temperature for softmax
            num_classes: Number of classes
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Raw logits from student (B, C)
            targets: Ground truth labels (B,)
            teacher_probs: Soft probabilities from teacher (B, C)

        Returns:
            Combined loss and dict of individual losses
        """
        # Cross-entropy with hard labels
        ce_loss = self.ce_loss(student_logits, targets)

        # KL divergence with soft labels
        # KL(P || Q) = sum(P * log(P / Q))
        # For numerical stability, use log_softmax
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs_temp = teacher_probs  # Already probabilities

        # Avoid log(0) by clamping
        teacher_probs_temp = teacher_probs_temp.clamp(min=1e-8)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs_temp,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss

        loss_dict = {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': loss.item(),
        }

        return loss, loss_dict


# =============================================================================
# Distillation Training Loop
# =============================================================================

def train_one_epoch_distill(
    model: nn.Module,
    loader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Train for one epoch with distillation.

    Args:
        model: Student model
        loader: Training dataloader with teacher logits
        criterion: Distillation loss
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        config: Experiment config
        device: Device

    Returns:
        Tuple of (avg_loss, avg_accuracy, loss_breakdown)
    """
    model.train()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    kl_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (images, targets, teacher_probs) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_probs = teacher_probs.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=config.amp_enabled and device.type == 'cuda'):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets, teacher_probs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if config.training.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)

        optimizer.step()
        scheduler.step()

        # Compute accuracy
        with torch.no_grad():
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        ce_meter.update(loss_dict['ce_loss'], batch_size)
        kl_meter.update(loss_dict['kl_loss'], batch_size)
        acc_meter.update(acc1.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if idx % config.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == 'cuda' else 0
            print(f'Epoch [{epoch}][{idx}/{len(loader)}] '
                  f'Loss: {loss_meter.val:.4f} (CE:{ce_meter.val:.4f} KL:{kl_meter.val:.4f}) '
                  f'Acc: {acc_meter.val:.2f} ({acc_meter.avg:.2f}) '
                  f'LR: {lr:.6f} Mem: {mem:.0f}MB')

    epoch_time = time.time() - start
    print(f'Epoch {epoch} completed in {datetime.timedelta(seconds=int(epoch_time))}')

    loss_breakdown = {
        'ce_loss': ce_meter.avg,
        'kl_loss': kl_meter.avg,
        'total_loss': loss_meter.avg,
    }

    return loss_meter.avg, acc_meter.avg, loss_breakdown


# =============================================================================
# Main Distillation Training Function
# =============================================================================

def train_with_distillation(config: ExperimentConfig) -> Dict:
    """
    Main training function with offline distillation.

    Args:
        config: Experiment configuration (must have distillation enabled)

    Returns:
        Dictionary with training results
    """
    assert config.distillation.enabled, "Distillation must be enabled"
    assert config.distillation.logits_path, "Logits path must be specified"

    print("=" * 80)
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Description: {config.description}")
    print(f"Distillation: {config.distillation.teacher_type.value} -> {config.student_type.value}")
    print(f"Alpha: {config.distillation.alpha}, Top-K: {config.distillation.topk}")
    print("=" * 80)

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = config.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    config.save()

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_dataset, train_loader = build_distill_dataloader(config, is_train=True)
    _, val_loader = build_dataloader(config, is_train=False)
    print(f"Training samples: {len(train_dataset)}")

    # Build student model
    print("\nBuilding student model...")
    model = build_student(
        student_type=config.student_type,
        num_classes=config.data.num_classes,
        pretrained=config.pretrained,
    )
    model = model.to(device)
    print_model_info(model, "Student")

    # Build optimizer and scheduler
    head_lr_scale = 10.0 if config.pretrained else 1.0
    optimizer = build_optimizer(model, config.training, head_lr_scale)
    scheduler = build_scheduler(optimizer, config.training, len(train_loader))

    # Build distillation criterion
    criterion = DistillationLoss(
        alpha=config.distillation.alpha,
        temperature=config.distillation.temperature,
        num_classes=config.data.num_classes,
    )

    # Initialize metrics logger
    metrics_logger = MetricsLogger(output_path)

    # Training loop
    print(f"\nStarting distillation training for {config.training.epochs} epochs...")
    print("=" * 80)

    best_acc = 0.0

    for epoch in range(config.training.epochs):
        # Train with distillation
        train_loss, train_acc, loss_breakdown = train_one_epoch_distill(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, config, device
        )

        # Validate
        val_acc, val_acc5, val_loss = validate(model, val_loader, config, device)

        # Log metrics
        lr = optimizer.param_groups[0]['lr']
        metrics_logger.log(epoch, train_loss, train_acc, val_loss, val_acc, val_acc5, lr)

        # Save best checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(output_path, 'best.pth'),
                config
            )

        print(f"Epoch {epoch}: Val Acc@1 {val_acc:.2f}% | Best: {best_acc:.2f}% "
              f"| CE: {loss_breakdown['ce_loss']:.4f} KL: {loss_breakdown['kl_loss']:.4f}")
        print("=" * 80)

        # Save periodic checkpoint
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(output_path, f'checkpoint_epoch{epoch}.pth'),
                config
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, config.training.epochs - 1, val_acc,
        os.path.join(output_path, 'final.pth'),
        config
    )

    print(f"\nDistillation training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_path}")

    return {
        'best_acc': best_acc,
        'final_acc': val_acc,
        'output_path': output_path,
    }
