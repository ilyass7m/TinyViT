"""
Unified Trainer for TinyViT CIFAR-100 Experiments

Supports:
- Standard training (baselines)
- Teacher finetuning
- Offline distillation training
"""

import os
import time
import datetime
import json
import numpy as np
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.data import Mixup
from torchvision import datasets, transforms

from .config import ExperimentConfig, TrainingConfig
from .models import build_model_from_config, save_checkpoint, print_model_info, freeze_backbone


# =============================================================================
# Data Loading
# =============================================================================

def build_transform(is_train: bool, config: ExperimentConfig):
    """Build data augmentation transforms."""
    img_size = config.data.img_size
    aug = config.augmentation

    if is_train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.RandomErasing(p=aug.reprob),
        ])
    else:
        # Validation transforms
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])

    return transform


def build_dataloader(config: ExperimentConfig, is_train: bool = True) -> Tuple[datasets.CIFAR100, DataLoader]:
    """
    Build CIFAR-100 dataloader.

    Args:
        config: Experiment configuration
        is_train: Whether to build training or validation loader

    Returns:
        Tuple of (dataset, dataloader)
    """
    transform = build_transform(is_train, config)

    dataset = datasets.CIFAR100(
        root=config.data.data_path,
        train=is_train,
        transform=transform,
        download=True
    )

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
# Optimizer and Scheduler
# =============================================================================

def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    head_lr_scale: float = 1.0
) -> torch.optim.Optimizer:
    """
    Build optimizer with optional head LR scaling.

    Args:
        model: Model to optimize
        config: Training configuration
        head_lr_scale: LR multiplier for classification head

    Returns:
        Optimizer
    """
    # Separate parameters
    decay_params = []
    no_decay_params = []
    head_decay_params = []
    head_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = any(h in name.lower() for h in ['head', 'fc', 'classifier'])
        no_decay = 'bias' in name or 'norm' in name or 'bn' in name

        if is_head:
            if no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
        else:
            if no_decay:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    base_lr = config.base_lr
    head_lr = base_lr * head_lr_scale

    param_groups = []

    if decay_params:
        param_groups.append({
            'params': decay_params,
            'lr': base_lr,
            'weight_decay': config.weight_decay,
        })
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0,
        })
    if head_decay_params:
        param_groups.append({
            'params': head_decay_params,
            'lr': head_lr,
            'weight_decay': config.weight_decay,
        })
    if head_no_decay_params:
        param_groups.append({
            'params': head_no_decay_params,
            'lr': head_lr,
            'weight_decay': 0.0,
        })

    if head_lr_scale != 1.0:
        print(f"Using different LR: backbone={base_lr:.2e}, head={head_lr:.2e} ({head_lr_scale}x)")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=config.betas,
        eps=config.eps
    )

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build cosine LR scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        steps_per_epoch: Number of steps per epoch

    Returns:
        LR scheduler
    """
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsLogger:
    """Log training metrics to file."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_acc5': [],
            'lr': [],
            'epoch': [],
        }
        os.makedirs(output_dir, exist_ok=True)

    def log(self, epoch: int, train_loss: float, train_acc: float,
            val_loss: float, val_acc: float, val_acc5: float, lr: float):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['val_acc5'].append(val_acc5)
        self.metrics['lr'].append(lr)

        # Save to file
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_best_acc(self) -> float:
        return max(self.metrics['val_acc']) if self.metrics['val_acc'] else 0.0


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    config: ExperimentConfig,
    device: torch.device,
    mixup_fn=None,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        config: Experiment config
        device: Device to use
        mixup_fn: Optional Mixup function

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup if enabled
        if mixup_fn is not None:
            images, targets_mixed = mixup_fn(images, targets)
            original_targets = targets
        else:
            targets_mixed = targets
            original_targets = targets

        # Forward pass
        with torch.cuda.amp.autocast(enabled=config.amp_enabled and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets_mixed)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if config.training.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)

        optimizer.step()
        scheduler.step()

        # Compute accuracy
        with torch.no_grad():
            acc1, _ = accuracy(outputs, original_targets, topk=(1, 5))

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc1.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if idx % config.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == 'cuda' else 0
            print(f'Epoch [{epoch}][{idx}/{len(loader)}] '
                  f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                  f'Acc: {acc_meter.val:.2f} ({acc_meter.avg:.2f}) '
                  f'LR: {lr:.6f} Mem: {mem:.0f}MB')

    epoch_time = time.time() - start
    print(f'Epoch {epoch} completed in {datetime.timedelta(seconds=int(epoch_time))}')

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Validate the model.

    Args:
        model: Model to evaluate
        loader: Validation dataloader
        config: Experiment config
        device: Device to use

    Returns:
        Tuple of (accuracy@1, accuracy@5, loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.amp_enabled and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)

    print(f'Validation: Acc@1 {acc1_meter.avg:.2f}% Acc@5 {acc5_meter.avg:.2f}% Loss {loss_meter.avg:.4f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: ExperimentConfig) -> Dict:
    """
    Main training function.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with training results
    """
    print("=" * 80)
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Description: {config.description}")
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
    train_dataset, train_loader = build_dataloader(config, is_train=True)
    val_dataset, val_loader = build_dataloader(config, is_train=False)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model_from_config(config)

    # Freeze backbone if requested
    if config.freeze_backbone:
        model = freeze_backbone(model)

    model = model.to(device)
    print_model_info(model, config.exp_name)

    # Build optimizer and scheduler
    head_lr_scale = 10.0 if config.pretrained else 1.0
    optimizer = build_optimizer(model, config.training, head_lr_scale)
    scheduler = build_scheduler(optimizer, config.training, len(train_loader))

    # Build criterion
    aug = config.augmentation
    if aug.mixup > 0 or aug.cutmix > 0:
        criterion = SoftTargetCrossEntropy()
    elif aug.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=aug.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Build mixup function
    mixup_fn = None
    if aug.mixup > 0 or aug.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=aug.mixup,
            cutmix_alpha=aug.cutmix,
            prob=aug.mixup_prob,
            switch_prob=aug.mixup_switch_prob,
            label_smoothing=aug.label_smoothing,
            num_classes=config.data.num_classes
        )

    # Initialize metrics logger
    metrics_logger = MetricsLogger(output_path)

    # Training loop
    print(f"\nStarting training for {config.training.epochs} epochs...")
    print("=" * 80)

    best_acc = 0.0

    for epoch in range(config.training.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, config, device, mixup_fn
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

        print(f"Epoch {epoch}: Val Acc@1 {val_acc:.2f}% | Best: {best_acc:.2f}%")
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

    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_path}")

    return {
        'best_acc': best_acc,
        'final_acc': val_acc,
        'output_path': output_path,
    }
