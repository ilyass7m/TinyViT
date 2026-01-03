"""
TinyViT Training Script for CIFAR-100 (Single GPU)
Simplified version of main.py for local training without distributed setup.
Supports transfer learning from ImageNet pretrained weights.

Usage:
    # Train from scratch
    python train_cifar.py --cfg configs/cifar100/tiny_vit_5m_cifar100.yaml

    # Fine-tune from ImageNet pretrained weights (auto-download)
    python train_cifar.py --cfg configs/cifar100/tiny_vit_5m_cifar100.yaml --pretrained imagenet

    # Fine-tune from a local checkpoint
    python train_cifar.py --cfg configs/cifar100/tiny_vit_5m_cifar100.yaml --pretrained /path/to/checkpoint.pth

    # Fine-tune with frozen backbone (only train classifier head)
    python train_cifar.py --cfg configs/cifar100/tiny_vit_5m_cifar100.yaml --pretrained imagenet --freeze-backbone

    # Evaluation
    python train_cifar.py --cfg configs/cifar100/tiny_vit_5m_cifar100.yaml --eval --resume checkpoints/best.pth
"""

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.data import Mixup

from config import get_config
from models import build_model
from models.tiny_vit import tiny_vit_5m_224, tiny_vit_11m_224, tiny_vit_21m_224
from data import build_transform
from torchvision import datasets

# Wandb integration (optional)
try:
    import wandb
except ImportError:
    wandb = None


# Mapping from config model name to pretrained model loader
PRETRAINED_MODELS = {
    'tiny_vit_5m': tiny_vit_5m_224,
    'tiny_vit_11m': tiny_vit_11m_224,
    'tiny_vit_21m': tiny_vit_21m_224,
}


def parse_args():
    parser = argparse.ArgumentParser('TinyViT CIFAR Training')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--data-path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--eval', action='store_true', help='evaluation only')
    parser.add_argument('--output', type=str, default='./output_cifar', help='output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--opts', nargs='+', default=None, help='Modify config options')

    # Transfer learning arguments
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights or "imagenet" to auto-download ImageNet weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone and only train the classification head')
    parser.add_argument('--head-lr-scale', type=float, default=10.0,
                        help='Learning rate multiplier for classification head (default: 10x)')

    # For compatibility with config system
    parser.add_argument('--accumulation-steps', type=int, default=None)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--only-cpu', action='store_true')
    parser.add_argument('--tag', default='default')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--local_rank', type=int, default=None)

    # Wandb arguments
    parser.add_argument('--use-wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='tinyvit-cifar100',
                        help='Wandb project name (default: tinyvit-cifar100)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity (username or team name). If None, uses default entity.')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name. If None, auto-generated.')

    args = parser.parse_args()

    if args.device == 'cpu':
        args.only_cpu = True

    return args


def get_model_key(config):
    """Extract model key from config name (e.g., 'TinyViT-5M-CIFAR100' -> 'tiny_vit_5m')."""
    name = config.MODEL.NAME.lower()
    if '5m' in name:
        return 'tiny_vit_5m'
    elif '11m' in name:
        return 'tiny_vit_11m'
    elif '21m' in name:
        return 'tiny_vit_21m'
    return None


def load_pretrained_weights(model, pretrained_path, num_classes, device):
    """
    Load pretrained weights and adapt the classification head.

    Args:
        model: Target model (with correct num_classes)
        pretrained_path: Path to checkpoint or 'imagenet' for auto-download
        num_classes: Number of classes for the target task
        device: Device to load weights to

    Returns:
        model with loaded weights
    """
    if pretrained_path.lower() == 'imagenet':
        # Use timm's pretrained loading - weights will be downloaded automatically
        print("Loading ImageNet pretrained weights...")
        model_key = None
        for key in PRETRAINED_MODELS:
            if key in model.__class__.__name__.lower() or any(
                key.replace('_', '') in str(p) for p in [model.depths, model.num_classes]
            ):
                model_key = key
                break

        # Load pretrained model with ImageNet weights
        pretrained_model = None
        embed_dims = [p.shape[0] for p in model.parameters() if len(p.shape) == 1][:4]

        # Match model by embed_dims
        if embed_dims and embed_dims[0] == 64 and len(embed_dims) > 2:
            if embed_dims[2] == 160:
                pretrained_model = tiny_vit_5m_224(pretrained=True)
                print("  -> Matched TinyViT-5M")
            elif embed_dims[2] == 256:
                pretrained_model = tiny_vit_11m_224(pretrained=True)
                print("  -> Matched TinyViT-11M")
        elif embed_dims and embed_dims[0] == 96:
            pretrained_model = tiny_vit_21m_224(pretrained=True)
            print("  -> Matched TinyViT-21M")

        if pretrained_model is None:
            # Fallback: try to detect from model structure
            total_params = sum(p.numel() for p in model.parameters())
            if total_params < 7_000_000:
                pretrained_model = tiny_vit_5m_224(pretrained=True)
                print("  -> Matched TinyViT-5M (by param count)")
            elif total_params < 15_000_000:
                pretrained_model = tiny_vit_11m_224(pretrained=True)
                print("  -> Matched TinyViT-11M (by param count)")
            else:
                pretrained_model = tiny_vit_21m_224(pretrained=True)
                print("  -> Matched TinyViT-21M (by param count)")

        pretrained_state = pretrained_model.state_dict()
    else:
        # Load from local checkpoint
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model' in checkpoint:
            pretrained_state = checkpoint['model']
        else:
            pretrained_state = checkpoint

    # Get current model state
    model_state = model.state_dict()

    # Filter out head weights (different num_classes)
    pretrained_filtered = {}
    skipped_keys = []

    for k, v in pretrained_state.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                pretrained_filtered[k] = v
            else:
                skipped_keys.append(k)
                print(f"  Skipping {k}: shape mismatch ({v.shape} vs {model_state[k].shape})")
        elif k.startswith('head.'):
            skipped_keys.append(k)
        elif 'attention_bias_idxs' in k:
            # These are buffers, not parameters - skip silently
            pass
        else:
            skipped_keys.append(k)

    # Load filtered weights
    msg = model.load_state_dict(pretrained_filtered, strict=False)

    print(f"  Loaded {len(pretrained_filtered)} weight tensors")
    if msg.missing_keys:
        head_keys = [k for k in msg.missing_keys if 'head' in k]
        other_keys = [k for k in msg.missing_keys if 'head' not in k and 'attention_bias_idxs' not in k]
        if head_keys:
            print(f"  Initialized new classification head: {head_keys}")
        if other_keys:
            print(f"  Missing keys (initialized randomly): {other_keys}")

    return model


def freeze_backbone(model):
    """Freeze all layers except the classification head."""
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if 'head' in name or 'norm_head' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    print(f"Frozen backbone: {frozen_params/1e6:.2f}M parameters")
    print(f"Trainable head: {trainable_params/1e6:.4f}M parameters")

    return model


def build_cifar_loader(config, is_train=True):
    """Build CIFAR-100 data loader."""
    transform = build_transform(is_train, config)

    dataset = datasets.CIFAR100(
        root=config.DATA.DATA_PATH if config.DATA.DATA_PATH else './data',
        train=is_train,
        transform=transform,
        download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=is_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=is_train
    )

    return dataset, loader


def build_optimizer(config, model, head_lr_scale=1.0):
    """
    Build optimizer with optional higher LR for classification head.

    Args:
        config: Training config
        model: The model
        head_lr_scale: Multiplier for head learning rate (default: 1.0)
    """
    # Separate parameters into groups:
    # 1. Head parameters (higher LR)
    # 2. Backbone parameters with weight decay
    # 3. Backbone parameters without weight decay (bias, norm)

    head_decay_params = []
    head_no_decay_params = []
    backbone_decay_params = []
    backbone_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = 'head' in name or 'norm_head' in name
        no_decay = 'bias' in name or 'norm' in name or 'bn' in name

        if is_head:
            if no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
        else:
            if no_decay:
                backbone_no_decay_params.append(param)
            else:
                backbone_decay_params.append(param)

    base_lr = config.TRAIN.BASE_LR
    head_lr = base_lr * head_lr_scale

    param_groups = []

    # Backbone groups
    if backbone_decay_params:
        param_groups.append({
            'params': backbone_decay_params,
            'lr': base_lr,
            'weight_decay': config.TRAIN.WEIGHT_DECAY,
            'name': 'backbone_decay'
        })
    if backbone_no_decay_params:
        param_groups.append({
            'params': backbone_no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'backbone_no_decay'
        })

    # Head groups (potentially higher LR)
    if head_decay_params:
        param_groups.append({
            'params': head_decay_params,
            'lr': head_lr,
            'weight_decay': config.TRAIN.WEIGHT_DECAY,
            'name': 'head_decay'
        })
    if head_no_decay_params:
        param_groups.append({
            'params': head_no_decay_params,
            'lr': head_lr,
            'weight_decay': 0.0,
            'name': 'head_no_decay'
        })

    if head_lr_scale != 1.0:
        print(f"Using different LR: backbone={base_lr:.2e}, head={head_lr:.2e} ({head_lr_scale}x)")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,  # Default LR (overridden by param groups)
        betas=config.TRAIN.OPTIMIZER.BETAS,
        eps=config.TRAIN.OPTIMIZER.EPS
    )

    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """Build learning rate scheduler."""
    num_steps = config.TRAIN.EPOCHS * n_iter_per_epoch
    warmup_steps = config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (num_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


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


def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, config, device, mixup_fn=None, args=None):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    num_steps = len(loader)
    start = time.time()
    end = time.time()

    for idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup/cutmix if enabled
        if mixup_fn is not None:
            images, targets_mixed = mixup_fn(images, targets)
            original_targets = targets
        else:
            targets_mixed = targets
            original_targets = targets

        # Forward pass
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets_mixed)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.TRAIN.CLIP_GRAD > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)

        optimizer.step()
        scheduler.step()

        # Compute accuracy
        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Print progress
        if idx % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if device.type == 'cuda' else 0
            print(f'Train: [{epoch}][{idx}/{num_steps}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                  f'Acc@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) '
                  f'Acc@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f}) '
                  f'LR {lr:.6f} '
                  f'Mem {memory_used:.0f}MB')

            # Wandb logging
            if args is not None and args.use_wandb and wandb is not None:
                global_step = epoch * num_steps + idx
                wandb.log({
                    "train/loss": loss_meter.val,
                    "train/acc@1": acc1_meter.val,
                    "train/acc@5": acc5_meter.val,
                    "train/lr": lr,
                    "train/batch_time": batch_time.val,
                    "train/memory_mb": memory_used,
                }, step=global_step)

    epoch_time = time.time() - start
    print(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')

    return loss_meter.avg, acc1_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, config, device, epoch=None, args=None):
    """Evaluate the model."""
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 50 == 0:
            print(f'Test: [{idx}/{len(loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                  f'Acc@1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f}) '
                  f'Acc@5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f})')

    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    # Wandb logging for validation
    if args is not None and args.use_wandb and wandb is not None and epoch is not None:
        wandb.log({
            "val/loss": loss_meter.avg,
            "val/acc@1": acc1_meter.avg,
            "val/acc@5": acc5_meter.avg,
            "epoch": epoch,
        })

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def save_checkpoint(state, filename):
    """Save checkpoint."""
    torch.save(state, filename)
    print(f'Checkpoint saved to {filename}')


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # Load config
    config = get_config(args)

    # Override config with args
    config.defrost()
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    config.freeze()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        if wandb is None:
            print("Warning: wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
        else:
            # Build run name if not provided
            run_name = args.wandb_run_name
            if run_name is None:
                run_name = f"{config.MODEL.NAME}_bs{config.DATA.BATCH_SIZE}_lr{config.TRAIN.BASE_LR}"
                if args.pretrained:
                    run_name += "_pretrained"
                if args.freeze_backbone:
                    run_name += "_frozen"

            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    # Model config
                    "model_name": config.MODEL.NAME,
                    "model_type": config.MODEL.TYPE,
                    "num_classes": config.MODEL.NUM_CLASSES,
                    "img_size": config.DATA.IMG_SIZE,

                    # Training config
                    "epochs": config.TRAIN.EPOCHS,
                    "batch_size": config.DATA.BATCH_SIZE,
                    "base_lr": config.TRAIN.BASE_LR,
                    "weight_decay": config.TRAIN.WEIGHT_DECAY,
                    "warmup_epochs": config.TRAIN.WARMUP_EPOCHS,
                    "clip_grad": config.TRAIN.CLIP_GRAD,

                    # Augmentation config
                    "mixup": config.AUG.MIXUP,
                    "cutmix": config.AUG.CUTMIX,
                    "label_smoothing": config.MODEL.LABEL_SMOOTHING,

                    # Transfer learning
                    "pretrained": args.pretrained,
                    "freeze_backbone": args.freeze_backbone,
                    "head_lr_scale": args.head_lr_scale if args.pretrained else 1.0,

                    # Other
                    "amp_enabled": config.AMP_ENABLE,
                    "seed": args.seed,
                },
                dir=args.output,
                resume="allow",
            )
            print(f"Wandb initialized: {wandb.run.url}")

    # Build data loaders
    print('Building CIFAR-100 data loaders...')
    train_dataset, train_loader = build_cifar_loader(config, is_train=True)
    val_dataset, val_loader = build_cifar_loader(config, is_train=False)
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # Build model
    print(f'Building model: {config.MODEL.NAME}')
    model = build_model(config)

    # Load pretrained weights if specified
    if args.pretrained:
        print('=' * 60)
        print('TRANSFER LEARNING MODE')
        print('=' * 60)
        model = load_pretrained_weights(
            model,
            args.pretrained,
            config.MODEL.NUM_CLASSES,
            device
        )

    # Freeze backbone if requested
    if args.freeze_backbone:
        print('Freezing backbone layers...')
        model = freeze_backbone(model)

    model = model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n_total / 1e6:.2f}M')
    print(f'Trainable parameters: {n_parameters / 1e6:.2f}M')

    # Build optimizer and scheduler
    # Use higher LR for head when fine-tuning
    head_lr_scale = args.head_lr_scale if args.pretrained else 1.0
    optimizer = build_optimizer(config, model, head_lr_scale=head_lr_scale)
    scheduler = build_scheduler(config, optimizer, len(train_loader))

    # Build criterion
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    # Build mixup function
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    # Resume from checkpoint
    start_epoch = 0
    max_accuracy = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                max_accuracy = checkpoint.get('max_accuracy', 0.0)
            print(f'Loaded checkpoint (epoch {checkpoint.get("epoch", "N/A")})')
        else:
            print(f'No checkpoint found at {args.resume}')

    # Evaluation only
    if args.eval:
        print('Running evaluation...')
        acc1, acc5, loss = validate(model, val_loader, nn.CrossEntropyLoss(), config, device)
        print(f'Validation Results: Acc@1 {acc1:.2f}% Acc@5 {acc5:.2f}% Loss {loss:.4f}')
        return
    
    # Training loop
    print(f'Starting training for {config.TRAIN.EPOCHS} epochs...')
    print('=' * 80)

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, config, device, mixup_fn, args=args
        )

        # Validate
        acc1, acc5, val_loss = validate(model, val_loader, nn.CrossEntropyLoss(), config, device,
                                        epoch=epoch, args=args)

        # Save best checkpoint
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)

        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Acc@1 {acc1:.2f}%, Best Acc@1 {max_accuracy:.2f}%')
        print('=' * 80)

        # Save checkpoint
        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'max_accuracy': max_accuracy,
                'config': config
            }, os.path.join(args.output, 'best.pth'))

        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'max_accuracy': max_accuracy,
                'config': config
            }, os.path.join(args.output, f'ckpt_epoch_{epoch}.pth'))

    print(f'Training complete! Best accuracy: {max_accuracy:.2f}%')

    # Wandb final summary
    if args.use_wandb and wandb is not None:
        wandb.run.summary['best_acc@1'] = max_accuracy
        wandb.run.summary['final_epoch'] = config.TRAIN.EPOCHS - 1
        wandb.finish()
        print("Wandb run finished.")


if __name__ == '__main__':
    main()
