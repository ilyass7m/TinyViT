#!/usr/bin/env python
# --------------------------------------------------------
# Standalone Evaluation Script for TinyViT on CIFAR-100
# Simple script to evaluate model checkpoints without
# distributed training setup.
# --------------------------------------------------------

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import timm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tiny_vit import TinyViT


# Model configurations
TINYVIT_CONFIGS = {
    'tiny_vit_5m': {
        'embed_dims': [64, 128, 160, 320],
        'depths': [2, 2, 6, 2],
        'num_heads': [2, 4, 5, 10],
        'window_sizes': [7, 7, 14, 7],
    },
    'tiny_vit_11m': {
        'embed_dims': [64, 128, 256, 448],
        'depths': [2, 2, 6, 2],
        'num_heads': [2, 4, 8, 14],
        'window_sizes': [7, 7, 14, 7],
    },
    'tiny_vit_21m': {
        'embed_dims': [96, 192, 384, 576],
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 18],
        'window_sizes': [7, 7, 14, 7],
    },
}

# CIFAR-100 normalization
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model on CIFAR-100')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model-type', '-m', type=str, default='tiny_vit_21m',
                        choices=['tiny_vit_5m', 'tiny_vit_11m', 'tiny_vit_21m', 'vit_base'],
                        help='Model architecture type')
    parser.add_argument('--data-path', '-d', type=str, default='./data',
                        help='Path to CIFAR-100 data directory')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', '-w', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-class accuracy')
    return parser.parse_args()


def build_model(model_type: str, num_classes: int = 100) -> torch.nn.Module:
    """Build model based on type."""
    if model_type.startswith('tiny_vit'):
        cfg = TINYVIT_CONFIGS[model_type]
        model = TinyViT(
            img_size=224,
            num_classes=num_classes,
            embed_dims=cfg['embed_dims'],
            depths=cfg['depths'],
            num_heads=cfg['num_heads'],
            window_sizes=cfg['window_sizes'],
        )
    elif model_type == 'vit_base':
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict:
    """Load checkpoint into model."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    # Return metadata if available
    metadata = {}
    if 'epoch' in checkpoint:
        metadata['epoch'] = checkpoint['epoch']
    if 'max_accuracy' in checkpoint:
        metadata['max_accuracy'] = checkpoint['max_accuracy']
    if 'config' in checkpoint:
        metadata['config'] = checkpoint['config']

    return metadata


def get_test_transform(img_size: int = 224):
    """Get test transform for CIFAR-100."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
    verbose: bool = False
) -> dict:
    """
    Evaluate model on test set.

    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()

    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    total_loss = 0.0

    # Per-class tracking
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    # Confusion matrix tracking
    all_preds = []
    all_labels = []

    pbar = tqdm(data_loader, desc='Evaluating', unit='batch')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        # Compute accuracy
        _, pred_1 = outputs.topk(1, dim=1)
        _, pred_5 = outputs.topk(5, dim=1)

        pred_1 = pred_1.squeeze()
        correct_1 = (pred_1 == labels).sum().item()
        correct_5 = sum([labels[i] in pred_5[i] for i in range(labels.size(0))])

        total_correct_1 += correct_1
        total_correct_5 += correct_5
        total_samples += labels.size(0)

        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if pred_1[i].item() == label:
                class_correct[label] += 1

        # Store predictions for analysis
        all_preds.extend(pred_1.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Update progress bar
        running_acc = 100 * total_correct_1 / total_samples
        pbar.set_postfix({'Acc@1': f'{running_acc:.2f}%'})

    # Compute final metrics
    acc1 = 100 * total_correct_1 / total_samples
    acc5 = 100 * total_correct_5 / total_samples
    avg_loss = total_loss / total_samples

    # Per-class accuracy
    per_class_acc = 100 * class_correct / (class_total + 1e-8)

    results = {
        'acc1': acc1,
        'acc5': acc5,
        'loss': avg_loss,
        'total_samples': total_samples,
        'per_class_acc': per_class_acc.numpy(),
        'predictions': all_preds,
        'labels': all_labels,
    }

    return results


def main():
    args = parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print("=" * 60)

    # Build model
    print("Building model...")
    model = build_model(args.model_type, args.num_classes)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Load checkpoint
    metadata = load_checkpoint(model, args.checkpoint)

    if metadata:
        print(f"Checkpoint metadata:")
        if 'epoch' in metadata:
            print(f"  - Trained for {metadata['epoch']} epochs")
        if 'max_accuracy' in metadata:
            print(f"  - Best validation accuracy: {metadata['max_accuracy']:.2f}%")

    model = model.to(device)
    model.eval()

    # Build data loader
    print("\nLoading CIFAR-100 test set...")
    transform = get_test_transform(args.img_size)
    test_dataset = CIFAR100(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )

    print(f"Test set size: {len(test_dataset)}")
    print("=" * 60)

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=args.num_classes,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Top-1 Accuracy: {results['acc1']:.2f}%")
    print(f"Top-5 Accuracy: {results['acc5']:.2f}%")
    print(f"Average Loss:   {results['loss']:.4f}")
    print(f"Total Samples:  {results['total_samples']}")
    print("=" * 60)

    # Per-class accuracy statistics
    per_class = results['per_class_acc']
    print(f"\nPer-class Accuracy Statistics:")
    print(f"  Mean:   {per_class.mean():.2f}%")
    print(f"  Std:    {per_class.std():.2f}%")
    print(f"  Min:    {per_class.min():.2f}% (class {per_class.argmin()})")
    print(f"  Max:    {per_class.max():.2f}% (class {per_class.argmax()})")

    # Verbose: print all per-class accuracies
    if args.verbose:
        # CIFAR-100 class names
        cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        print("\nPer-class Accuracies:")
        print("-" * 40)

        # Sort by accuracy
        sorted_indices = per_class.argsort()

        print("\nWorst 10 classes:")
        for i in sorted_indices[:10]:
            print(f"  {cifar100_classes[i]:20s}: {per_class[i]:.1f}%")

        print("\nBest 10 classes:")
        for i in sorted_indices[-10:][::-1]:
            print(f"  {cifar100_classes[i]:20s}: {per_class[i]:.1f}%")

    return results


if __name__ == '__main__':
    main()
