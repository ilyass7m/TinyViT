#!/usr/bin/env python
# --------------------------------------------------------
# Validate Saved Teacher Logits
# Quick script to verify that saved logits are correct
# by computing accuracy from the argmax of logits.
# --------------------------------------------------------

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Validate saved teacher logits')
    parser.add_argument('--logits-path', '-l', type=str, required=True,
                        help='Path to directory containing saved logits (.pth files)')
    parser.add_argument('--data-path', '-d', type=str, default='./data',
                        help='Path to CIFAR-100 data directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Which split to validate (train or val)')
    parser.add_argument('--topk', '-k', type=int, default=None,
                        help='If logits are sparse (TopK), specify K value')
    return parser.parse_args()


def load_logits_file(filepath: str):
    """Load a single logits file."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    return data


def main():
    args = parse_args()

    logits_path = Path(args.logits_path)

    if not logits_path.exists():
        print(f"Error: Logits path does not exist: {logits_path}")
        return

    # Find all logits files
    if args.split == 'train':
        logits_files = sorted(logits_path.glob('train_logits_*.pth'))
        is_train = True
    else:
        logits_files = sorted(logits_path.glob('val_logits_*.pth'))
        is_train = False

    if not logits_files:
        # Try alternative naming
        logits_files = sorted(logits_path.glob('*.pth'))
        print(f"Found {len(logits_files)} .pth files in {logits_path}")

    if not logits_files:
        print(f"No logits files found in {logits_path}")
        print("Expected files like: train_logits_0.pth, train_logits_1.pth, ...")
        return

    print(f"Found {len(logits_files)} logits files")
    print(f"Validating {args.split} split...")
    print("=" * 60)

    # Load CIFAR-100 to get ground truth labels
    transform = transforms.ToTensor()  # Just need labels, transform doesn't matter
    dataset = CIFAR100(
        root=args.data_path,
        train=is_train,
        download=True,
        transform=transform
    )

    print(f"Dataset size: {len(dataset)}")

    # Collect all predictions
    all_preds = []
    all_labels = []
    all_top5_correct = []

    total_samples_in_logits = 0

    for filepath in tqdm(logits_files, desc='Loading logits'):
        data = load_logits_file(filepath)

        # Handle different formats
        if isinstance(data, dict):
            if 'logits' in data:
                logits = data['logits']
                indices = data.get('indices', None)
                labels = data.get('labels', None)
            elif 'teacher_logits' in data:
                logits = data['teacher_logits']
                indices = data.get('teacher_indices', None)
                labels = data.get('labels', None)
            else:
                print(f"Unknown dict format. Keys: {data.keys()}")
                continue
        elif isinstance(data, (list, tuple)):
            # Format: [(logits, indices, label), ...]
            logits_list = []
            indices_list = []
            labels_list = []
            for item in data:
                if len(item) == 3:
                    l, idx, lab = item
                    logits_list.append(l)
                    indices_list.append(idx)
                    labels_list.append(lab)
                elif len(item) == 2:
                    l, lab = item
                    logits_list.append(l)
                    labels_list.append(lab)
            logits = torch.stack(logits_list) if logits_list else None
            indices = torch.stack(indices_list) if indices_list else None
            labels = torch.tensor(labels_list) if labels_list else None
        else:
            logits = data
            indices = None
            labels = None

        if logits is None:
            continue

        # Convert to tensor if needed
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)

        batch_size = logits.shape[0]
        total_samples_in_logits += batch_size

        # Handle sparse logits (TopK format)
        if indices is not None:
            # Sparse format: logits are values, indices are class indices
            # Prediction is the class with highest logit value
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices)

            # Get argmax of logit values
            max_idx = logits.argmax(dim=-1)  # Index into the K values

            # Map back to actual class indices
            preds = indices[torch.arange(batch_size), max_idx]

            # Top-5: check if true label is in top-5 indices
            if labels is not None:
                for i in range(batch_size):
                    top5_indices = indices[i, :5] if indices.shape[-1] >= 5 else indices[i]
                    label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                    all_top5_correct.append(label in top5_indices.tolist())
        else:
            # Dense format: full logits
            preds = logits.argmax(dim=-1)

            # Top-5
            if labels is not None:
                _, top5 = logits.topk(5, dim=-1)
                for i in range(batch_size):
                    label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                    all_top5_correct.append(label in top5[i].tolist())

        all_preds.extend(preds.tolist())

        # Use labels from logits file if available, else use dataset order
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels)

    # If no labels in logits, get from dataset
    if not all_labels:
        print("\nNo labels found in logits files, using dataset labels...")
        all_labels = [dataset[i][1] for i in range(min(len(all_preds), len(dataset)))]

    # Compute accuracy
    print(f"\nTotal samples in logits: {total_samples_in_logits}")
    print(f"Predictions collected: {len(all_preds)}")
    print(f"Labels collected: {len(all_labels)}")

    if len(all_preds) != len(all_labels):
        print(f"\nWarning: Mismatch between predictions ({len(all_preds)}) and labels ({len(all_labels)})")
        min_len = min(len(all_preds), len(all_labels))
        all_preds = all_preds[:min_len]
        all_labels = all_labels[:min_len]

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100 * correct / len(all_labels)

    print("\n" + "=" * 60)
    print("LOGITS VALIDATION RESULTS")
    print("=" * 60)
    print(f"Top-1 Accuracy: {accuracy:.2f}%")

    if all_top5_correct:
        top5_acc = 100 * sum(all_top5_correct) / len(all_top5_correct)
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    print(f"Samples: {len(all_labels)}")
    print("=" * 60)

    # Interpretation
    print("\nInterpretation:")
    if accuracy > 85:
        print("  [OK] Logits appear to be from a well-trained model")
    elif accuracy > 50:
        print("  [WARNING] Accuracy is moderate - model may be undertrained")
    elif accuracy > 5:
        print("  [WARNING] Accuracy is low - check if correct checkpoint was used")
    else:
        print("  [ERROR] Accuracy is very low - logits may be random or corrupted")


if __name__ == '__main__':
    main()
