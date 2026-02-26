#!/usr/bin/env python
"""
Validate saved binary logits by computing accuracy from TopK predictions.

Binary format per sample (from save_logits.py):
- seed: int32 (4 bytes) - random seed for augmentation
- indices: K × int16 (K×2 bytes) - top-K class indices
- values: K × float16 (K×2 bytes) - top-K softmax probabilities

Total bytes per sample: 4 + K×2 + K×2 = 4 + K×4
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import CIFAR100


def parse_args():
    parser = argparse.ArgumentParser(description='Validate binary logits')
    parser.add_argument('--logits-dir', '-l', type=str, required=True,
                        help='Path to logits directory (e.g., output/logits/vit_base_top50)')
    parser.add_argument('--data-path', '-d', type=str, default='./data',
                        help='Path to CIFAR-100 data')
    parser.add_argument('--topk', '-k', type=int, default=None,
                        help='TopK value (auto-detected if not specified)')
    parser.add_argument('--epoch', '-e', type=int, default=0,
                        help='Epoch number to validate (default: 0)')
    return parser.parse_args()


def detect_topk(file_size: int, num_samples: int) -> int:
    """
    Detect TopK from file size.

    Format: seed(4) + indices(K×2) + values(K×2) = 4 + K×4 bytes per sample
    So: file_size = num_samples × (4 + K×4)
    Therefore: K = (file_size/num_samples - 4) / 4
    """
    bytes_per_sample = file_size / num_samples
    k = (bytes_per_sample - 4) / 4
    return int(k)


def load_binary_logits(values_path: str, keys_path: str, topk: int):
    """
    Load binary logits file.

    Format per sample:
    - seed: int32 (4 bytes)
    - indices: K × int16 (K×2 bytes)
    - values: K × float16 (K×2 bytes)
    """
    # Load keys (image paths/identifiers)
    with open(keys_path, 'r') as f:
        keys = [line.strip() for line in f.readlines()]

    num_samples = len(keys)

    # Load binary data
    with open(values_path, 'rb') as f:
        data = f.read()

    # Auto-detect topk if needed
    detected_k = detect_topk(len(data), num_samples)
    if topk is None or detected_k != topk:
        print(f"  Auto-detected TopK = {detected_k} (specified: {topk})")
        topk = detected_k

    # Bytes per sample: 4 (seed) + K×2 (indices) + K×2 (values)
    bytes_per_sample = 4 + topk * 2 + topk * 2

    expected_bytes = num_samples * bytes_per_sample
    if len(data) != expected_bytes:
        print(f"  Warning: Expected {expected_bytes} bytes, got {len(data)}")

    # Parse binary data
    seeds_list = []
    indices_list = []
    values_list = []

    offset = 0
    for i in range(num_samples):
        # Read seed (int32, 4 bytes)
        seed = np.frombuffer(data[offset:offset + 4], dtype=np.int32)[0]
        offset += 4

        # Read indices (K × int16)
        indices = np.frombuffer(data[offset:offset + topk * 2], dtype=np.int16).copy()
        offset += topk * 2

        # Read values (K × float16)
        values = np.frombuffer(data[offset:offset + topk * 2], dtype=np.float16).copy()
        offset += topk * 2

        seeds_list.append(seed)
        indices_list.append(indices)
        values_list.append(values)

    return keys, np.array(seeds_list), np.array(indices_list), np.array(values_list), topk


def main():
    args = parse_args()

    logits_dir = Path(args.logits_dir)

    if not logits_dir.exists():
        print(f"Error: Directory not found: {logits_dir}")
        return

    # Find epoch directory
    if args.topk is None:
        epoch_dirs = list(logits_dir.glob(f"logits_top*_epoch{args.epoch}"))
        if epoch_dirs:
            epoch_dir = epoch_dirs[0]
            # Extract topk from directory name
            dirname = epoch_dir.name
            topk_str = dirname.split('_')[1].replace('top', '')
            args.topk = int(topk_str)
            print(f"Found epoch directory: {epoch_dir}")
            print(f"TopK from dirname = {args.topk}")
        else:
            print(f"No epoch directory found for epoch {args.epoch}")
            available = list(logits_dir.glob("logits_*"))
            print(f"Available: {[d.name for d in available]}")
            return
    else:
        epoch_dir = logits_dir / f"logits_top{args.topk}_epoch{args.epoch}"

    if not epoch_dir.exists():
        print(f"Error: Epoch directory not found: {epoch_dir}")
        return

    # Find all rank files
    values_files = sorted(epoch_dir.glob("rank*-values.bin"))

    if not values_files:
        print(f"No values.bin files found in {epoch_dir}")
        return

    print(f"Found {len(values_files)} rank files")
    print("=" * 60)

    # Load CIFAR-100 training set to get labels
    dataset = CIFAR100(root=args.data_path, train=True, download=True)
    print(f"CIFAR-100 training set: {len(dataset)} samples")

    # Build a mapping from image key to label
    # Keys are typically integer indices as strings

    # Load and validate logits from each rank
    all_keys = []
    all_predictions = []
    all_indices = []
    all_values = []
    actual_topk = args.topk

    for values_path in values_files:
        rank_name = values_path.stem.replace('-values', '')
        keys_path = epoch_dir / f"{rank_name}-keys.txt"

        if not keys_path.exists():
            print(f"Warning: Keys file not found: {keys_path}")
            continue

        print(f"\nLoading {rank_name}...")

        keys, seeds, indices, values, detected_k = load_binary_logits(
            str(values_path), str(keys_path), args.topk
        )
        actual_topk = detected_k

        print(f"  Samples: {len(keys)}")
        print(f"  Indices shape: {indices.shape}")
        print(f"  Values shape: {values.shape}")

        # Show sample data
        print(f"  Sample key: {keys[0]}")
        print(f"  Sample indices[0][:5]: {indices[0][:5]}")
        print(f"  Sample values[0][:5]: {values[0][:5]}")

        all_keys.extend(keys)
        all_indices.append(indices)
        all_values.append(values)

        # Get predictions (class with highest probability)
        for i in range(len(keys)):
            max_idx = values[i].argmax()
            pred_class = indices[i][max_idx]
            all_predictions.append(pred_class)

    # Concatenate arrays
    all_indices = np.concatenate(all_indices, axis=0)
    all_values = np.concatenate(all_values, axis=0)

    print(f"\nTotal samples loaded: {len(all_keys)}")

    # Get ground truth labels
    # Keys should be integer indices
    all_labels = []
    for key in all_keys:
        try:
            idx = int(key)
            if idx < len(dataset.targets):
                all_labels.append(dataset.targets[idx])
            else:
                all_labels.append(-1)
        except ValueError:
            # Key is not an integer, try to match differently
            all_labels.append(-1)

    valid_mask = [l >= 0 for l in all_labels]
    valid_count = sum(valid_mask)

    if valid_count == 0:
        print("\nCould not match keys to dataset labels.")
        print("Sample keys:", all_keys[:5])
        return

    print(f"Matched {valid_count}/{len(all_labels)} samples to labels")

    # Compute accuracy
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for i, (pred, label, valid) in enumerate(zip(all_predictions, all_labels, valid_mask)):
        if not valid:
            continue
        total += 1
        if pred == label:
            correct_top1 += 1
        # Top-5: check if label is in top-5 indices
        top5_indices = all_indices[i][:5]
        if label in top5_indices:
            correct_top5 += 1

    acc1 = 100 * correct_top1 / total if total > 0 else 0
    acc5 = 100 * correct_top5 / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("LOGITS VALIDATION RESULTS")
    print("=" * 60)
    print(f"Logits directory: {logits_dir}")
    print(f"TopK: {actual_topk}")
    print(f"Epoch: {args.epoch}")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print("=" * 60)

    # Interpretation
    print("\nInterpretation:")
    if acc1 > 85:
        print("  [OK] Logits appear to be from a well-trained model")
        print("  You can proceed with distillation training.")
    elif acc1 > 50:
        print("  [WARNING] Accuracy is moderate - teacher may be undertrained")
    elif acc1 > 5:
        print("  [WARNING] Accuracy is low - verify the teacher checkpoint")
    else:
        print("  [ERROR] Accuracy is very low - logits may be corrupted or from wrong model")

    # Show confusion analysis
    print("\n" + "-" * 60)
    print("Sample predictions vs ground truth:")
    for i in range(min(10, total)):
        if valid_mask[i]:
            pred = all_predictions[i]
            label = all_labels[i]
            top5 = all_indices[i][:5].tolist()
            probs = all_values[i][:5]
            status = "✓" if pred == label else "✗"
            in_top5 = "✓" if label in top5 else "✗"
            print(f"  [{i}] Pred: {pred:3d}, Label: {label:3d} {status}  | Top5: {top5} {in_top5} | Probs: {probs}")


if __name__ == '__main__':
    main()
