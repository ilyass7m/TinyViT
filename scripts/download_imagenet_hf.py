#!/usr/bin/env python3
"""
Download ImageNet-1k from Hugging Face and convert to folder structure.

This is the EASIEST way to get ImageNet-1k for academic use.

Requirements:
    pip install datasets huggingface_hub pillow tqdm

Usage:
    # First, login to Hugging Face (one-time):
    huggingface-cli login

    # Then accept the dataset terms at:
    # https://huggingface.co/datasets/imagenet-1k

    # Finally, run this script:
    python scripts/download_imagenet_hf.py --output ./ImageNet

Note:
    - Requires ~150GB of disk space
    - Download takes several hours depending on connection
    - The dataset is cached in ~/.cache/huggingface/
"""

import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def save_image(args):
    """Save a single image to disk."""
    image, label, idx, output_dir, class_names, split = args
    class_name = class_names[label]
    class_dir = output_dir / split / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    image_path = class_dir / f"{split}_{idx:08d}.JPEG"
    if not image_path.exists():
        image.save(image_path, "JPEG", quality=95)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet-1k from Hugging Face")
    parser.add_argument("--output", type=str, default="./ImageNet",
                        help="Output directory for ImageNet")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for saving images")
    parser.add_argument("--split", type=str, choices=["train", "validation", "both"],
                        default="both", help="Which split to download")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("Please install required packages:")
        print("  pip install datasets huggingface_hub pillow tqdm")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ImageNet-1k Download from Hugging Face")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    print("NOTE: You must first:")
    print("  1. Run: huggingface-cli login")
    print("  2. Accept terms at: https://huggingface.co/datasets/imagenet-1k")
    print()

    # Load dataset (this handles caching automatically)
    print("Loading dataset metadata from Hugging Face...")
    print("(This may take a while on first run)")

    splits_to_process = []
    if args.split in ["train", "both"]:
        splits_to_process.append("train")
    if args.split in ["validation", "both"]:
        splits_to_process.append("validation")

    for split in splits_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split...")
        print("=" * 60)

        # Load this split
        dataset = load_dataset(
            "imagenet-1k",
            split=split,
            trust_remote_code=True
        )

        # Get class names
        class_names = dataset.features["label"].names
        print(f"Found {len(class_names)} classes")
        print(f"Total images: {len(dataset)}")

        # Rename validation to val for consistency with PyTorch ImageFolder
        output_split = "val" if split == "validation" else split
        split_dir = output_dir / output_split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for parallel processing
        print(f"\nSaving images to {split_dir}...")

        # Process in batches for better memory management
        batch_size = 1000
        total_saved = 0

        with tqdm(total=len(dataset), desc=f"Saving {split}") as pbar:
            for batch_start in range(0, len(dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset))
                batch = dataset[batch_start:batch_end]

                # Prepare tasks
                tasks = []
                for i, (image, label) in enumerate(zip(batch["image"], batch["label"])):
                    idx = batch_start + i
                    tasks.append((image, label, idx, output_dir, class_names, output_split))

                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(save_image, task) for task in tasks]
                    for future in as_completed(futures):
                        future.result()
                        total_saved += 1
                        pbar.update(1)

        print(f"Saved {total_saved} images to {split_dir}")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    for split in ["train", "val"]:
        split_dir = output_dir / split
        if split_dir.exists():
            n_classes = len(list(split_dir.iterdir()))
            n_images = sum(1 for _ in split_dir.rglob("*.JPEG"))
            expected = "~1.28M" if split == "train" else "50,000"
            print(f"{split}: {n_classes} classes, {n_images:,} images (expected: 1000 classes, {expected} images)")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nUsage with TinyViT:")
    print(f"  torchrun --nproc_per_node=4 save_logits.py \\")
    print(f"      --cfg configs/1k_distill/resnet152_1k_save_logits.yaml \\")
    print(f"      --data-path {output_dir} \\")
    print(f"      --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits_1k_resnet152/")


if __name__ == "__main__":
    main()
