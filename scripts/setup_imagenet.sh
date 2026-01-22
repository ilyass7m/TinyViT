#!/bin/bash
# ImageNet-1k Setup Script for TinyViT
# This script organizes ImageNet data into the expected directory structure
#
# Expected final structure:
#   ImageNet/
#   ├── train/
#   │   ├── n01440764/
#   │   │   ├── n01440764_10026.JPEG
#   │   │   └── ...
#   │   └── ... (1000 class folders)
#   └── val/
#       ├── n01440764/
#       │   ├── ILSVRC2012_val_00000293.JPEG
#       │   └── ...
#       └── ... (1000 class folders)

set -e

# Configuration
IMAGENET_DIR="${1:-./ImageNet}"
TRAIN_TAR="${2:-ILSVRC2012_img_train.tar}"
VAL_TAR="${3:-ILSVRC2012_img_val.tar}"

echo "=========================================="
echo "ImageNet-1k Setup for TinyViT"
echo "=========================================="
echo "Target directory: $IMAGENET_DIR"
echo ""

# Create directories
mkdir -p "$IMAGENET_DIR/train"
mkdir -p "$IMAGENET_DIR/val"

# ==========================================
# Extract Training Set
# ==========================================
if [ -f "$TRAIN_TAR" ]; then
    echo "[1/4] Extracting training set..."
    tar -xf "$TRAIN_TAR" -C "$IMAGENET_DIR/train"

    echo "[2/4] Extracting class-specific tar files..."
    cd "$IMAGENET_DIR/train"
    for f in *.tar; do
        if [ -f "$f" ]; then
            dir="${f%.tar}"
            mkdir -p "$dir"
            tar -xf "$f" -C "$dir"
            rm "$f"
        fi
    done
    cd - > /dev/null

    echo "Training set ready: $(find $IMAGENET_DIR/train -type f -name '*.JPEG' | wc -l) images"
else
    echo "[SKIP] Training tar not found: $TRAIN_TAR"
    echo "       Download from https://image-net.org/download-images.php"
fi

# ==========================================
# Extract Validation Set
# ==========================================
if [ -f "$VAL_TAR" ]; then
    echo "[3/4] Extracting validation set..."
    tar -xf "$VAL_TAR" -C "$IMAGENET_DIR/val"

    echo "[4/4] Organizing validation images into class folders..."

    # Download the validation labels mapping script
    VALPREP_SCRIPT="$IMAGENET_DIR/valprep.sh"
    if [ ! -f "$VALPREP_SCRIPT" ]; then
        curl -L -o "$VALPREP_SCRIPT" \
            "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh"
    fi

    cd "$IMAGENET_DIR/val"
    bash ../valprep.sh
    cd - > /dev/null

    echo "Validation set ready: $(find $IMAGENET_DIR/val -type f -name '*.JPEG' | wc -l) images"
else
    echo "[SKIP] Validation tar not found: $VAL_TAR"
    echo "       Download from https://image-net.org/download-images.php"
fi

# ==========================================
# Verify Setup
# ==========================================
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

TRAIN_CLASSES=$(find "$IMAGENET_DIR/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_CLASSES=$(find "$IMAGENET_DIR/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
TRAIN_IMAGES=$(find "$IMAGENET_DIR/train" -type f -name '*.JPEG' 2>/dev/null | wc -l)
VAL_IMAGES=$(find "$IMAGENET_DIR/val" -type f -name '*.JPEG' 2>/dev/null | wc -l)

echo "Training:   $TRAIN_CLASSES classes, $TRAIN_IMAGES images (expected: 1000 classes, ~1.28M images)"
echo "Validation: $VAL_CLASSES classes, $VAL_IMAGES images (expected: 1000 classes, 50000 images)"

if [ "$TRAIN_CLASSES" -eq 1000 ] && [ "$VAL_CLASSES" -eq 1000 ]; then
    echo ""
    echo "ImageNet-1k setup complete!"
    echo ""
    echo "Usage with TinyViT:"
    echo "  torchrun --nproc_per_node=4 save_logits.py \\"
    echo "      --cfg configs/1k_distill/resnet152_1k_save_logits.yaml \\"
    echo "      --data-path $IMAGENET_DIR \\"
    echo "      --opts DISTILL.TEACHER_LOGITS_PATH ./teacher_logits_1k_resnet152/"
else
    echo ""
    echo "WARNING: Setup may be incomplete. Please check the extracted files."
fi
