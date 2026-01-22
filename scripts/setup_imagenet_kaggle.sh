#!/usr/bin/env bash
#
# Efficient ImageNet-1K setup from Kaggle Object Localization dataset
#
# - Extracts ONLY train + val
# - Skips test set (saves ~40GB)
# - Avoids duplicate data
# - Safe for RunPod volumes
#
# Usage:
#   cd /workspace/datasets
#   kaggle competitions download -c imagenet-object-localization-challenge
#   bash setup_imagenet_kaggle_efficient.sh ImageNet imagenet-object-localization-challenge.zip
#

set -euo pipefail

############################
# Arguments
############################
OUT_DIR="${1:-ImageNet}"
ZIP_FILE="${2:-imagenet-object-localization-challenge.zip}"

OUT_DIR="$(realpath "$OUT_DIR")"
ZIP_FILE="$(realpath "$ZIP_FILE")"

############################
# Sanity checks
############################
echo "========================================="
echo " ImageNet-1K setup (efficient)"
echo "========================================="
echo "Output dir : $OUT_DIR"
echo "Zip file   : $ZIP_FILE"
echo ""

if [[ ! -f "$ZIP_FILE" ]]; then
  echo "ERROR: Kaggle zip not found: $ZIP_FILE"
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

############################
# Step 1: Extract ONLY what we need
############################
echo "[1/4] Extracting train + val only (this takes time)..."

unzip -q "$ZIP_FILE" \
  "ILSVRC/Data/CLS-LOC/train/*" \
  "ILSVRC/Data/CLS-LOC/val/*" \
  "LOC_val_solution.csv"

echo "      Extraction done."

############################
# Step 2: Move to final layout (no duplication)
############################
echo ""
echo "[2/4] Reorganizing directory structure..."

mv ILSVRC/Data/CLS-LOC/train ./train
mv ILSVRC/Data/CLS-LOC/val   ./val

############################
# Step 3: Organize validation into class folders
############################
echo ""
echo "[3/4] Organizing validation set..."

VAL_DIR="$OUT_DIR/val"
MAP_FILE="$OUT_DIR/LOC_val_solution.csv"

cd "$VAL_DIR"

# Only reorganize if val is flat
VAL_SUBDIRS=$(find . -mindepth 1 -maxdepth 1 -type d | wc -l)

if [[ "$VAL_SUBDIRS" -lt 100 ]]; then
  echo "      Rebuilding val/class_x folders..."

  tail -n +2 "$MAP_FILE" | while IFS=',' read -r image_id prediction; do
    class=$(echo "$prediction" | awk '{print $1}')
    img="${image_id}.JPEG"

    if [[ -f "$img" ]]; then
      mkdir -p "$class"
      mv "$img" "$class/"
    fi
  done
else
  echo "      Validation already organized."
fi

cd "$OUT_DIR"

############################
# Step 4: Cleanup
############################
echo ""
echo "[4/4] Cleaning up..."

rm -rf ILSVRC
rm -f "$ZIP_FILE"
rm -f LOC_val_solution.csv

############################
# Verification
############################
echo ""
echo "========================================="
echo " Verification"
echo "========================================="

TRAIN_CLASSES=$(find train -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_CLASSES=$(find val   -mindepth 1 -maxdepth 1 -type d | wc -l)
TRAIN_IMAGES=$(find train -type f -name "*.JPEG" | wc -l)
VAL_IMAGES=$(find val   -type f -name "*.JPEG" | wc -l)

echo "Train: $TRAIN_CLASSES classes, $TRAIN_IMAGES images"
echo "Val:   $VAL_CLASSES classes, $VAL_IMAGES images"
echo ""
echo "Expected:"
echo "  - Train: 1000 classes, ~1.28M images"
echo "  - Val:   1000 classes, 50,000 images"
echo ""

if [[ "$TRAIN_CLASSES" -eq 1000 && "$VAL_CLASSES" -eq 1000 ]]; then
  echo "✅ ImageNet-1K setup SUCCESSFUL"
else
  echo "⚠️  WARNING: Class count mismatch"
fi
