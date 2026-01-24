#!/bin/bash
# =============================================================================
# TinyViT-11M Distillation vs Scratch Experiments
# =============================================================================
# Reproducing TinyViT paper methodology with:
# - ImageNet-1K for pretraining (instead of ImageNet-22k)
# - CIFAR-100 for evaluation (downstream task)
# - CLIP-ViT-L/14 as teacher (finetuned on ImageNet-1K)
#
# Two experiments:
# A) Distillation: TinyViT-11M trained with CLIP soft labels
# B) Baseline: TinyViT-11M trained from scratch
#
# Both models are fine-tuned on CIFAR-100 for final comparison.
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_GPUS=${NUM_GPUS:-4}
DATA_PATH=${DATA_PATH:-"./data/ImageNet"}
CIFAR_PATH=${CIFAR_PATH:-"./data"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=============================================="
echo "TinyViT-11M Experiment Pipeline"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "ImageNet path: $DATA_PATH"
echo "CIFAR-100 path: $CIFAR_PATH"
echo "=============================================="

# =============================================================================
# STEP 0: Finetune CLIP on ImageNet-1K (head-only)
# =============================================================================
# CLIP was trained for contrastive learning, not classification.
# We finetune just the classification head to get ~85% accuracy.
# Time: ~4-8 hours on single GPU (20 epochs)
# =============================================================================

echo ""
echo "[Step 0/6] Finetuning CLIP on ImageNet-1K (head-only)..."
echo "Output: ./output/clip_vit_l_1k_finetune/"

CLIP_CKPT="./output/clip_vit_l_1k_finetune/CLIP-ViT-L-Finetune-1K/default/ckpt_epoch_19.pth"

if [ ! -f "$CLIP_CKPT" ]; then
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        main.py \
        --cfg configs/1k_distill/clip_vit_l_1k_finetune.yaml \
        --data-path "$DATA_PATH" \
        --output ./output/clip_vit_l_1k_finetune

    # Find the actual checkpoint
    CLIP_CKPT=$(ls -t ./output/clip_vit_l_1k_finetune/CLIP-ViT-L-Finetune-1K/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
    echo "[Step 0/6] DONE - CLIP finetuned, checkpoint: $CLIP_CKPT"
else
    echo "[Step 0/6] SKIP - Finetuned CLIP checkpoint already exists"
fi

# =============================================================================
# STEP 1: Save CLIP teacher logits on ImageNet-1K (top-50)
# =============================================================================
# This generates soft labels from finetuned CLIP-ViT-L/14 for distillation
# Output: ~50GB for 10 epochs of logits
# Time: ~2-4 hours on single GPU
# =============================================================================

echo ""
echo "[Step 1/6] Saving CLIP teacher logits on ImageNet-1K..."
echo "Output: ./output/logits/clip_vit_l_1k/"

if [ ! -d "./output/logits/clip_vit_l_1k/epoch_9" ]; then
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        main.py \
        --cfg configs/1k_distill/clip_vit_l_1k_save_logits.yaml \
        --data-path "$DATA_PATH" \
        --output ./output/logits/clip_vit_l_1k \
        --pretrained "$CLIP_CKPT"
    echo "[Step 1/6] DONE - Logits saved"
else
    echo "[Step 1/6] SKIP - Logits already exist"
fi

# =============================================================================
# STEP 2: Pretrain TinyViT-11M with CLIP distillation
# =============================================================================
# Train student using saved soft labels
# Time: ~24-48 hours on 4 GPUs (300 epochs)
# =============================================================================

echo ""
echo "[Step 2/6] Pretraining TinyViT-11M with CLIP distillation..."
echo "Output: ./output/tiny_vit_11m_1k_distill_clip/"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg configs/1k_distill/tiny_vit_11m_1k_distill_clip.yaml \
    --data-path "$DATA_PATH" \
    --output ./output/tiny_vit_11m_1k_distill_clip \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/clip_vit_l_1k/

echo "[Step 2/6] DONE - Distillation pretraining complete"

# =============================================================================
# STEP 3: Pretrain TinyViT-11M from scratch (baseline)
# =============================================================================
# Train without distillation for comparison
# Time: ~24-48 hours on 4 GPUs (300 epochs)
# =============================================================================

echo ""
echo "[Step 3/6] Pretraining TinyViT-11M from scratch (baseline)..."
echo "Output: ./output/tiny_vit_11m_1k_scratch/"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg configs/1k_distill/tiny_vit_11m_1k_scratch.yaml \
    --data-path "$DATA_PATH" \
    --output ./output/tiny_vit_11m_1k_scratch

echo "[Step 3/6] DONE - Scratch pretraining complete"

# =============================================================================
# STEP 4: Fine-tune distillation model on CIFAR-100
# =============================================================================
# Fine-tune for downstream evaluation
# Time: ~1-2 hours on single GPU (50 epochs)
# =============================================================================

echo ""
echo "[Step 4/6] Fine-tuning distillation model on CIFAR-100..."
echo "Output: ./output/cifar100_eval/distill/"

DISTILL_CKPT="./output/tiny_vit_11m_1k_distill_clip/TinyViT-11M-1k-Distill-CLIP/default/ckpt_epoch_299.pth"

# Find best checkpoint if epoch 299 doesn't exist
if [ ! -f "$DISTILL_CKPT" ]; then
    DISTILL_CKPT=$(ls -t ./output/tiny_vit_11m_1k_distill_clip/TinyViT-11M-1k-Distill-CLIP/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
fi

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg configs/cifar100/tiny_vit_11m_cifar100_finetune_from_distill.yaml \
    --data-path "$CIFAR_PATH" \
    --pretrained "$DISTILL_CKPT" \
    --output ./output/cifar100_eval/distill

echo "[Step 4/6] DONE - Distillation model fine-tuned"

# =============================================================================
# STEP 5: Fine-tune scratch model on CIFAR-100
# =============================================================================
# Fine-tune for downstream evaluation
# Time: ~1-2 hours on single GPU (50 epochs)
# =============================================================================

echo ""
echo "[Step 5/6] Fine-tuning scratch model on CIFAR-100..."
echo "Output: ./output/cifar100_eval/scratch/"

SCRATCH_CKPT="./output/tiny_vit_11m_1k_scratch/TinyViT-11M-1k-Scratch/default/ckpt_epoch_299.pth"

# Find best checkpoint if epoch 299 doesn't exist
if [ ! -f "$SCRATCH_CKPT" ]; then
    SCRATCH_CKPT=$(ls -t ./output/tiny_vit_11m_1k_scratch/TinyViT-11M-1k-Scratch/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
fi

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    main.py \
    --cfg configs/cifar100/tiny_vit_11m_cifar100_finetune_from_scratch.yaml \
    --data-path "$CIFAR_PATH" \
    --pretrained "$SCRATCH_CKPT" \
    --output ./output/cifar100_eval/scratch

echo "[Step 5/6] DONE - Scratch model fine-tuned"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE!"
echo "=============================================="
echo ""
echo "Results Summary:"
echo "0. CLIP finetuned: ./output/clip_vit_l_1k_finetune/"
echo "1. CLIP logits: ./output/logits/clip_vit_l_1k/"
echo "2. Distill pretrain: ./output/tiny_vit_11m_1k_distill_clip/"
echo "3. Scratch pretrain: ./output/tiny_vit_11m_1k_scratch/"
echo "4. CIFAR-100 (distill): ./output/cifar100_eval/distill/"
echo "5. CIFAR-100 (scratch): ./output/cifar100_eval/scratch/"
echo ""
echo "Compare CIFAR-100 accuracy between distill and scratch models."
echo "Expected: Distillation model should achieve higher accuracy"
echo "due to knowledge transfer from CLIP teacher."
echo "=============================================="
