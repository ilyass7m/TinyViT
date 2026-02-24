#!/bin/bash
# =============================================================================
# TinyViT-11M Experiments: Distillation vs Scratch
# =============================================================================
# Reproducing TinyViT paper with:
#   - Pretraining: ImageNet-1K
#   - Teacher: CLIP-ViT-L/14 (finetuned)
#   - Student: TinyViT-11M
#   - Evaluation: CIFAR-100
#
# Two experiments:
#   A) TinyViT-11M + CLIP distillation (top-50 logits)
#   B) TinyViT-11M from scratch (baseline)
# =============================================================================

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-4}
IMAGENET_PATH=${IMAGENET_PATH:-"./data/ImageNet"}
CIFAR_PATH=${CIFAR_PATH:-"./data"}
PORT=${PORT:-29500}

echo "=============================================="
echo "TinyViT-11M Experiments"
echo "=============================================="
echo "GPUs: $NUM_GPUS"
echo "ImageNet: $IMAGENET_PATH"
echo "CIFAR-100: $CIFAR_PATH"
echo "=============================================="

# =============================================================================
# STEP 0: Finetune CLIP on ImageNet-1K
# =============================================================================
echo ""
echo "[STEP 0/5] Finetuning CLIP-ViT-L/14 on ImageNet-1K..."

CLIP_CKPT="./output/exp_clip_finetune/CLIP-ViT-L-Finetune/default/ckpt_epoch_19.pth"

if [ ! -f "$CLIP_CKPT" ]; then
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$PORT \
        main.py \
        --cfg configs/experiments/step0_clip_finetune.yaml \
        --data-path "$IMAGENET_PATH" \
        --output ./output/exp_clip_finetune

    CLIP_CKPT=$(ls -t ./output/exp_clip_finetune/CLIP-ViT-L-Finetune/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
    echo "[STEP 0/5] DONE - Checkpoint: $CLIP_CKPT"
else
    echo "[STEP 0/5] SKIP - Checkpoint exists: $CLIP_CKPT"
fi

# =============================================================================
# STEP 1: Save CLIP logits (top-50)
# =============================================================================
echo ""
echo "[STEP 1/5] Saving CLIP teacher logits (top-50)..."

LOGITS_DIR="./output/exp_clip_logits/CLIP-ViT-L-SaveLogits/default"

if [ ! -d "$LOGITS_DIR/epoch_9" ]; then
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$PORT \
        main.py \
        --cfg configs/experiments/step1_clip_save_logits.yaml \
        --data-path "$IMAGENET_PATH" \
        --output ./output/exp_clip_logits \
        --pretrained "$CLIP_CKPT"
    echo "[STEP 1/5] DONE - Logits saved to $LOGITS_DIR"
else
    echo "[STEP 1/5] SKIP - Logits exist in $LOGITS_DIR"
fi

# =============================================================================
# STEP 2: Experiment A - TinyViT-11M with Distillation
# =============================================================================
echo ""
echo "[STEP 2/5] Experiment A: TinyViT-11M with CLIP distillation..."

torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    main.py \
    --cfg configs/experiments/exp_a_tinyvit11m_distill.yaml \
    --data-path "$IMAGENET_PATH" \
    --output ./output/exp_a_distill \
    --opts DISTILL.TEACHER_LOGITS_PATH "$LOGITS_DIR/"

echo "[STEP 2/5] DONE - Distillation pretraining complete"

# =============================================================================
# STEP 3: Experiment B - TinyViT-11M from Scratch
# =============================================================================
echo ""
echo "[STEP 3/5] Experiment B: TinyViT-11M from scratch (baseline)..."

torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
    main.py \
    --cfg configs/experiments/exp_b_tinyvit11m_scratch.yaml \
    --data-path "$IMAGENET_PATH" \
    --output ./output/exp_b_scratch

echo "[STEP 3/5] DONE - Scratch pretraining complete"

# =============================================================================
# STEP 4: CIFAR-100 Evaluation
# =============================================================================
echo ""
echo "[STEP 4/5] Evaluating on CIFAR-100..."

# Find checkpoints
DISTILL_CKPT=$(ls -t ./output/exp_a_distill/TinyViT-11M-Distill/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
SCRATCH_CKPT=$(ls -t ./output/exp_b_scratch/TinyViT-11M-Scratch/default/ckpt_epoch_*.pth 2>/dev/null | head -1)

echo "Distill checkpoint: $DISTILL_CKPT"
echo "Scratch checkpoint: $SCRATCH_CKPT"

# Evaluate distillation model
echo "Finetuning distillation model on CIFAR-100..."
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$PORT \
    main.py \
    --cfg configs/experiments/eval_cifar100_from_distill.yaml \
    --data-path "$CIFAR_PATH" \
    --pretrained "$DISTILL_CKPT" \
    --output ./output/eval_cifar100_distill

# Evaluate scratch model
echo "Finetuning scratch model on CIFAR-100..."
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$PORT \
    main.py \
    --cfg configs/experiments/eval_cifar100_from_scratch.yaml \
    --data-path "$CIFAR_PATH" \
    --pretrained "$SCRATCH_CKPT" \
    --output ./output/eval_cifar100_scratch

echo "[STEP 4/5] DONE - CIFAR-100 evaluation complete"

# =============================================================================
# Results Summary
# =============================================================================
echo ""
echo "=============================================="
echo "EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Results:"
echo "  1. CLIP finetuned:     ./output/exp_clip_finetune/"
echo "  2. CLIP logits:        ./output/exp_clip_logits/"
echo "  3. Exp A (Distill):    ./output/exp_a_distill/"
echo "  4. Exp B (Scratch):    ./output/exp_b_scratch/"
echo "  5. CIFAR-100 Distill:  ./output/eval_cifar100_distill/"
echo "  6. CIFAR-100 Scratch:  ./output/eval_cifar100_scratch/"
echo ""
echo "Compare CIFAR-100 accuracy in log files."
echo "Expected: Distillation > Scratch"
echo "=============================================="
