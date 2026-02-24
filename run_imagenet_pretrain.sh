#!/bin/bash
# =============================================================================
# TinyViT ImageNet-1K Pretraining Experiments
# =============================================================================
#
# Academic Project: Reproduce TinyViT paper results
#
# Two main pretraining experiments:
#   Exp A: TinyViT-21M + DISTILLATION from CLIP → CIFAR-100 finetune
#   Exp B: TinyViT-21M + SCRATCH training      → CIFAR-100 finetune
#
# Compare CIFAR-100 accuracy to demonstrate distillation benefit
#
# =============================================================================

set -e

# --- Environment Setup ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /workspace/TinyViT

# --- Configuration ---
DATA_PATH="/data/ImageNet"
CIFAR_PATH="./data"
NUM_GPUS=2
MASTER_PORT=29500

echo "============================================"
echo "TinyViT ImageNet-1K Pretraining Experiments"
echo "============================================"
echo "Hardware: ${NUM_GPUS}x A40 GPUs"
echo "ImageNet: ${DATA_PATH}"
echo ""

# =============================================================================
# STEP 1: Save TIMM CLIP Teacher Logits
# =============================================================================
step1_save_logits() {
    echo "============================================"
    echo "STEP 1: Saving TIMM CLIP Teacher Logits"
    echo "Model: vit_large_patch14_clip_336.laion2b_ft_in12k_in1k"
    echo "Time: ~3-5 hours"
    echo "============================================"

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        save_logits.py \
        --cfg configs/1k_distill/timm_clip_vit_l_336_1k_save_logits.yaml \
        --data-path ${DATA_PATH} \
        --output ./output/logits/timm_clip_vit_l_336_1k

    echo "✓ Logits saved to ./output/logits/timm_clip_vit_l_336_1k"
}

# =============================================================================
# STEP 2A: Distillation Pretraining (100 epochs)
# =============================================================================
step2a_distill() {
    EPOCHS=${1:-100}
    echo "============================================"
    echo "STEP 2A: Distillation Pretraining"
    echo "Epochs: ${EPOCHS}"
    echo "Time: ~12-24 hours"
    echo "============================================"

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        main.py \
        --cfg configs/1k_distill/tiny_vit_21m_1k_distill_timm_clip.yaml \
        --data-path ${DATA_PATH} \
        --output ./output/exp_a_distill \
        --opts \
            DISTILL.TEACHER_LOGITS_PATH ./output/logits/timm_clip_vit_l_336_1k \
            TRAIN.EPOCHS ${EPOCHS}

    echo "✓ Distillation checkpoint: ./output/exp_a_distill/"
}

# =============================================================================
# STEP 2B: Scratch Pretraining (100 epochs) - Already Running!
# =============================================================================
step2b_scratch() {
    EPOCHS=${1:-100}
    echo "============================================"
    echo "STEP 2B: Scratch Pretraining"
    echo "Epochs: ${EPOCHS}"
    echo "Time: ~12-24 hours"
    echo "============================================"

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        main.py \
        --cfg configs/1k_distill/tiny_vit_21m_1k_scratch.yaml \
        --data-path ${DATA_PATH} \
        --output ./output/exp_b_scratch \
        --opts TRAIN.EPOCHS ${EPOCHS}

    echo "✓ Scratch checkpoint: ./output/exp_b_scratch/"
}

# =============================================================================
# STEP 3A: CIFAR-100 Finetune from Distillation
# =============================================================================
step3a_cifar_distill() {
    echo "============================================"
    echo "STEP 3A: CIFAR-100 Finetune (Distillation)"
    echo "Time: ~30 minutes"
    echo "============================================"

    # Find checkpoint
    CKPT=$(ls -t ./output/exp_a_distill/*/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No distillation checkpoint found!"
        return 1
    fi
    echo "Using: $CKPT"

    torchrun \
        --nproc_per_node=1 \
        --master_port=${MASTER_PORT} \
        main.py \
        --cfg configs/cifar100/tiny_vit_21m_cifar100_finetune_from_distill.yaml \
        --data-path ${CIFAR_PATH} \
        --pretrained ${CKPT} \
        --output ./output/cifar100_from_distill

    echo "✓ Results: ./output/cifar100_from_distill/"
}

# =============================================================================
# STEP 3B: CIFAR-100 Finetune from Scratch
# =============================================================================
step3b_cifar_scratch() {
    echo "============================================"
    echo "STEP 3B: CIFAR-100 Finetune (Scratch)"
    echo "Time: ~30 minutes"
    echo "============================================"

    # Find checkpoint - check both output locations
    CKPT=$(ls -t ./output/exp_b_scratch/*/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        # Check alternative location from current training
        CKPT=$(ls -t ./output/tiny_vit_21m_1k_scratch/*/default/ckpt_epoch_*.pth 2>/dev/null | head -1)
    fi
    if [ -z "$CKPT" ]; then
        echo "ERROR: No scratch checkpoint found!"
        return 1
    fi
    echo "Using: $CKPT"

    torchrun \
        --nproc_per_node=1 \
        --master_port=${MASTER_PORT} \
        main.py \
        --cfg configs/cifar100/tiny_vit_21m_cifar100_finetune_from_scratch.yaml \
        --data-path ${CIFAR_PATH} \
        --pretrained ${CKPT} \
        --output ./output/cifar100_from_scratch

    echo "✓ Results: ./output/cifar100_from_scratch/"
}

# =============================================================================
# SUMMARY
# =============================================================================
summary() {
    echo "============================================"
    echo "EXPERIMENT SUMMARY"
    echo "============================================"
    echo ""
    echo "Expected Results (paper-like):"
    echo "┌─────────────────────┬───────────────────┬─────────────┐"
    echo "│ Pretraining Method  │ CIFAR-100 Acc     │ Improvement │"
    echo "├─────────────────────┼───────────────────┼─────────────┤"
    echo "│ Scratch (Exp B)     │ ~82-85%           │ baseline    │"
    echo "│ Distillation (Exp A)│ ~87-90%           │ +3-5%       │"
    echo "└─────────────────────┴───────────────────┴─────────────┘"
    echo ""
    echo "Key Insight: Distillation provides better representations"
    echo "             that transfer better to downstream tasks!"
    echo ""
    echo "Check logs:"
    echo "  ls ./output/cifar100_from_distill/"
    echo "  ls ./output/cifar100_from_scratch/"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
    step1)
        step1_save_logits
        ;;
    step2a)
        step2a_distill ${2:-100}
        ;;
    step2b)
        step2b_scratch ${2:-100}
        ;;
    step3a)
        step3a_cifar_distill
        ;;
    step3b)
        step3b_cifar_scratch
        ;;
    summary)
        summary
        ;;
    all)
        step1_save_logits
        step2a_distill 100
        step2b_scratch 100
        step3a_cifar_distill
        step3b_cifar_scratch
        summary
        ;;
    *)
        echo "TinyViT ImageNet-1K Pretraining Experiments"
        echo ""
        echo "Usage: $0 <step> [epochs]"
        echo ""
        echo "Steps:"
        echo "  step1  - Save TIMM CLIP teacher logits (~3-5h)"
        echo "  step2a - Distillation pretraining (~12-24h)"
        echo "  step2b - Scratch pretraining (~12-24h)"
        echo "  step3a - CIFAR-100 finetune from distillation (~30min)"
        echo "  step3b - CIFAR-100 finetune from scratch (~30min)"
        echo "  summary- Show expected results"
        echo "  all    - Run everything"
        echo ""
        echo "Examples:"
        echo "  $0 step1           # Save logits"
        echo "  $0 step2a 100      # Distill for 100 epochs"
        echo "  $0 step2b 100      # Scratch for 100 epochs"
        echo "  $0 step3a          # Finetune on CIFAR-100"
        echo ""
        echo "Your current scratch training is at epoch 58/300."
        echo "You can stop it and restart with 100 epochs for fair comparison."
        ;;
esac
