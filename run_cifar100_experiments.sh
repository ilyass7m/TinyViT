#!/bin/bash
# =============================================================================
# TinyViT CIFAR-100 Ablation Study - Experiment Runner
# For 2xA40 GPUs (48GB each) with large batch training
# =============================================================================

# Configuration
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
DATA_PATH="./data"  # CIFAR-100 will auto-download here
OUTPUT_BASE="./output_experiments"

# Wandb settings (uncomment to enable)
# WANDB_ARGS="--use-wandb --wandb-project tinyvit-cifar100-ablations"
WANDB_ARGS=""

# =============================================================================
# EXPERIMENT ORDER (Recommended for poster/ablation)
# =============================================================================
#
# Priority 1: Fine-tuning experiments (fastest results, best performance)
#   - Exp 1: TinyViT-5M fine-tuned from ImageNet (30 epochs, ~1-2 hours)
#   - Exp 2: TinyViT-11M fine-tuned from ImageNet (30 epochs, ~2-3 hours)
#
# Priority 2: Ablation on training strategy
#   - Exp 3: TinyViT-5M from scratch (100 epochs, ~6-8 hours)
#   - Compare with Exp 1 to show transfer learning benefit
#
# Priority 3: Architecture ablation
#   - Exp 4: TinyViT-5M vs 11M comparison (already have from Exp 1 & 2)
#
# Priority 4: Batch size ablation (if time permits)
#   - Exp 5: Compare BS=256 vs BS=512 vs BS=1024
# =============================================================================

echo "=============================================="
echo "TinyViT CIFAR-100 Ablation Experiments"
echo "GPUs: ${NUM_GPUS}, Output: ${OUTPUT_BASE}"
echo "=============================================="

# -----------------------------------------------------------------------------
# EXPERIMENT 1: TinyViT-5M Fine-tuning (RECOMMENDED FIRST)
# Expected: ~88-90% accuracy in 30 epochs
# Time: ~1-2 hours on 2xA40
# -----------------------------------------------------------------------------
run_exp1_5m_finetune() {
    echo "[Exp 1] TinyViT-5M Fine-tuning from ImageNet pretrained..."

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_finetune_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/exp1_5m_finetune \
        --pretrained pretrained/tiny_vit_5m_22k_distill.pth \
        ${WANDB_ARGS}
}

# -----------------------------------------------------------------------------
# EXPERIMENT 2: TinyViT-11M Fine-tuning
# Expected: ~89-91% accuracy in 30 epochs
# Time: ~2-3 hours on 2xA40
# -----------------------------------------------------------------------------
run_exp2_11m_finetune() {
    echo "[Exp 2] TinyViT-11M Fine-tuning from ImageNet pretrained..."

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_11m_cifar100_finetune_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/exp2_11m_finetune \
        --pretrained pretrained/tiny_vit_11m_22k_distill.pth \
        ${WANDB_ARGS}
}

# -----------------------------------------------------------------------------
# EXPERIMENT 3: TinyViT-5M From Scratch (Baseline)
# Expected: ~82-85% accuracy in 100 epochs
# Time: ~6-8 hours on 2xA40
# -----------------------------------------------------------------------------
run_exp3_5m_scratch() {
    echo "[Exp 3] TinyViT-5M Training from scratch..."

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/exp3_5m_scratch \
        ${WANDB_ARGS}
}

# -----------------------------------------------------------------------------
# EXPERIMENT 4: Batch Size Ablation (BS=256)
# Compare with default BS=1024 to show scaling behavior
# -----------------------------------------------------------------------------
run_exp4_bs256() {
    echo "[Exp 4] Batch size ablation: BS=256..."

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_finetune_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/exp4_bs256 \
        --pretrained pretrained/tiny_vit_5m_22k_distill.pth \
        --batch-size 128 \
        --opts TRAIN.BASE_LR 5e-5 \
        ${WANDB_ARGS}
}

# -----------------------------------------------------------------------------
# EXPERIMENT 5: Learning Rate Ablation
# Test different LR schedules
# -----------------------------------------------------------------------------
run_exp5_lr_ablation() {
    echo "[Exp 5] Learning rate ablation: LR=5e-4..."

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_finetune_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/exp5_lr5e4 \
        --pretrained pretrained/tiny_vit_5m_22k_distill.pth \
        --opts TRAIN.BASE_LR 5e-4 \
        ${WANDB_ARGS}
}

# -----------------------------------------------------------------------------
# DOWNLOAD PRETRAINED WEIGHTS
# -----------------------------------------------------------------------------
download_pretrained() {
    echo "Downloading pretrained weights..."
    mkdir -p pretrained

    # TinyViT-5M (ImageNet-22k distilled)
    if [ ! -f "pretrained/tiny_vit_5m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-5M..."
        wget -O pretrained/tiny_vit_5m_22k_distill.pth \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth"
    fi

    # TinyViT-11M (ImageNet-22k distilled)
    if [ ! -f "pretrained/tiny_vit_11m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-11M..."
        wget -O pretrained/tiny_vit_11m_22k_distill.pth \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth"
    fi

    # TinyViT-21M (ImageNet-22k distilled)
    if [ ! -f "pretrained/tiny_vit_21m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-21M..."
        wget -O pretrained/tiny_vit_21m_22k_distill.pth \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth"
    fi

    echo "Done downloading pretrained weights!"
}

# -----------------------------------------------------------------------------
# MAIN MENU
# -----------------------------------------------------------------------------
case "${1:-all}" in
    download)
        download_pretrained
        ;;
    exp1)
        run_exp1_5m_finetune
        ;;
    exp2)
        run_exp2_11m_finetune
        ;;
    exp3)
        run_exp3_5m_scratch
        ;;
    exp4)
        run_exp4_bs256
        ;;
    exp5)
        run_exp5_lr_ablation
        ;;
    quick)
        # Quick results for poster: just fine-tuning experiments
        echo "Running quick experiments (fine-tuning only)..."
        download_pretrained
        run_exp1_5m_finetune
        run_exp2_11m_finetune
        ;;
    all)
        # Full ablation study
        echo "Running all experiments..."
        download_pretrained
        run_exp1_5m_finetune
        run_exp2_11m_finetune
        run_exp3_5m_scratch
        ;;
    *)
        echo "Usage: $0 {download|exp1|exp2|exp3|exp4|exp5|quick|all}"
        echo ""
        echo "  download  - Download pretrained weights"
        echo "  exp1      - TinyViT-5M fine-tuning (30 epochs)"
        echo "  exp2      - TinyViT-11M fine-tuning (30 epochs)"
        echo "  exp3      - TinyViT-5M from scratch (100 epochs)"
        echo "  exp4      - Batch size ablation (BS=256)"
        echo "  exp5      - Learning rate ablation"
        echo "  quick     - Just fine-tuning experiments (fastest)"
        echo "  all       - All main experiments"
        ;;
esac
