#!/bin/bash
# =============================================================================
# TinyViT CIFAR-100 Experiment Runner
# =============================================================================
# Usage:
#   ./run_experiments.sh <experiment> [gpu_id] [port]
#
# Examples:
#   ./run_experiments.sh exp1 0 29500        # Run exp1 on GPU 0
#   ./run_experiments.sh exp6_vit 1 29501    # Run exp6 ViT on GPU 1
#   ./run_experiments.sh all                  # Run all experiments sequentially
# =============================================================================

set -e

DATA_PATH="./data"
OUTPUT_BASE="./output"
PRETRAINED_DIR="./pretrained"

# Default settings
GPU=${2:-0}
PORT=${3:-29500}

# Helper function
run_training() {
    local cfg=$1
    local output=$2
    local extra_args=$3

    echo "========================================"
    echo "Running: $output"
    echo "Config: $cfg"
    echo "GPU: $GPU, Port: $PORT"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT main.py \
        --cfg "$cfg" \
        --data-path "$DATA_PATH" \
        --output "$output" \
        $extra_args
}

run_save_logits() {
    local cfg=$1
    local resume=$2
    local logits_path=$3

    echo "========================================"
    echo "Saving logits to: $logits_path"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT save_logits.py \
        --cfg "$cfg" \
        --data-path "$DATA_PATH" \
        --output "$OUTPUT_BASE/save_logits" \
        --resume "$resume" \
        --opts DISTILL.TEACHER_LOGITS_PATH "$logits_path"
}

download_pretrained() {
    mkdir -p "$PRETRAINED_DIR"

    if [ ! -f "$PRETRAINED_DIR/tiny_vit_5m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-5M pretrained..."
        wget -O "$PRETRAINED_DIR/tiny_vit_5m_22k_distill.pth" \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth"
    fi

    if [ ! -f "$PRETRAINED_DIR/tiny_vit_21m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-21M pretrained..."
        wget -O "$PRETRAINED_DIR/tiny_vit_21m_22k_distill.pth" \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth"
    fi
}

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

exp1() {
    echo "=== EXPERIMENT 1: Scratch Training ==="
    run_training \
        "configs/cifar100/experiments/exp1_student_scratch.yaml" \
        "$OUTPUT_BASE/exp1_scratch"
}

exp2() {
    echo "=== EXPERIMENT 2: Transfer Learning ==="
    download_pretrained
    run_training \
        "configs/cifar100/experiments/exp2_student_finetune.yaml" \
        "$OUTPUT_BASE/exp2_finetune" \
        "--pretrained $PRETRAINED_DIR/tiny_vit_5m_22k_distill.pth"
}

exp3_resnet() {
    echo "=== EXPERIMENT 3a: Train ResNet-152 Teacher ==="
    run_training \
        "configs/cifar100/experiments/exp3_teacher_resnet152.yaml" \
        "$OUTPUT_BASE/exp3_teacher_resnet152"
}

exp3_vit() {
    echo "=== EXPERIMENT 3b: Train ViT-Base Teacher ==="
    run_training \
        "configs/cifar100/experiments/exp3_teacher_vit_base.yaml" \
        "$OUTPUT_BASE/exp3_teacher_vit_base"
}

exp3_tinyvit() {
    echo "=== EXPERIMENT 3c: Train TinyViT-21M Teacher ==="
    download_pretrained
    run_training \
        "configs/cifar100/experiments/exp3_teacher_tinyvit21m.yaml" \
        "$OUTPUT_BASE/exp3_teacher_tinyvit21m" \
        "--pretrained $PRETRAINED_DIR/tiny_vit_21m_22k_distill.pth"
}

exp4_resnet() {
    echo "=== EXPERIMENT 4a: Save ResNet-152 Logits ==="
    run_save_logits \
        "configs/cifar100/experiments/exp4_save_logits_resnet152.yaml" \
        "$OUTPUT_BASE/exp3_teacher_resnet152/Exp3-ResNet152-Teacher/default/ckpt_best.pth" \
        "$OUTPUT_BASE/logits/resnet152"
}

exp4_vit() {
    echo "=== EXPERIMENT 4b: Save ViT-Base Logits ==="
    run_save_logits \
        "configs/cifar100/experiments/exp4_save_logits_vit_base.yaml" \
        "$OUTPUT_BASE/exp3_teacher_vit_base/Exp3-ViTBase-Teacher/default/ckpt_best.pth" \
        "$OUTPUT_BASE/logits/vit_base"
}

exp4_tinyvit() {
    echo "=== EXPERIMENT 4c: Save TinyViT-21M Logits ==="
    run_save_logits \
        "configs/cifar100/experiments/exp4_save_logits_tinyvit21m.yaml" \
        "$OUTPUT_BASE/exp3_teacher_tinyvit21m/Exp3-TinyViT21M-Teacher/default/ckpt_best.pth" \
        "$OUTPUT_BASE/logits/tinyvit21m"
}

exp5() {
    echo "=== EXPERIMENT 5: Distillation with TinyViT-21M ==="
    run_training \
        "configs/cifar100/experiments/exp5_distill_tinyvit21m.yaml" \
        "$OUTPUT_BASE/exp5_distill_tinyvit21m" \
        "--opts DISTILL.TEACHER_LOGITS_PATH $OUTPUT_BASE/logits/tinyvit21m/"
}

exp6_resnet() {
    echo "=== EXPERIMENT 6a: Distillation with ResNet-152 ==="
    run_training \
        "configs/cifar100/experiments/exp6_distill_resnet152.yaml" \
        "$OUTPUT_BASE/exp6_distill_resnet152" \
        "--opts DISTILL.TEACHER_LOGITS_PATH $OUTPUT_BASE/logits/resnet152/"
}

exp6_vit() {
    echo "=== EXPERIMENT 6b: Distillation with ViT-Base ==="
    run_training \
        "configs/cifar100/experiments/exp6_distill_vit_base.yaml" \
        "$OUTPUT_BASE/exp6_distill_vit_base" \
        "--opts DISTILL.TEACHER_LOGITS_PATH $OUTPUT_BASE/logits/vit_base/"
}

exp8() {
    echo "=== EXPERIMENT 8: Transfer + Distillation ==="
    download_pretrained
    run_training \
        "configs/cifar100/experiments/exp8_transfer_distill.yaml" \
        "$OUTPUT_BASE/exp8_transfer_distill" \
        "--pretrained $PRETRAINED_DIR/tiny_vit_5m_22k_distill.pth --opts DISTILL.TEACHER_LOGITS_PATH $OUTPUT_BASE/logits/tinyvit21m/"
}

# =============================================================================
# MAIN
# =============================================================================

case "${1:-help}" in
    exp1) exp1 ;;
    exp2) exp2 ;;
    exp3_resnet) exp3_resnet ;;
    exp3_vit) exp3_vit ;;
    exp3_tinyvit) exp3_tinyvit ;;
    exp4_resnet) exp4_resnet ;;
    exp4_vit) exp4_vit ;;
    exp4_tinyvit) exp4_tinyvit ;;
    exp5) exp5 ;;
    exp6_resnet) exp6_resnet ;;
    exp6_vit) exp6_vit ;;
    exp8) exp8 ;;

    phase1)
        exp1
        exp2
        ;;
    phase2)
        exp3_resnet
        exp3_vit
        exp3_tinyvit
        exp4_resnet
        exp4_vit
        exp4_tinyvit
        ;;
    phase3)
        exp5
        ;;
    phase4)
        exp6_resnet
        exp6_vit
        ;;

    all)
        exp1
        exp2
        exp3_resnet
        exp3_vit
        exp3_tinyvit
        exp4_resnet
        exp4_vit
        exp4_tinyvit
        exp5
        exp6_resnet
        exp6_vit
        ;;

    *)
        echo "TinyViT CIFAR-100 Experiment Runner"
        echo ""
        echo "Usage: $0 <experiment> [gpu_id] [port]"
        echo ""
        echo "Single Experiments:"
        echo "  exp1          - Scratch training (baseline)"
        echo "  exp2          - Transfer learning (baseline)"
        echo "  exp3_resnet   - Train ResNet-152 teacher"
        echo "  exp3_vit      - Train ViT-Base teacher"
        echo "  exp3_tinyvit  - Train TinyViT-21M teacher"
        echo "  exp4_resnet   - Save ResNet-152 logits"
        echo "  exp4_vit      - Save ViT-Base logits"
        echo "  exp4_tinyvit  - Save TinyViT-21M logits"
        echo "  exp5          - Distill with TinyViT-21M"
        echo "  exp6_resnet   - Distill with ResNet-152"
        echo "  exp6_vit      - Distill with ViT-Base"
        echo "  exp8          - Transfer + Distillation"
        echo ""
        echo "Phase Groups:"
        echo "  phase1        - Run exp1 + exp2"
        echo "  phase2        - Run all teacher training + logits"
        echo "  phase3        - Run exp5"
        echo "  phase4        - Run exp6"
        echo "  all           - Run all experiments"
        echo ""
        echo "Examples:"
        echo "  $0 exp1 0 29500     # Run exp1 on GPU 0, port 29500"
        echo "  $0 exp6_vit 1 29501 # Run exp6 ViT on GPU 1, port 29501"
        ;;
esac
