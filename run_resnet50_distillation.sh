#!/bin/bash
# =============================================================================
# ResNet50 → TinyViT-5M Distillation on CIFAR-100
# Complete Experiment Pipeline for 2xA40 GPUs
# =============================================================================
#
# This experiment:
#   1. Fine-tunes ResNet50 (ImageNet pretrained) on CIFAR-100 → Teacher
#   2. Saves teacher's soft predictions (logits)
#   3. Trains TinyViT-5M using teacher's knowledge → Student
#
# Expected Results:
#   - ResNet50 Teacher: ~88-90% accuracy (25M params)
#   - TinyViT-5M Student (distilled): ~85-88% accuracy (5M params)
#   - TinyViT-5M Baseline (no distill): ~82-85% accuracy
#
# =============================================================================

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
DATA_PATH="./data"
OUTPUT_BASE="./output_resnet50_distill"
LOGITS_PATH="./teacher_logits_resnet50"

# Wandb (uncomment to enable)
# WANDB_ARGS="--use-wandb --wandb-project tinyvit-cifar100-distill"
WANDB_ARGS=""

echo "============================================================"
echo "  ResNet50 → TinyViT-5M Distillation Experiment"
echo "  GPUs: ${NUM_GPUS} | Output: ${OUTPUT_BASE}"
echo "============================================================"

# =============================================================================
# STEP 1: Fine-tune ResNet50 Teacher on CIFAR-100
# =============================================================================
#
# What happens:
#   - timm loads ResNet50 with ImageNet pretrained weights
#   - Classifier head is automatically replaced: fc(2048→1000) → fc(2048→100)
#   - Fine-tune for 30 epochs on CIFAR-100
#
# Expected: ~88-90% accuracy
# Time: ~1-2 hours on 2xA40
# =============================================================================
step1_train_teacher() {
    echo ""
    echo "============================================================"
    echo "  STEP 1: Fine-tuning ResNet50 Teacher on CIFAR-100"
    echo "============================================================"
    echo ""
    echo "  Model: ResNet50 (25M params)"
    echo "  Pretrained: ImageNet-1k (auto-loaded by timm)"
    echo "  Target: CIFAR-100 (100 classes)"
    echo "  Epochs: 30"
    echo ""

    mkdir -p ${OUTPUT_BASE}/teacher

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/resnet50_cifar100_teacher.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/teacher \
        ${WANDB_ARGS}

    echo ""
    echo "✓ Teacher training complete!"
    echo "  Best checkpoint: ${OUTPUT_BASE}/teacher/ckpt_best.pth"
    echo ""
}

# =============================================================================
# STEP 2: Save Teacher Logits
# =============================================================================
#
# What happens:
#   - Load fine-tuned ResNet50 checkpoint
#   - Run inference on CIFAR-100 training data (50k images)
#   - For each image: save top-50 class probabilities + augmentation seed
#   - Repeat for 10 epochs (different augmentations)
#
# Storage: ~50k images × 10 epochs × 204 bytes ≈ 100 MB
# Time: ~30 minutes on 2xA40
# =============================================================================
step2_save_logits() {
    echo ""
    echo "============================================================"
    echo "  STEP 2: Saving Teacher Logits"
    echo "============================================================"
    echo ""

    # Check teacher checkpoint exists
    TEACHER_CKPT="${OUTPUT_BASE}/teacher/ckpt_best.pth"
    if [ ! -f "${TEACHER_CKPT}" ]; then
        echo "ERROR: Teacher checkpoint not found at ${TEACHER_CKPT}"
        echo "Please run step1 first: $0 step1"
        exit 1
    fi

    mkdir -p ${LOGITS_PATH}

    echo "  Loading teacher from: ${TEACHER_CKPT}"
    echo "  Saving logits to: ${LOGITS_PATH}"
    echo "  Saving top-50 class probabilities per sample"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} save_logits.py \
        --cfg configs/cifar100/resnet50_cifar100_save_logits.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/save_logits \
        --resume ${TEACHER_CKPT} \
        --opts DISTILL.TEACHER_LOGITS_PATH ${LOGITS_PATH}

    echo ""
    echo "✓ Teacher logits saved!"
    echo "  Location: ${LOGITS_PATH}"
    ls -la ${LOGITS_PATH}/
    echo ""
}

# =============================================================================
# STEP 3: Train TinyViT-5M Student with Distillation
# =============================================================================
#
# What happens:
#   - Build TinyViT-5M model (random init, 5M params)
#   - Load saved teacher logits
#   - Train using KL divergence loss: student matches teacher's soft predictions
#   - Student learns "dark knowledge" from teacher
#
# Expected: ~85-88% accuracy (vs ~82-85% without distillation)
# Time: ~3-4 hours on 2xA40
# =============================================================================
step3_train_student() {
    echo ""
    echo "============================================================"
    echo "  STEP 3: Training TinyViT-5M with Distillation"
    echo "============================================================"
    echo ""

    # Check logits exist
    if [ ! -d "${LOGITS_PATH}" ] || [ -z "$(ls -A ${LOGITS_PATH} 2>/dev/null)" ]; then
        echo "ERROR: Teacher logits not found at ${LOGITS_PATH}"
        echo "Please run step2 first: $0 step2"
        exit 1
    fi

    echo "  Student: TinyViT-5M (5M params)"
    echo "  Teacher logits: ${LOGITS_PATH}"
    echo "  Loss: KL Divergence (soft targets)"
    echo "  Epochs: 100"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_distill_from_resnet.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/student_distill \
        --opts DISTILL.TEACHER_LOGITS_PATH ${LOGITS_PATH} \
        ${WANDB_ARGS}

    echo ""
    echo "✓ Student distillation training complete!"
    echo "  Best checkpoint: ${OUTPUT_BASE}/student_distill/ckpt_best.pth"
    echo ""
}

# =============================================================================
# STEP 4 (Optional): Train Baseline for Comparison
# =============================================================================
step4_train_baseline() {
    echo ""
    echo "============================================================"
    echo "  STEP 4: Training TinyViT-5M Baseline (no distillation)"
    echo "============================================================"
    echo ""

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/student_baseline \
        ${WANDB_ARGS}

    echo ""
    echo "✓ Baseline training complete!"
    echo ""
}

# =============================================================================
# Compare Results
# =============================================================================
compare_results() {
    echo ""
    echo "============================================================"
    echo "  EXPECTED RESULTS COMPARISON"
    echo "============================================================"
    echo ""
    echo "  ┌────────────────────────────────────────────────────────────┐"
    echo "  │ Model                          │ Params │ Expected Acc    │"
    echo "  ├────────────────────────────────────────────────────────────┤"
    echo "  │ ResNet50 Teacher (fine-tuned)  │ 25M    │ 88-90%          │"
    echo "  │ TinyViT-5M + Distillation      │ 5M     │ 85-88%          │"
    echo "  │ TinyViT-5M Baseline (scratch)  │ 5M     │ 82-85%          │"
    echo "  └────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  Key insight: Distillation transfers knowledge from 25M → 5M model"
    echo "  Student achieves ~3-5% higher accuracy than training from scratch!"
    echo ""
}

# =============================================================================
# Run Full Pipeline
# =============================================================================
run_full() {
    echo "Running full distillation pipeline..."
    step1_train_teacher
    step2_save_logits
    step3_train_student
    compare_results
}

# =============================================================================
# Quick Test (1 epoch each)
# =============================================================================
quick_test() {
    echo "Running quick test (1 epoch each)..."

    # Test teacher training (1 epoch)
    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/resnet50_cifar100_teacher.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/test_teacher \
        --opts TRAIN.EPOCHS 1 PRINT_FREQ 10

    echo ""
    echo "✓ Quick test passed! Setup is working."
    echo ""
}

# =============================================================================
# Main Menu
# =============================================================================
print_usage() {
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  step1          Train ResNet50 teacher on CIFAR-100 (~1-2h)"
    echo "  step2          Save teacher logits (~30min)"
    echo "  step3          Train TinyViT-5M with distillation (~3-4h)"
    echo "  step4          Train TinyViT-5M baseline for comparison (~4h)"
    echo "  full           Run complete pipeline (step1 → step3)"
    echo "  compare        Show expected results"
    echo "  test           Quick test (1 epoch) to verify setup"
    echo ""
    echo "Example:"
    echo "  $0 step1       # First, train teacher"
    echo "  $0 step2       # Then, save logits"
    echo "  $0 step3       # Finally, train student"
    echo ""
}

case "${1:-help}" in
    step1|train_teacher)
        step1_train_teacher
        ;;
    step2|save_logits)
        step2_save_logits
        ;;
    step3|train_student)
        step3_train_student
        ;;
    step4|baseline)
        step4_train_baseline
        ;;
    full|all)
        run_full
        ;;
    compare)
        compare_results
        ;;
    test)
        quick_test
        ;;
    *)
        print_usage
        ;;
esac
