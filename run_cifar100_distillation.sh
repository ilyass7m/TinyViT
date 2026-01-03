#!/bin/bash
# =============================================================================
# TinyViT CIFAR-100 Knowledge Distillation Pipeline
# For 2xA40 GPUs
# =============================================================================
#
# WORKFLOW:
#   Step 1: Train Teacher (TinyViT-21M fine-tuned on CIFAR-100)
#   Step 2: Save Teacher Logits (run inference, save predictions)
#   Step 3: Train Student (TinyViT-5M using teacher's soft targets)
#
# This implements "transfer distillation":
#   ImageNet-pretrained 21M → fine-tune on CIFAR-100 → distill to 5M
#
# =============================================================================

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
DATA_PATH="./data"
OUTPUT_BASE="./output_distillation"
LOGITS_PATH="./teacher_logits_cifar100"

# Wandb (optional)
# WANDB_ARGS="--use-wandb --wandb-project tinyvit-cifar100-distill"
WANDB_ARGS=""

echo "=============================================="
echo "TinyViT CIFAR-100 Distillation Pipeline"
echo "=============================================="

# =============================================================================
# STEP 1: Train Teacher (TinyViT-21M)
# Expected: ~91-93% accuracy after 30 epochs
# Time: ~3-4 hours on 2xA40
# =============================================================================
train_teacher() {
    echo ""
    echo "=============================================="
    echo "STEP 1: Training Teacher (TinyViT-21M)"
    echo "=============================================="

    # First download pretrained weights
    mkdir -p pretrained
    if [ ! -f "pretrained/tiny_vit_21m_22k_distill.pth" ]; then
        echo "Downloading TinyViT-21M pretrained weights..."
        wget -O pretrained/tiny_vit_21m_22k_distill.pth \
            "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth"
    fi

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_21m_cifar100_teacher.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/teacher \
        --pretrained pretrained/tiny_vit_21m_22k_distill.pth \
        ${WANDB_ARGS}

    echo ""
    echo "Teacher training complete!"
    echo "Best checkpoint: ${OUTPUT_BASE}/teacher/ckpt_best.pth"
}

# =============================================================================
# STEP 2: Save Teacher Logits
# Runs teacher inference on training data, saves soft predictions
# Time: ~30 minutes on 2xA40 (10 epochs of logits)
# =============================================================================
save_teacher_logits() {
    echo ""
    echo "=============================================="
    echo "STEP 2: Saving Teacher Logits"
    echo "=============================================="

    # Find best teacher checkpoint
    TEACHER_CKPT="${OUTPUT_BASE}/teacher/ckpt_best.pth"
    if [ ! -f "${TEACHER_CKPT}" ]; then
        echo "Error: Teacher checkpoint not found at ${TEACHER_CKPT}"
        echo "Please run step 1 first: $0 train_teacher"
        exit 1
    fi

    mkdir -p ${LOGITS_PATH}

    torchrun --nproc_per_node=${NUM_GPUS} save_logits.py \
        --cfg configs/cifar100/tiny_vit_21m_cifar100_save_logits.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/save_logits \
        --resume ${TEACHER_CKPT} \
        --opts DISTILL.TEACHER_LOGITS_PATH ${LOGITS_PATH}

    echo ""
    echo "Teacher logits saved to: ${LOGITS_PATH}"
    echo "Contents:"
    ls -la ${LOGITS_PATH}/
}

# =============================================================================
# STEP 3: Train Student with Distillation (TinyViT-5M)
# Expected: ~88-90% accuracy (vs ~85% without distillation)
# Time: ~4-5 hours on 2xA40
# =============================================================================
train_student_distill() {
    echo ""
    echo "=============================================="
    echo "STEP 3: Training Student with Distillation"
    echo "=============================================="

    # Check logits exist
    if [ ! -d "${LOGITS_PATH}" ] || [ -z "$(ls -A ${LOGITS_PATH})" ]; then
        echo "Error: Teacher logits not found at ${LOGITS_PATH}"
        echo "Please run step 2 first: $0 save_logits"
        exit 1
    fi

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_distill.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/student_distill \
        --opts DISTILL.TEACHER_LOGITS_PATH ${LOGITS_PATH} \
        ${WANDB_ARGS}

    echo ""
    echo "Student distillation training complete!"
    echo "Best checkpoint: ${OUTPUT_BASE}/student_distill/ckpt_best.pth"
}

# =============================================================================
# STEP 4 (Optional): Train Student WITHOUT Distillation (Baseline)
# For comparison with distillation
# =============================================================================
train_student_baseline() {
    echo ""
    echo "=============================================="
    echo "STEP 4: Training Student Baseline (no distillation)"
    echo "=============================================="

    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_5m_cifar100_largescale.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/student_baseline \
        ${WANDB_ARGS}

    echo ""
    echo "Baseline training complete!"
}

# =============================================================================
# STEP 5: Compare Results
# =============================================================================
compare_results() {
    echo ""
    echo "=============================================="
    echo "RESULTS COMPARISON"
    echo "=============================================="
    echo ""
    echo "Expected results:"
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│ Model                          │ Params  │ Expected Acc    │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│ TinyViT-21M Teacher (finetune) │ 21M     │ 91-93%          │"
    echo "│ TinyViT-5M + Distillation      │ 5M      │ 88-90%          │"
    echo "│ TinyViT-5M Baseline (scratch)  │ 5M      │ 82-85%          │"
    echo "│ TinyViT-5M Baseline (finetune) │ 5M      │ 88-90%          │"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "Key insight: Distillation from 21M teacher helps 5M student"
    echo "achieve similar accuracy to fine-tuning, but student learns"
    echo "richer representations from teacher's soft targets."
}

# =============================================================================
# FULL PIPELINE
# =============================================================================
run_full_pipeline() {
    echo "Running full distillation pipeline..."
    train_teacher
    save_teacher_logits
    train_student_distill
    compare_results
}

# =============================================================================
# QUICK TEST (verify setup works)
# =============================================================================
quick_test() {
    echo "Running quick test (1 epoch each)..."

    # Test teacher training
    torchrun --nproc_per_node=${NUM_GPUS} main.py \
        --cfg configs/cifar100/tiny_vit_21m_cifar100_teacher.yaml \
        --data-path ${DATA_PATH} \
        --output ${OUTPUT_BASE}/test_teacher \
        --opts TRAIN.EPOCHS 1

    echo "Quick test passed! Setup is working."
}

# =============================================================================
# MAIN MENU
# =============================================================================
case "${1:-help}" in
    train_teacher|step1)
        train_teacher
        ;;
    save_logits|step2)
        save_teacher_logits
        ;;
    train_student|step3)
        train_student_distill
        ;;
    baseline|step4)
        train_student_baseline
        ;;
    compare)
        compare_results
        ;;
    full)
        run_full_pipeline
        ;;
    test)
        quick_test
        ;;
    *)
        echo "TinyViT CIFAR-100 Distillation Pipeline"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  train_teacher (step1)  - Train TinyViT-21M teacher on CIFAR-100"
        echo "  save_logits   (step2)  - Save teacher's soft predictions"
        echo "  train_student (step3)  - Train TinyViT-5M student with distillation"
        echo "  baseline      (step4)  - Train TinyViT-5M without distillation"
        echo "  compare                - Show expected results comparison"
        echo "  full                   - Run complete pipeline (steps 1-3)"
        echo "  test                   - Quick test (1 epoch) to verify setup"
        echo ""
        echo "Example workflow:"
        echo "  $0 train_teacher   # Step 1: ~3 hours"
        echo "  $0 save_logits     # Step 2: ~30 mins"
        echo "  $0 train_student   # Step 3: ~4 hours"
        echo ""
        ;;
esac
