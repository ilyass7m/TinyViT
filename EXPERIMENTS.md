# TinyViT CIFAR-100 Distillation Experiments

This document describes all experiments for reproducing TinyViT distillation results on CIFAR-100.

## Project Overview

**Goal**: Reproduce TinyViT offline distillation on CIFAR-100 and extend with teacher architecture comparisons.

**Paper Reference**: [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666)

## Fixed Global Settings

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-100 |
| Input Resolution | 224×224 |
| Training Epochs | 200 |
| Optimizer | AdamW |
| LR Schedule | Cosine with warmup |
| Random Seed | 42 |
| Metric | Top-1 Accuracy |

## Model Summary

| Model | Type | Params | Role |
|-------|------|--------|------|
| TinyViT-5M | Student | ~5.4M | Student model for all experiments |
| TinyViT-21M | Teacher | ~21M | Same-family teacher |
| ViT-Base | Teacher | ~86M | Transformer teacher |
| ResNet-152 | Teacher | ~60M | CNN teacher |

---

## Experiment Overview

### Phase 1: Baselines

| Exp | Name | Purpose | Config |
|-----|------|---------|--------|
| 1 | Scratch Training | Lower bound baseline | `exp1_student_scratch.yaml` |
| 2 | Transfer Learning | Extension baseline (not in paper) | `exp2_student_finetune.yaml` |

### Phase 2: Teacher Preparation

| Exp | Name | Purpose | Config |
|-----|------|---------|--------|
| 3a | ResNet-152 Teacher | CNN teacher | `exp3_teacher_resnet152.yaml` |
| 3b | ViT-Base Teacher | Transformer teacher | `exp3_teacher_vit_base.yaml` |
| 3c | TinyViT-21M Teacher | Same-family teacher | `exp3_teacher_tinyvit21m.yaml` |
| 4 | Save Logits | Save teacher predictions | `exp4_save_logits_*.yaml` |

### Phase 3: Core Distillation

| Exp | Name | Purpose | Config |
|-----|------|---------|--------|
| 5 | TinyViT-21M Distill | Core paper reproduction | `exp5_distill_tinyvit21m.yaml` |

### Phase 4: Extensions

| Exp | Name | Purpose | Config |
|-----|------|---------|--------|
| 6a | ResNet-152 Distill | CNN teacher comparison | `exp6_distill_resnet152.yaml` |
| 6b | ViT-Base Distill | Transformer teacher comparison | `exp6_distill_vit_base.yaml` |
| 7 | K Ablation | Logit sparsity study | `exp7_ablation_k*.yaml` |

### Phase 5: Optional

| Exp | Name | Purpose | Config |
|-----|------|---------|--------|
| 8 | Transfer + Distill | Do benefits stack? | `exp8_transfer_distill.yaml` |

---

## Quick Start Commands

### Running Single Experiments (Single GPU)

```bash
# Exp-1: Scratch Training
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp1_student_scratch.yaml \
    --data-path ./data \
    --output ./output/exp1_scratch
```

### Running Two Experiments in Parallel (Different GPUs)

```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp1_student_scratch.yaml \
    --data-path ./data --output ./output/exp1_scratch

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 main.py \
    --cfg configs/cifar100/experiments/exp6_distill_vit_base.yaml \
    --data-path ./data --output ./output/exp6_distill_vit \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/vit_base/
```

---

## Detailed Experiment Instructions

### Experiment 1: Scratch Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp1_student_scratch.yaml \
    --data-path ./data \
    --output ./output/exp1_scratch
```

**Expected**: 75-82% accuracy

### Experiment 2: Transfer Learning

```bash
# Download pretrained weights first
mkdir -p pretrained
wget -O pretrained/tiny_vit_5m_22k_distill.pth \
    https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp2_student_finetune.yaml \
    --data-path ./data \
    --output ./output/exp2_finetune \
    --pretrained pretrained/tiny_vit_5m_22k_distill.pth
```

**Expected**: 85-88% accuracy

### Experiment 3: Train Teachers

**3a - ResNet-152:**
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp3_teacher_resnet152.yaml \
    --data-path ./data \
    --output ./output/exp3_teacher_resnet152
```

**3b - ViT-Base:**
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp3_teacher_vit_base.yaml \
    --data-path ./data \
    --output ./output/exp3_teacher_vit_base
```

**3c - TinyViT-21M:**
```bash
wget -O pretrained/tiny_vit_21m_22k_distill.pth \
    https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp3_teacher_tinyvit21m.yaml \
    --data-path ./data \
    --output ./output/exp3_teacher_tinyvit21m \
    --pretrained pretrained/tiny_vit_21m_22k_distill.pth
```

**Expected Teacher Accuracy**: 88-92%

### Experiment 4: Save Teacher Logits

For each teacher, run save_logits.py:

```bash
# ResNet-152 logits
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 save_logits.py \
    --cfg configs/cifar100/experiments/exp4_save_logits_resnet152.yaml \
    --data-path ./data \
    --output ./output/exp4_logits \
    --resume ./output/exp3_teacher_resnet152/Exp3-ResNet152-Teacher/default/ckpt_best.pth \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/resnet152/

# ViT-Base logits
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 save_logits.py \
    --cfg configs/cifar100/experiments/exp4_save_logits_vit_base.yaml \
    --data-path ./data \
    --output ./output/exp4_logits \
    --resume ./output/exp3_teacher_vit_base/Exp3-ViTBase-Teacher/default/ckpt_best.pth \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/vit_base/

# TinyViT-21M logits
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 save_logits.py \
    --cfg configs/cifar100/experiments/exp4_save_logits_tinyvit21m.yaml \
    --data-path ./data \
    --output ./output/exp4_logits \
    --resume ./output/exp3_teacher_tinyvit21m/Exp3-TinyViT21M-Teacher/default/ckpt_best.pth \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/tinyvit21m/
```

### Experiment 5 & 6: Distillation

```bash
# Exp-5: TinyViT-21M teacher (core reproduction)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp5_distill_tinyvit21m.yaml \
    --data-path ./data \
    --output ./output/exp5_distill_tinyvit21m \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/tinyvit21m/

# Exp-6a: ResNet-152 teacher
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp6_distill_resnet152.yaml \
    --data-path ./data \
    --output ./output/exp6_distill_resnet152 \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/resnet152/

# Exp-6b: ViT-Base teacher
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp6_distill_vit_base.yaml \
    --data-path ./data \
    --output ./output/exp6_distill_vit_base \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/vit_base/
```

### Experiment 7: K Ablation

First save logits with different K values, then train:

```bash
# Save logits with K=10
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 save_logits.py \
    --cfg configs/cifar100/experiments/exp4_save_logits_tinyvit21m.yaml \
    --data-path ./data \
    --resume ./output/exp3_teacher_tinyvit21m/Exp3-TinyViT21M-Teacher/default/ckpt_best.pth \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/tinyvit21m_k10/ DISTILL.LOGITS_TOPK 10

# Train with K=10
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 main.py \
    --cfg configs/cifar100/experiments/exp7_ablation_k10.yaml \
    --data-path ./data \
    --output ./output/exp7_ablation_k10 \
    --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/tinyvit21m_k10/
```

---

## Expected Results Summary

| Experiment | Method | Expected Acc |
|------------|--------|--------------|
| Exp-1 | Scratch | 75-82% |
| Exp-2 | Transfer | 85-88% |
| Exp-5 | Distill (TinyViT-21M) | 84-87% |
| Exp-6a | Distill (ResNet-152) | 82-85% |
| Exp-6b | Distill (ViT-Base) | 84-87% |
| Exp-8 | Transfer + Distill | 87-90% |

---

## Output Directory Structure

```
output/
├── exp1_scratch/           # Experiment 1 results
├── exp2_finetune/          # Experiment 2 results
├── exp3_teacher_*/         # Teacher checkpoints
├── logits/                 # Saved teacher logits
│   ├── resnet152/
│   ├── vit_base/
│   └── tinyvit21m/
├── exp5_distill_*/         # Distillation results
├── exp6_distill_*/         # Teacher comparison results
└── exp7_ablation_*/        # K ablation results
```

---

## Code Structure

```
TinyViT/
├── main.py                 # Main training script
├── save_logits.py          # Save teacher logits
├── config.py               # Configuration system
├── models/
│   ├── build.py            # Model factory
│   └── tiny_vit.py         # TinyViT architecture
├── data/
│   ├── build.py            # Data loader builder
│   └── augmentation/
│       ├── dataset_wrapper.py  # Logits loading/saving
│       └── manager.py          # Binary logits storage
├── configs/
│   └── cifar100/
│       └── experiments/    # All experiment configs
└── output/                 # Results and checkpoints
```

## Key Implementation Details

### Distillation Loss
- Soft target cross-entropy: `loss = -sum(teacher_prob * log(student_prob))`
- Teacher probabilities from saved logits (top-K sparse format)
- Same augmentation seed replayed for consistency

### Logits Format
- Binary file: `[seed(4B) | indices(K×2B) | values(K×2B)]`
- 10 epochs of logits saved (different augmentations)
- Student training cycles through epochs with modulo
