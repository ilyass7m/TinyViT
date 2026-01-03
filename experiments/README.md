# CIFAR-100 Experiments for TinyViT Reproduction

## Experimental Setup

This directory contains scripts to reproduce TinyViT results on CIFAR-100.

## Experiments Overview

### Phase 1: Baselines
- `exp1_scratch.py` - Train TinyViT from scratch
- `exp1_transfer.py` - Finetune ImageNet pretrained TinyViT

### Phase 2: Teacher Preparation
- `exp2_train_teacher.py` - Finetune teacher on CIFAR-100
- `exp2_save_logits.py` - Save teacher logits

### Phase 3: Distillation
- `exp3_distillation.py` - Train with saved logits

### Phase 4: Ablations
- `exp4_ablations.py` - Top-K, teacher quality, epochs

## Quick Start

```bash
# 1. Train baseline (from scratch)
python experiments/run_experiment.py --exp scratch --model tiny_vit_5m

# 2. Train baseline (transfer learning)
python experiments/run_experiment.py --exp transfer --model tiny_vit_5m

# 3. Prepare teacher and save logits
python experiments/run_experiment.py --exp teacher --model tiny_vit_21m
python experiments/run_experiment.py --exp save_logits --model tiny_vit_21m

# 4. Train with distillation
python experiments/run_experiment.py --exp distill --model tiny_vit_5m
```

## Expected Results

| Model | Scratch | Transfer | Distillation |
|-------|---------|----------|--------------|
| TinyViT-5M | ~83% | ~89% | ~87% |
| TinyViT-11M | ~85% | ~90% | ~89% |
