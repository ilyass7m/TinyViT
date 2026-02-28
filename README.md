# TinyViT: Fast Pretraining Distillation for Small Vision Transformers

This repository contains our reproduction and extension of the TinyViT paper for a CentraleSupélec Deep Learning course project.

**Original Paper**: [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf) (ECCV 2022)

**Authors of this reproduction**: Ilyas Madah, Moghit Yebari

---

## Overview

TinyViT is a family of compact vision transformers (5-21M parameters) trained using an efficient offline distillation framework. The key idea is to **pre-compute and store** sparse teacher logits, eliminating the teacher from the training loop entirely.



## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/TinyViT.git
cd TinyViT

# Create environment
conda create -n tinyvit python=3.9 -y
conda activate tinyvit

# Install dependencies
pip install torch torchvision torchaudio
pip install timm wandb yacs termcolor
```

---

## Reproduction Roadmap

Our experiments are organized into 4 parts matching the report structure.

### Part 1: CIFAR-100 Direct Training

Train TinyViT-5M on CIFAR-100 with different teachers.

```bash
# 1. Scratch Baseline
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part1_scratch_baseline.yaml \
  --data-path ./data --output output/P1_scratch_baseline

# 2. Train ViT-Base Teacher
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part1_teacher_vit_base.yaml \
  --data-path ./data --output output/P1_teacher_vit_base

# 3. Save Teacher Logits (TopK=50)
torchrun --nproc_per_node=1 save_logits.py \
  --cfg configs/cifar100/experiments/part1_save_logits_vit_base.yaml \
  --data-path ./data \
  --resume output/P1_teacher_vit_base/ViT-Base-CIFAR100-Teacher/default/ckpt_epoch_29.pth \
  --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/vit_base_top50/

# 4. Distill Student from Saved Logits
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part1_distill_vit_base.yaml \
  --data-path ./data --output output/P1_distill_vit_base \
  --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/vit_base_top50/
```

**TopK Ablation**: Change `DISTILL.LOGITS_TOPK` to 10, 20, 50, or 75 when saving logits.

### Part 2: ImageNet-1K Pretraining

Pretrain TinyViT-21M on ImageNet-1K with CLIP distillation.

```bash
# 1. Save CLIP Logits
torchrun --nproc_per_node=2 save_logits.py \
  --cfg configs/1k_distill/part2_save_logits_clip.yaml \
  --data-path /path/to/imagenet \
  --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/clip_vit_l/

# 2. Distill TinyViT-21M
torchrun --nproc_per_node=2 main.py \
  --cfg configs/1k_distill/part2_tinyvit21m_distill_clip.yaml \
  --data-path /path/to/imagenet \
  --output output/P2_tinyvit21m_distill_clip \
  --opts DISTILL.TEACHER_LOGITS_PATH ./output/logits/clip_vit_l/
```

### Part 3: Transfer Learning (IN-1K → CIFAR-100)

Fine-tune pretrained checkpoints on CIFAR-100.

```bash
# From Scratch Pretrain
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part3_finetune_from_scratch.yaml \
  --data-path ./data --output output/P3_finetune_from_scratch \
  --pretrained /path/to/tiny_vit_21m_1k.pth

# From Distill Pretrain
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part3_finetune_from_distill.yaml \
  --data-path ./data --output output/P3_finetune_from_distill \
  --pretrained output/P2_tinyvit21m_distill_clip/.../ckpt_best.pth
```

### Part 4: Online Feature Distillation (Extension)

Our extension: combine logit and feature distillation online.

```bash
# First, train TinyViT-21M teacher on CIFAR-100
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part1_teacher_resnet50.yaml \
  --data-path ./data --output output/P1_teacher_tinyvit21m \
  --opts MODEL.TYPE tiny_vit MODEL.NAME TinyViT-21M-Teacher \
         MODEL.TINY_VIT.EMBED_DIMS [96,192,384,576] \
         MODEL.TINY_VIT.NUM_HEADS [3,6,12,18]

# Online Distill with Logits + Features
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part4_online_distill.yaml \
  --data-path ./data --output output/P4_online_logits_features \
  --opts DISTILL.TEACHER_CHECKPOINT output/P1_teacher_tinyvit21m/.../ckpt_best.pth

# Logits Only (ablation)
torchrun --nproc_per_node=1 main.py \
  --cfg configs/cifar100/experiments/part4_online_distill.yaml \
  --data-path ./data --output output/P4_online_logits_only \
  --opts DISTILL.TEACHER_CHECKPOINT output/P1_teacher_tinyvit21m/.../ckpt_best.pth \
         DISTILL.FEATURE_ENABLED False
```

**Beta Ablation**: Change `DISTILL.FEATURE_WEIGHT` to 0.25, 0.5, or 1.0.

---

## Evaluation

Evaluate any checkpoint on CIFAR-100:

```bash
python evaluate.py \
  --checkpoint output/P1_distill_vit_base/.../ckpt_best.pth \
  --model-type tiny_vit_5m \
  --data-path ./data \
  --batch-size 64
```

For ViT-Base teacher:
```bash
python evaluate.py \
  --checkpoint output/P1_teacher_vit_base/.../ckpt_best.pth \
  --model-type vit_base \
  --data-path ./data
```

---

## Explainability (GradCAM)

Visualize attention patterns across models:

```bash
jupyter notebook explainability.ipynb
```

The notebook compares GradCAM heatmaps for:
- Teacher (ViT-Base or TinyViT-21M)
- Distilled Student (logits only)
- Distilled Student (logits + features)
- Scratch Baseline

---







## Citation

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

## Acknowledgments

Based on [microsoft/Cream/TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT). Our reproduction adds:
- CIFAR-100 experiment configs
- Online feature distillation extension
- GradCAM explainability analysis
- Standalone evaluation scripts
