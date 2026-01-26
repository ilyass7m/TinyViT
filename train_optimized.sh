#!/bin/bash

# ============================================
# OPTIMAL TRAINING SCRIPT FOR 2x A40
# TinyViT-21M ImageNet-1K from Scratch
# ============================================

set -e

# --- NCCL (Required for this environment) ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# --- CUDA Performance ---
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

# --- CPU Threading ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# --- Debug Level (WARN for production, INFO for debugging) ---
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN

# --- Navigate to project ---
cd /workspace/TinyViT

echo "============================================"
echo "Starting TinyViT-21M Training"
echo "GPUs: 2x A40"
echo "Batch: 256/GPU × 2 = 512 effective"
echo "Epochs: 300"
echo "============================================"

# --- Launch Training ---
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    main.py \
    --cfg configs/1k_distill/tiny_vit_21m_1k_scratch.yaml \
    --data-path /data/ImageNet/ \
    --output ./output/tiny_vit_21m_1k_scratch \
    --use-wandb \
    --wandb-run-name SCRATCH-TinyViT21M-ImageNet1K-Pretrain

echo "============================================"
echo "Training complete!"
echo "============================================"
