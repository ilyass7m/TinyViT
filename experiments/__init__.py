"""
TinyViT CIFAR-100 Distillation Experiments

This package provides a unified framework for reproducing TinyViT paper results
and extending with additional baselines and teacher architecture comparisons.
"""

from .config import (
    ExperimentConfig,
    DataConfig,
    TrainingConfig,
    DistillationConfig,
    EXPERIMENTS,
    get_experiment_config,
)

__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "TrainingConfig",
    "DistillationConfig",
    "EXPERIMENTS",
    "get_experiment_config",
]
