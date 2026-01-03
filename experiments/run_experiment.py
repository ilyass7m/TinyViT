#!/usr/bin/env python
"""
Main Entry Point for TinyViT CIFAR-100 Distillation Experiments

Usage:
    # List available experiments
    python run_experiment.py --list

    # Run baseline experiments
    python run_experiment.py --exp B1                    # Train from scratch
    python run_experiment.py --exp B2                    # Transfer learning

    # Train teachers
    python run_experiment.py --exp T1                    # ResNet-50 teacher
    python run_experiment.py --exp T2                    # ViT-Base teacher
    python run_experiment.py --exp T3                    # TinyViT-21M teacher

    # Save teacher logits (after teacher is trained)
    python run_experiment.py --save-logits T3 --teacher-checkpoint output/T3_teacher_tinyvit21m/best.pth

    # Run distillation experiments
    python run_experiment.py --exp D1 --logits-path output/logits/tiny_vit_21m_top100
    python run_experiment.py --exp D2 --logits-path output/logits/resnet50_top100
    python run_experiment.py --exp D3 --logits-path output/logits/vit_base_patch16_224_top100

    # Ablation experiments
    python run_experiment.py --exp A1 --logits-path output/logits/tiny_vit_21m_top10   # K=10
    python run_experiment.py --exp A2 --logits-path output/logits/tiny_vit_21m_top50   # K=50
    python run_experiment.py --exp A3 --logits-path output/logits/tiny_vit_21m_top100  # K=100
    python run_experiment.py --exp A4 --logits-path output/logits/tiny_vit_21m_top100  # Transfer + Distill

    # Run all baselines
    python run_experiment.py --run-phase baselines

    # Run all teachers
    python run_experiment.py --run-phase teachers
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import (
    ExperimentConfig,
    TeacherType,
    get_experiment_config,
    list_experiments,
    EXPERIMENTS,
    DISTILLATION_EXPERIMENTS,
)
from experiments.trainer import train
from experiments.distiller import train_with_distillation, save_teacher_logits


def parse_args():
    parser = argparse.ArgumentParser(
        description='TinyViT CIFAR-100 Distillation Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Main commands
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment ID to run (B1, B2, T1-T3, D1-D3, A1-A4)')

    # Logits saving
    parser.add_argument('--save-logits', type=str, default=None,
                        help='Save logits for teacher (T1, T2, or T3)')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                        help='Path to finetuned teacher checkpoint')
    parser.add_argument('--topk', type=int, default=100,
                        help='Top-K logits to save (default: 100)')

    # Distillation paths
    parser.add_argument('--logits-path', type=str, default=None,
                        help='Path to saved teacher logits (for D1-D3, A1-A4)')

    # Run phases
    parser.add_argument('--run-phase', type=str, default=None,
                        choices=['baselines', 'teachers', 'distillation', 'ablations', 'all'],
                        help='Run all experiments in a phase')

    # Overrides
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')

    return parser.parse_args()


def print_experiment_list():
    """Print list of all available experiments."""
    experiments = list_experiments()

    print("\n" + "=" * 70)
    print("TinyViT CIFAR-100 Distillation Experiments")
    print("=" * 70)

    print("\n--- BASELINES ---")
    for exp_id in ['B1', 'B2']:
        print(f"  {exp_id}: {experiments[exp_id]}")

    print("\n--- TEACHERS ---")
    for exp_id in ['T1', 'T2', 'T3']:
        print(f"  {exp_id}: {experiments[exp_id]}")

    print("\n--- DISTILLATION ---")
    for exp_id in ['D1', 'D2', 'D3']:
        print(f"  {exp_id}: {experiments[exp_id]}")

    print("\n--- ABLATIONS ---")
    for exp_id in ['A1', 'A2', 'A3', 'A4']:
        print(f"  {exp_id}: {experiments[exp_id]}")

    print("\n" + "=" * 70)
    print("\nUsage Examples:")
    print("  python run_experiment.py --exp B1              # Train from scratch")
    print("  python run_experiment.py --exp T3              # Train TinyViT-21M teacher")
    print("  python run_experiment.py --save-logits T3 --teacher-checkpoint output/T3_.../best.pth")
    print("  python run_experiment.py --exp D1 --logits-path output/logits/...")
    print("")


def apply_overrides(config: ExperimentConfig, args) -> ExperimentConfig:
    """Apply command-line overrides to config."""
    config.output_dir = args.output_dir
    config.data.data_path = args.data_path
    config.seed = args.seed
    config.device = args.device

    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs

    return config


def run_experiment(args):
    """Run a single experiment."""
    exp_id = args.exp

    print(f"\n{'=' * 70}")
    print(f"Running Experiment: {exp_id}")
    print(f"{'=' * 70}")

    # Get experiment config
    try:
        config = get_experiment_config(
            exp_id=exp_id,
            logits_path=args.logits_path,
            topk=args.topk if exp_id in ['A1', 'A2', 'A3'] else None,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Apply overrides
    config = apply_overrides(config, args)

    # Run appropriate training function
    if config.distillation.enabled and config.distillation.logits_path:
        results = train_with_distillation(config)
    else:
        results = train(config)

    print(f"\n{'=' * 70}")
    print(f"Experiment {exp_id} Complete!")
    print(f"Best Accuracy: {results['best_acc']:.2f}%")
    print(f"Output: {results['output_path']}")
    print(f"{'=' * 70}")

    return results


def run_save_logits(args):
    """Save teacher logits for distillation."""
    teacher_id = args.save_logits

    if teacher_id not in ['T1', 'T2', 'T3']:
        print(f"Error: Invalid teacher ID '{teacher_id}'. Use T1, T2, or T3.")
        return

    if not args.teacher_checkpoint:
        print("Error: --teacher-checkpoint is required for saving logits")
        return

    if not os.path.exists(args.teacher_checkpoint):
        print(f"Error: Teacher checkpoint not found: {args.teacher_checkpoint}")
        return

    # Get teacher config
    config = get_experiment_config(teacher_id)
    config = apply_overrides(config, args)
    config.distillation.topk = args.topk

    print(f"\n{'=' * 70}")
    print(f"Saving Teacher Logits: {config.teacher_type.value}")
    print(f"Top-K: {args.topk}")
    print(f"{'=' * 70}")

    logits_path = save_teacher_logits(config, args.teacher_checkpoint)

    print(f"\n{'=' * 70}")
    print(f"Logits saved to: {logits_path}")
    print(f"{'=' * 70}")


def run_phase(args):
    """Run all experiments in a phase."""
    phase = args.run_phase

    if phase == 'baselines':
        experiments = ['B1', 'B2']
    elif phase == 'teachers':
        experiments = ['T1', 'T2', 'T3']
    elif phase == 'distillation':
        experiments = ['D1', 'D2', 'D3']
    elif phase == 'ablations':
        experiments = ['A1', 'A2', 'A3', 'A4']
    elif phase == 'all':
        experiments = ['B1', 'B2', 'T1', 'T2', 'T3']
    else:
        print(f"Error: Unknown phase '{phase}'")
        return

    print(f"\n{'=' * 70}")
    print(f"Running Phase: {phase.upper()}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"{'=' * 70}")

    results = {}

    for exp_id in experiments:
        # Check if distillation experiment needs logits path
        if exp_id in ['D1', 'D2', 'D3', 'A1', 'A2', 'A3', 'A4']:
            if not args.logits_path:
                print(f"Skipping {exp_id}: --logits-path required")
                continue

        args.exp = exp_id
        try:
            result = run_experiment(args)
            results[exp_id] = result
        except Exception as e:
            print(f"Error running {exp_id}: {e}")
            results[exp_id] = {'error': str(e)}

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Phase {phase.upper()} Complete!")
    print(f"{'=' * 70}")
    print("\nResults Summary:")
    for exp_id, result in results.items():
        if 'error' in result:
            print(f"  {exp_id}: ERROR - {result['error']}")
        else:
            print(f"  {exp_id}: {result['best_acc']:.2f}%")


def main():
    args = parse_args()

    # List experiments
    if args.list:
        print_experiment_list()
        return

    # Save logits
    if args.save_logits:
        run_save_logits(args)
        return

    # Run phase
    if args.run_phase:
        run_phase(args)
        return

    # Run single experiment
    if args.exp:
        run_experiment(args)
        return

    # No command specified
    print("No command specified. Use --help for usage information.")
    print_experiment_list()


if __name__ == '__main__':
    main()
