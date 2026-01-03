@echo off
REM ============================================================================
REM TinyViT CIFAR-100 Distillation - Complete Pipeline
REM ============================================================================
REM This script runs all experiments in order:
REM   Phase 1: Baselines (B1, B2)
REM   Phase 2: Teachers (T1, T2, T3) + Save Logits
REM   Phase 3: Distillation (D1, D2, D3)
REM   Phase 4: Ablations (A1, A2, A3, A4)
REM ============================================================================

set OUTPUT_DIR=output
set DATA_DIR=data

echo ============================================================================
echo TinyViT CIFAR-100 Distillation Experiments
echo ============================================================================
echo.

REM ============================================================================
REM Phase 1: Baselines
REM ============================================================================
echo [PHASE 1] Running Baselines...
echo.

echo [B1] Training TinyViT-5M from scratch...
python experiments/run_experiment.py --exp B1 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [B2] Finetuning pretrained TinyViT-5M...
python experiments/run_experiment.py --exp B2 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo Phase 1 Complete!
echo.

REM ============================================================================
REM Phase 2: Teachers
REM ============================================================================
echo [PHASE 2] Training Teachers...
echo.

echo [T1] Training ResNet-50 teacher...
python experiments/run_experiment.py --exp T1 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [T2] Training ViT-Base teacher...
python experiments/run_experiment.py --exp T2 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [T3] Training TinyViT-21M teacher...
python experiments/run_experiment.py --exp T3 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo Phase 2a Complete! Now saving logits...
echo.

REM Save logits for each teacher
echo Saving ResNet-50 logits (K=100)...
python experiments/run_experiment.py --save-logits T1 --teacher-checkpoint %OUTPUT_DIR%/T1_teacher_resnet50/best.pth --topk 100 --output-dir %OUTPUT_DIR%

echo Saving ViT-Base logits (K=100)...
python experiments/run_experiment.py --save-logits T2 --teacher-checkpoint %OUTPUT_DIR%/T2_teacher_vit_base/best.pth --topk 100 --output-dir %OUTPUT_DIR%

echo Saving TinyViT-21M logits (K=100)...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 100 --output-dir %OUTPUT_DIR%

REM Save additional topk for ablation
echo Saving TinyViT-21M logits (K=10)...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 10 --output-dir %OUTPUT_DIR%

echo Saving TinyViT-21M logits (K=50)...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 50 --output-dir %OUTPUT_DIR%

echo.
echo Phase 2 Complete!
echo.

REM ============================================================================
REM Phase 3: Core Distillation
REM ============================================================================
echo [PHASE 3] Running Distillation Experiments...
echo.

echo [D1] Distillation: TinyViT-21M to TinyViT-5M (same-family)...
python experiments/run_experiment.py --exp D1 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [D2] Distillation: ResNet-50 to TinyViT-5M (CNN teacher)...
python experiments/run_experiment.py --exp D2 --logits-path %OUTPUT_DIR%/logits/resnet50_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [D3] Distillation: ViT-Base to TinyViT-5M (ViT teacher)...
python experiments/run_experiment.py --exp D3 --logits-path %OUTPUT_DIR%/logits/vit_base_patch16_224_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo Phase 3 Complete!
echo.

REM ============================================================================
REM Phase 4: Ablations
REM ============================================================================
echo [PHASE 4] Running Ablation Experiments...
echo.

echo [A1] Ablation: Top-K=10 logit sparsity...
python experiments/run_experiment.py --exp A1 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top10 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [A2] Ablation: Top-K=50 logit sparsity...
python experiments/run_experiment.py --exp A2 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top50 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [A3] Ablation: Top-K=100 logit sparsity...
python experiments/run_experiment.py --exp A3 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo [A4] Ablation: Transfer + Distillation...
python experiments/run_experiment.py --exp A4 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo ALL EXPERIMENTS COMPLETE!
echo ============================================================================
echo.
echo Results are saved in: %OUTPUT_DIR%
echo.

pause
