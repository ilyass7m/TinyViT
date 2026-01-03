@echo off
REM ============================================================================
REM Phase 2: Teacher Preparation
REM ============================================================================
REM T1: Finetune ResNet-50 (CNN teacher)
REM T2: Finetune ViT-Base (Transformer teacher)
REM T3: Finetune TinyViT-21M (Same-family teacher)
REM Then save logits for each teacher
REM ============================================================================

set OUTPUT_DIR=output
set DATA_DIR=data

echo ============================================================================
echo Phase 2: Teacher Preparation
echo ============================================================================
echo.

REM --- Train Teachers ---
echo [T1] Training ResNet-50 teacher...
echo     Architecture: CNN (from timm)
echo.
python experiments/run_experiment.py --exp T1 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [T2] Training ViT-Base teacher...
echo     Architecture: Vision Transformer (from timm)
echo.
python experiments/run_experiment.py --exp T2 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [T3] Training TinyViT-21M teacher...
echo     Architecture: TinyViT (same family as student)
echo.
python experiments/run_experiment.py --exp T3 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo Teachers trained! Now saving logits...
echo ============================================================================
echo.

REM --- Save Logits ---
echo Saving ResNet-50 logits (K=100)...
python experiments/run_experiment.py --save-logits T1 --teacher-checkpoint %OUTPUT_DIR%/T1_teacher_resnet50/best.pth --topk 100 --output-dir %OUTPUT_DIR%

echo.
echo Saving ViT-Base logits (K=100)...
python experiments/run_experiment.py --save-logits T2 --teacher-checkpoint %OUTPUT_DIR%/T2_teacher_vit_base/best.pth --topk 100 --output-dir %OUTPUT_DIR%

echo.
echo Saving TinyViT-21M logits (K=100)...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 100 --output-dir %OUTPUT_DIR%

REM Additional topk values for ablation study
echo.
echo Saving TinyViT-21M logits (K=10) for ablation...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 10 --output-dir %OUTPUT_DIR%

echo.
echo Saving TinyViT-21M logits (K=50) for ablation...
python experiments/run_experiment.py --save-logits T3 --teacher-checkpoint %OUTPUT_DIR%/T3_teacher_tinyvit21m/best.pth --topk 50 --output-dir %OUTPUT_DIR%

echo.
echo ============================================================================
echo Phase 2 complete!
echo Teacher checkpoints: %OUTPUT_DIR%/T1_*, T2_*, T3_*
echo Saved logits: %OUTPUT_DIR%/logits/
echo ============================================================================
pause
