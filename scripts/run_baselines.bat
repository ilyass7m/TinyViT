@echo off
REM ============================================================================
REM Phase 1: Baseline Experiments
REM ============================================================================
REM B1: Train TinyViT-5M from scratch (lower bound)
REM B2: Finetune pretrained TinyViT-5M (missing baseline from paper)
REM ============================================================================

set OUTPUT_DIR=output
set DATA_DIR=data

echo ============================================================================
echo Phase 1: Baseline Experiments
echo ============================================================================
echo.

echo [B1] Training TinyViT-5M from scratch...
echo     Purpose: Lower bound baseline
echo.
python experiments/run_experiment.py --exp B1 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [B2] Finetuning pretrained TinyViT-5M...
echo     Purpose: Transfer learning baseline (missing from paper)
echo.
python experiments/run_experiment.py --exp B2 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo Baseline experiments complete!
echo Results saved in: %OUTPUT_DIR%
echo ============================================================================
pause
