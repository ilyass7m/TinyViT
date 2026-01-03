@echo off
REM ============================================================================
REM Phase 4: Ablation Experiments
REM ============================================================================
REM A1: Top-K=10 logit sparsity
REM A2: Top-K=50 logit sparsity
REM A3: Top-K=100 logit sparsity (default)
REM A4: Transfer learning + Distillation (benefit stacking)
REM ============================================================================

set OUTPUT_DIR=output
set DATA_DIR=data

echo ============================================================================
echo Phase 4: Ablation Experiments
echo ============================================================================
echo.
echo Prerequisites: Teachers must be trained and logits saved (run_teachers.bat)
echo.

REM --- Check if logits exist ---
if not exist "%OUTPUT_DIR%\logits\tiny_vit_21m_top10" (
    echo ERROR: Logits for K=10 not found. Run run_teachers.bat first!
    pause
    exit /b 1
)

echo ============================================================================
echo Ablation: Logit Sparsity (K = 10, 50, 100)
echo Purpose: How sparse can teacher logits be without hurting performance?
echo ============================================================================
echo.

echo [A1] Top-K=10 logit sparsity...
python experiments/run_experiment.py --exp A1 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top10 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [A2] Top-K=50 logit sparsity...
python experiments/run_experiment.py --exp A2 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top50 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [A3] Top-K=100 logit sparsity...
python experiments/run_experiment.py --exp A3 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo Ablation: Transfer + Distillation
echo Purpose: Do transfer learning and distillation benefits stack?
echo ============================================================================
echo.

echo [A4] Pretrained TinyViT-5M + TinyViT-21M distillation...
python experiments/run_experiment.py --exp A4 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo Phase 4 complete!
echo.
echo Comparisons:
echo   - A1 vs A2 vs A3: Effect of logit sparsity
echo   - A4 vs D1 vs B2: Benefit stacking (transfer + distill vs each alone)
echo ============================================================================
pause
