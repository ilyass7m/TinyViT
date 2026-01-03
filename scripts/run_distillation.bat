@echo off
REM ============================================================================
REM Phase 3: Core Distillation Experiments
REM ============================================================================
REM D1: TinyViT-21M -> TinyViT-5M (same-family, reproduce paper)
REM D2: ResNet-50 -> TinyViT-5M (cross-architecture: CNN to ViT)
REM D3: ViT-Base -> TinyViT-5M (different ViT family)
REM ============================================================================

set OUTPUT_DIR=output
set DATA_DIR=data

echo ============================================================================
echo Phase 3: Core Distillation Experiments
echo ============================================================================
echo.
echo Prerequisites: Teachers must be trained and logits saved (run_teachers.bat)
echo.

REM --- Check if logits exist ---
if not exist "%OUTPUT_DIR%\logits\tiny_vit_21m_top100" (
    echo ERROR: Logits not found. Run run_teachers.bat first!
    pause
    exit /b 1
)

echo [D1] Distillation: TinyViT-21M to TinyViT-5M
echo     Purpose: Reproduce core TinyViT paper claim (same-family distillation)
echo.
python experiments/run_experiment.py --exp D1 --logits-path %OUTPUT_DIR%/logits/tiny_vit_21m_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [D2] Distillation: ResNet-50 to TinyViT-5M
echo     Purpose: Cross-architecture transfer (CNN teaches ViT)
echo.
python experiments/run_experiment.py --exp D2 --logits-path %OUTPUT_DIR%/logits/resnet50_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo [D3] Distillation: ViT-Base to TinyViT-5M
echo     Purpose: Different ViT family comparison
echo.
python experiments/run_experiment.py --exp D3 --logits-path %OUTPUT_DIR%/logits/vit_base_patch16_224_top100 --output-dir %OUTPUT_DIR% --data-path %DATA_DIR%

echo.
echo ============================================================================
echo Phase 3 complete!
echo Compare: D1 vs D2 vs D3 for teacher architecture effect
echo Compare: D1 vs B1 for distillation benefit
echo Compare: D1 vs B2 for distillation vs transfer learning
echo ============================================================================
pause
