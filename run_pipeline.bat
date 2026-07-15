@echo off
title Multimodal Stress Pipeline Runner
echo ==========================================================
echo   STRESS DETECTION PIPELINE INTEGRATED BENCHMARKING SUITE
echo ==========================================================
echo.

:: Detect virtual environment
set PYTHON_CMD=python
if exist venv\Scripts\python.exe (
    set PYTHON_CMD=venv\Scripts\python.exe
    echo [INFO] Detected virtual environment: venv
) else if exist .venv\Scripts\python.exe (
    set PYTHON_CMD=.venv\Scripts\python.exe
    echo [INFO] Detected virtual environment: .venv
) else (
    echo [WARNING] No virtual environment found. Using system global 'python'.
)
echo.

:: 1. Step 1: Modality Feature Extraction
set /p run_ext="[STEP 1/2] Do you want to run the Feature Extraction pipeline? (Y/N): "
if /i "%run_ext%"=="Y" (
    echo.
    echo Running feature extraction on GPU/CPU...
    %PYTHON_CMD% extract_stress_features.py
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Feature extraction failed. Exiting.
        pause
        exit /b %errorlevel%
    )
) else (
    echo [INFO] Skipping feature extraction. Using existing CSV stores.
)
echo.

:: 2. Step 2: Model Training & Evaluation
echo [STEP 2/2] Select the Cross-Validation mode to evaluate the 18 models:
echo   [1] Group K-Fold (5-Fold Subject-Independent - Safe, fast)
echo   [2] Random Split (5-Fold Stratified CV)
echo   [3] Full LOSO (65-Fold Leave-One-Subject-Out - True subject-wise evaluation)
echo.
set /p cv_choice="Enter your choice (1, 2, or 3): "

set MODE_FLAG=group_kfold
if "%cv_choice%"=="2" set MODE_FLAG=random_split
if "%cv_choice%"=="3" set MODE_FLAG=full_loso

echo.
echo Running training benchmarking suite with mode: %MODE_FLAG%...
echo.

%PYTHON_CMD% train_and_evaluate_all.py --mode %MODE_FLAG%
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model evaluation failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================================
echo   Pipeline Execution Completed Successfully!
echo   Outputs saved in: loso_evaluation_results/
echo ==========================================================
echo.
explorer loso_evaluation_results
pause
