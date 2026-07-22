@echo off
title SSVB-CASA-AIS Production Pipeline Runner
echo ==========================================================
echo   SSVB-CASA-AIS PRODUCTION PIPELINE (PHASE 8)
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

:: 1. Step 1: Feature Extraction
echo [STEP 1/2] Running Feature Extraction Pipeline...
echo            (This process uses CPU-bound libraries like OpenCV and Librosa)
%PYTHON_CMD% webapp\training\phase8\feature_extraction_service.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Feature extraction failed. Exiting pipeline.
    pause
    exit /b %errorlevel%
)
echo.

:: 2. Step 2: Model Architecture and Training
echo [STEP 2/2] Running SSVB-CASA-AIS Model Training...
echo            (This process will automatically use your GPU via CUDA if available, 
echo             otherwise it will fall back to CPU)
%PYTHON_CMD% webapp\training\phase8\train_ssvb_production.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================================
echo   Pipeline Execution Completed Successfully!
echo ==========================================================
echo.
pause
