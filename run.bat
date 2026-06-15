@echo off
echo ===================================================
echo   Starting Multimodal Stress Detection System
echo ===================================================

echo.
echo [0/2] Cleaning up existing processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
echo Done.

echo.
echo [1/2] Checking Python environment and starting Backend Server (Flask)...

:: Detect virtual environment
set PYTHON_CMD=python
if exist venv\Scripts\python.exe (
    set PYTHON_CMD=..\venv\Scripts\python.exe
    echo Found virtual environment in root: venv
) else if exist .venv\Scripts\python.exe (
    set PYTHON_CMD=..\.venv\Scripts\python.exe
    echo Found virtual environment in root: .venv
) else if exist backend\venv\Scripts\python.exe (
    set PYTHON_CMD=venv\Scripts\python.exe
    echo Found virtual environment in backend: venv
) else if exist backend\.venv\Scripts\python.exe (
    set PYTHON_CMD=.venv\Scripts\python.exe
    echo Found virtual environment in backend: .venv
) else (
    echo No virtual environment found. Using system global 'python'.
)

start "Backend Server" cmd /k "cd backend && %PYTHON_CMD% app.py"

echo.
echo [2/2] Starting Frontend Application (React)...
start "Frontend App" cmd /k "cd frontend && npm start"

echo.
echo ===================================================
echo   System Started Successfully!
echo   ---------------------------------------------
echo   - Frontend: http://localhost:3000
echo   - Backend:  http://localhost:5000
echo ===================================================
pause
