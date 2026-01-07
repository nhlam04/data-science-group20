@echo off
REM Quick start script for Windows

echo ===================================
echo Emotion Detection - Quick Start
echo ===================================

REM Check if virtual environment exists
if not exist "backend\venv" (
    echo Creating virtual environment...
    cd backend
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    cd ..
) else (
    echo Activating virtual environment...
    call backend\venv\Scripts\activate
)

REM Check if models exist
if not exist "models\classifiers" (
    echo.
    echo WARNING: Classifier models not found!
    echo Please:
    echo 1. Train models in Kaggle notebook
    echo 2. Run export_models.py in Kaggle
    echo 3. Download and extract to deployment\models\
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b
)

REM Check GPU
echo.
echo Checking GPU...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>nul
if errorlevel 1 (
    echo WARNING: nvidia-smi not found. GPU may not be available.
)

REM Start backend
echo.
echo Starting backend server...
echo API will be available at: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.

cd backend
start /b python main.py
cd ..

REM Wait for backend
timeout /t 5 /nobreak >nul

REM Start frontend
echo.
echo Starting frontend server...
echo Frontend will be available at: http://localhost:3000
echo.

cd frontend
start /b python -m http.server 3000
cd ..

timeout /t 2 /nobreak >nul

echo.
echo ===================================
echo Servers started successfully!
echo ===================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

pause
