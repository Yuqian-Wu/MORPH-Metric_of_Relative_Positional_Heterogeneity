@echo off
chcp 65001 >nul
REM MORPH Project Environment Setup Script
REM Graph-based Tactical Analytics Framework (G-TAF)
REM Sub-project 1: MORPH - Tactical Recognition
REM Last Updated: 2025-12-07

echo ========================================
echo MORPH Project Environment Setup
echo ========================================
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment MORPHenv...
if exist MORPHenv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv MORPHenv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call MORPHenv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies with Tsinghua mirror
echo [5/6] Installing dependencies (this may take several minutes)...
echo Using Tsinghua mirror for faster download...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install, please check error messages
)
echo.

REM Install unravelsports from GitHub
echo [5.1/6] Installing unravelsports from GitHub...
pip install git+https://github.com/UnravelSports/unravelsports.git
if %errorlevel% neq 0 (
    echo WARNING: unravelsports installation failed, please install manually
)
echo.

REM Configure Jupyter kernel
echo [6/6] Configuring Jupyter kernel...
python -m ipykernel install --user --name=MORPHenv --display-name="Python (MORPH)"
if %errorlevel% neq 0 (
    echo WARNING: Jupyter kernel configuration failed
) else (
    echo Jupyter kernel configured successfully
)
echo.

echo ========================================
echo Environment Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Open Jupyter Notebook in VSCode
echo 2. Select kernel: "Python (MORPH)"
echo 3. Run: Step1_Contextualization_Scaling/Test/1.1_test_Convert_TrackingData.ipynb
echo.
echo To manually activate environment, run:
echo     MORPHenv\Scripts\activate.bat
echo.
pause