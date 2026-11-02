@echo off
setlocal enabledelayedexpansion
title AI Trading Copilot - TimesFM-3 & 7-Model Terminal

:: Ensure working directory is the script folder (regardless of how it was launched)
cd /d "%~dp0"

echo =====================================================================
echo           AI TRADING COPILOT -- 1-CLICK LAUNCHER
echo   Google TimesFM-3 Multivariate & 7-Engine AI Consensus Terminal
echo =====================================================================
echo.

:: 1. Detect Python (try 'python', then Windows 'py' launcher)
set "PYTHON_EXE="
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=py"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found on your system!
    echo.
    echo Please install Python 3.10, 3.11, or 3.12 (64-bit) from:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo IMPORTANT: During setup, be sure to check the box:
    echo   [x] "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

:: Show detected Python version
for /f "tokens=*" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set "PY_VER=%%v"
echo [1/3] Detected: %PY_VER%

:: 2. Check or Create Virtual Environment (.venv)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [2/3] Setting up Python virtual environment (.venv)...
    "%PYTHON_EXE%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment (.venv).
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully.
) else (
    echo [2/3] Virtual environment (.venv) ready.
)

:: 3. Check or Install Dependencies
if not exist ".venv\.installed" (
    echo.
    echo [*] Installing required AI & financial packages (first launch only)...
    echo     This may take 1-3 minutes. Please wait...
    echo.
    .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed!
        echo If you see an error related to LightGBM or XGBoost, install:
        echo   Microsoft Visual C++ Redistributable (x64):
        echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
        echo.
        pause
        exit /b 1
    )
    echo 1 > .venv\.installed
    echo.
    echo       All dependencies installed successfully!
)

:: 4. Launch Streamlit Application
echo.
echo =====================================================================
echo [3/3] Launching AI Trading Copilot on http://localhost:8501 ...
echo       Opening in your default web browser now.
echo       (Keep this command window open while using the app)
echo       (Press Ctrl+C in this window anytime to stop)
echo =====================================================================
echo.

.venv\Scripts\python.exe -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo [NOTE] The application closed or encountered an error.
    pause
)

