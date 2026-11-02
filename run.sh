#!/usr/bin/env bash
# AI Trading Copilot -- 1-Click Launcher for macOS & Linux

set -e

# Change directory to script folder
cd "$(dirname "$0")"

echo "====================================================================="
echo "          AI TRADING COPILOT -- 1-CLICK LAUNCHER"
echo "  Google TimesFM-3 Multivariate & 7-Engine AI Consensus Terminal"
echo "====================================================================="
echo ""

# 1. Detect Python 3
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python was not found on your system!"
    echo "Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/"
    exit 1
fi

echo "[1/3] Detected: $($PY_CMD --version)"

# 2. Check or Create Virtual Environment
if [ ! -f ".venv/bin/python" ]; then
    echo "[2/3] Setting up Python virtual environment (.venv)..."
    $PY_CMD -m venv .venv
    echo "      Virtual environment created successfully."
else
    echo "[2/3] Virtual environment (.venv) ready."
fi

# 3. Check or Install Dependencies
if [ ! -f ".venv/.installed" ]; then
    echo ""
    echo "[*] Installing required AI & financial packages (first launch only)..."
    echo "    This may take 1-3 minutes. Please wait..."
    echo ""
    .venv/bin/python -m pip install --upgrade pip --quiet
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/.installed
    echo "      All dependencies installed successfully!"
fi

# 4. Launch Streamlit Application
echo ""
echo "====================================================================="
echo "[3/3] Launching AI Trading Copilot on http://localhost:8501 ..."
echo "      Opening in your default web browser now."
echo "      (Press Ctrl+C in this terminal anytime to stop)"
echo "====================================================================="
echo ""

.venv/bin/python -m streamlit run app.py

