@echo off
:: ============================================================
:: AskMyPDF — Windows Startup Script
:: ============================================================

echo.
echo  ╔══════════════════════════════════════╗
echo  ║         🧠  AskMyPDF v4.0           ║
echo  ║   OCR-Powered RAG Chatbot            ║
echo  ╚══════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check .env
if not exist ".env" (
    echo [WARN]  .env file not found. Copying from .env.example ...
    copy ".env.example" ".env" >nul
    echo [INFO]  Please edit .env and add your MISTRAL_API_KEY, then re-run this script.
    pause
    exit /b 1
)

:: Install / upgrade dependencies
echo [INFO]  Installing dependencies ...
pip install -r requirements.txt --quiet

:: Set Python path so `app` package is importable
set PYTHONPATH=%CD%

:: Launch
echo [INFO]  Starting AskMyPDF ...
echo [INFO]  Open http://localhost:8501 in your browser
echo.
streamlit run app/main.py --server.headless false

pause
