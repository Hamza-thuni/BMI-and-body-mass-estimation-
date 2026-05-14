@echo off
echo ====================================================
echo   BMI & Body Mass Estimation - Laptop Deployment
echo ====================================================
echo.

cd /d "%~dp0"

:: 1. Check if virtual environment exists
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

:: 2. Activate and install
echo [2/3] Activating environment and checking dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

:: 3. Run the app
echo [3/3] Starting Demo App...
echo.
echo Application will open in your browser automatically.
echo Press Ctrl+C in this window to stop the server.
echo.
python demo_app.py

pause
