@echo off
REM Automated Environment Setup Script
echo ========================================================
echo Installing Required Dependencies from requirements.txt...
echo ========================================================
cd /d "%~dp0"
pip install -r requirements.txt
echo Setup complete. You can now launch the application using run_app.bat
pause
