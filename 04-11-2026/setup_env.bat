@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo Setting Up Isolated Virtual Environment (04-11-2026)
echo ========================================================
if not exist "venv" (
    echo Creating virtual environment in .\venv...
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ========================================================
echo Environment setup completed successfully!
echo Run 'run_app.bat' to launch the Streamlit dashboard.
echo ========================================================
pause
