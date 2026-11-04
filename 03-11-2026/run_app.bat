@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
echo Starting Women's Clothing Streamlit Application...
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
