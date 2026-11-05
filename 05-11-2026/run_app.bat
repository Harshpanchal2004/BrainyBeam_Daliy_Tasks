@echo off
REM Streamlit App Launcher Script for Windows Environment
echo ========================================================
echo Launching Women's Clothing Trends Streamlit Cloud App...
echo ========================================================
cd /d "%~dp0"
streamlit run app.py
pause
