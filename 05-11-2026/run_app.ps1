# Streamlit App Launcher Script for PowerShell
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Launching Women's Clothing Trends Streamlit Cloud App..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
streamlit run app.py
