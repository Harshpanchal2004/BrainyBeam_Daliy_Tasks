$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
if (Test-Path "$ScriptDir\venv\Scripts\Activate.ps1") {
    & "$ScriptDir\venv\Scripts\Activate.ps1"
}
Write-Host "Starting Women's Clothing Streamlit Application (04-11-2026)..." -ForegroundColor Cyan
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
