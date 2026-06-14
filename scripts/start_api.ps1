$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found. Creating .venv..."
    py -m venv (Join-Path $ProjectRoot ".venv")
}

& $Python -m pip show agent-trading *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $Python -m pip install -e ".[dev,data]"
}

Write-Host "Starting AgentTrading API at http://127.0.0.1:8000"
& $Python -m uvicorn agent_trading.api.main:app --reload --host 127.0.0.1 --port 8000
