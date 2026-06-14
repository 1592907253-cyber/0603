$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$WebRoot = Join-Path $ProjectRoot "web"
$Port = 5173

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts/start_api.ps1 first or create the environment manually."
}

if (-not (Test-Path $WebRoot)) {
    throw "Missing web directory: $WebRoot"
}

Write-Host "Starting AgentTrading Web at http://127.0.0.1:$Port"
Write-Host "Backend API should run separately at http://127.0.0.1:8000"
& $Python -m http.server $Port --bind 127.0.0.1 --directory $WebRoot
