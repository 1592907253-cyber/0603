$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts/start_api.ps1 first or create the environment manually."
}

& $Python -m compileall (Join-Path $ProjectRoot "src") (Join-Path $ProjectRoot "tests")
& $Python -m pytest
