$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run scripts/start_api.ps1 first or create the environment manually."
}

if (-not (Test-Path (Join-Path $ProjectRoot "web\index.html"))) {
    throw "Missing frontend entry: web/index.html"
}

if (-not (Test-Path (Join-Path $ProjectRoot "scripts\start_web.ps1"))) {
    throw "Missing frontend startup script: scripts/start_web.ps1"
}

& $Python -m compileall (Join-Path $ProjectRoot "src") (Join-Path $ProjectRoot "tests")
& $Python -m pytest

$WebRoot = Join-Path $ProjectRoot "web"
$Job = Start-Job -ScriptBlock {
    param($PythonPath, $Root)
    & $PythonPath -m http.server 5173 --bind 127.0.0.1 --directory $Root
} -ArgumentList $Python, $WebRoot

try {
    Start-Sleep -Seconds 2
    $Response = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -UseBasicParsing
    if ($Response.StatusCode -ne 200 -or -not $Response.Content.Contains("AgentTrading Pro")) {
        throw "Frontend health check failed."
    }
    Write-Host "Frontend health check passed at http://127.0.0.1:5173"
}
finally {
    Stop-Job $Job -ErrorAction SilentlyContinue
    Remove-Job $Job -ErrorAction SilentlyContinue
}
