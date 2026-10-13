param([switch]$NoPreload)

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project Python is missing. Create the virtual environment and install requirements first.'
}

Set-Location -LiteralPath $projectRoot
$env:PHISHGUARD_PRELOAD_EMAIL = if ($NoPreload) { '0' } else { '1' }
# One process on this 8 GB laptop; reload would add a watcher/process restart.
& $pythonPath -u -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
exit $LASTEXITCODE
