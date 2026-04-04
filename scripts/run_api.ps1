# Start Medical AI API — run from Cursor terminal (PATH includes Conda).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:MEDICAL_AI_SKIP_RAG = "1"
$env:PYTHONPATH = $Root

function Start-WithPython([string] $Exe, [string[]] $Prefix) {
    if ($Prefix.Count -gt 0) {
        & $Exe @($Prefix + @("-c", "import uvicorn")) 2>$null
    }
    else {
        & $Exe -c "import uvicorn" 2>$null
    }
    if ($LASTEXITCODE -ne 0) { return $false }
    Write-Host "Or run: python launch_api.py" -ForegroundColor DarkGray
    Write-Host "Open http://127.0.0.1:8000/docs (leave this window open)" -ForegroundColor Green
    if ($Prefix.Count -gt 0) {
        & $Exe @($Prefix + @("-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"))
    }
    else {
        & $Exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    }
    return $true
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    if (Start-WithPython "python" @()) { exit 0 }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    if (Start-WithPython "py" @("-3")) { exit 0 }
}

$condaPy = Join-Path $env:USERPROFILE "anaconda3\python.exe"
if (Test-Path $condaPy) {
    if (Start-WithPython $condaPy @()) { exit 0 }
}

Write-Host "ERROR: Python/uvicorn not available. Run: pip install -r requirements.txt" -ForegroundColor Red
exit 1
