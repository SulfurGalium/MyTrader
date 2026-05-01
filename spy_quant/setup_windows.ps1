# setup_windows.ps1
# Windows setup script for SPY Quant
# Run in PowerShell as Administrator:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  SPY Quant - Windows Setup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Python ────────────────────────────────────────────────────────────
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "  Found: $pyver" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install from https://python.org (3.10+)" -ForegroundColor Red
    exit 1
}

# ── 2. Create virtualenv ───────────────────────────────────────────────────────
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $ProjectDir "venv"
if (-Not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
    Write-Host "  Created: $VenvDir" -ForegroundColor Green
} else {
    Write-Host "  Already exists: $VenvDir" -ForegroundColor Green
}

# ── 3. Install dependencies ────────────────────────────────────────────────────
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
$Pip = Join-Path $VenvDir "Scripts\pip.exe"
& $Pip install --upgrade pip --quiet
& $Pip install -r (Join-Path $ProjectDir "requirements.txt") --quiet
Write-Host "  Dependencies installed." -ForegroundColor Green

# ── 4. Copy .env ───────────────────────────────────────────────────────────────
Write-Host "[4/6] Setting up environment file..." -ForegroundColor Yellow
$EnvFile    = Join-Path $ProjectDir ".env"
$EnvExample = Join-Path $ProjectDir ".env.example"
if (-Not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "  Created .env from template." -ForegroundColor Green
    Write-Host "  ⚠  Edit .env and add your API keys before running!" -ForegroundColor Yellow
} else {
    Write-Host "  .env already exists." -ForegroundColor Green
}

# ── 5. Create runtime directories ─────────────────────────────────────────────
Write-Host "[5/6] Creating runtime directories..." -ForegroundColor Yellow
foreach ($dir in @("data", "models", "cache", "logs", "backups")) {
    $path = Join-Path $ProjectDir $dir
    if (-Not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}
Write-Host "  Directories ready." -ForegroundColor Green

# ── 6. Register Task Scheduler job (optional) ──────────────────────────────────
Write-Host "[6/6] Task Scheduler setup..." -ForegroundColor Yellow
$TaskName   = "SPYQuantLive"
$PythonExe  = Join-Path $VenvDir "Scripts\python.exe"
$ScriptPath = Join-Path $ProjectDir "scripts\run_live.py"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Task '$TaskName' already registered." -ForegroundColor Green
} else {
    $action  = New-ScheduledTaskAction -Execute $PythonExe -Argument $ScriptPath -WorkingDirectory $ProjectDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Hours 0)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
    Write-Host "  Task '$TaskName' registered (runs at logon)." -ForegroundColor Green
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "  1. Edit .env  (add FRED_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)"
Write-Host "  2. Activate venv:   .\venv\Scripts\Activate.ps1"
Write-Host "  3. Train:           python scripts\train.py --alpaca"
Write-Host "  4. Evaluate:        python scripts\evaluate.py --alpaca"
Write-Host "  5. Paper trade:     python scripts\run_live.py"
Write-Host "  6. Dashboard:       python -m dashboard.server"
Write-Host "  7. Monitor:         python scripts\monitor.py --watch"
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
