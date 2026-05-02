# setup_windows.ps1
# Windows setup script for SPY Quant
# Run in PowerShell:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup_windows.ps1

# Use Continue so a single step failing doesn't abort everything.
# Individual critical steps re-throw their own errors explicitly.
$ErrorActionPreference = "Continue"
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
    Write-Host "  ERROR: Python not found. Install 3.10-3.12 from https://python.org" -ForegroundColor Red
    exit 1
}

# Warn on Python 3.13+ — deps work but PyTorch wheels may lag behind releases.
# Python 3.14 specifically: numpy 1.x has no binary wheel and will fail to
# compile from source without MSVC (Visual Studio Build Tools). requirements.txt
# now allows numpy 2.x which ships 3.14 wheels, so this is informational only.
$verMatch = [regex]::Match($pyver, '(\d+)\.(\d+)')
if ($verMatch.Success) {
    $major = [int]$verMatch.Groups[1].Value
    $minor = [int]$verMatch.Groups[2].Value
    if ($major -eq 3 -and $minor -ge 13) {
        Write-Host ""
        Write-Host "  ⚠  Python $major.$minor detected." -ForegroundColor Yellow
        Write-Host "     Most packages support 3.10-3.12 best." -ForegroundColor Yellow
        Write-Host "     If installation fails, install Python 3.12 from python.org" -ForegroundColor Yellow
        Write-Host "     and re-run this script." -ForegroundColor Yellow
        Write-Host ""
    }
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
$Pip    = Join-Path $VenvDir "Scripts\pip.exe"
$Python = Join-Path $VenvDir "Scripts\python.exe"

# Upgrade pip first using python -m pip to avoid the "can't modify pip" error
& $Python -m pip install --upgrade pip --quiet

# Install packages. On Python 3.14 numpy 2.x will be resolved automatically.
# PyTorch is NOT in requirements.txt — install it separately after setup:
#   CPU:   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   CUDA:  pip install torch --index-url https://download.pytorch.org/whl/cu121
& $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ⚠  Some packages failed to install." -ForegroundColor Yellow
    Write-Host "     Common fix: install Python 3.12 from https://python.org" -ForegroundColor Yellow
    Write-Host "     then delete the 'venv' folder and re-run this script." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  Dependencies installed." -ForegroundColor Green
}

# ── 4. .env.example + .env ────────────────────────────────────────────────────
Write-Host "[4/6] Setting up environment file..." -ForegroundColor Yellow
$EnvFile    = Join-Path $ProjectDir ".env"
$EnvExample = Join-Path $ProjectDir ".env.example"

# Generate .env.example if it wasn't committed to the repo (mirrors deploy.sh fix)
if (-Not (Test-Path $EnvExample)) {
    Write-Host "  .env.example not found — generating from template..." -ForegroundColor Yellow
    $envTemplate = @"
# ── Alpaca credentials ──────────────────────────────────────────────────────
# Paper trading (default safe): https://paper-api.alpaca.markets
# Live trading:                 https://api.alpaca.markets
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ── FRED (macro data) ───────────────────────────────────────────────────────
# Free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# ── Safety gate ─────────────────────────────────────────────────────────────
# Keep false until fully validated in paper trading
LIVE_TRADING_ENABLED=false

# ── Hardware ────────────────────────────────────────────────────────────────
# "cuda" (NVIDIA GPU), "mps" (Apple Silicon), or "cpu"
DEVICE=cpu

# ── Model / training ────────────────────────────────────────────────────────
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=0.0003

# Target horizon: bars ahead to predict (6 = 30-min forward return, recommended)
TARGET_HORIZON=6

# ── Risk parameters ─────────────────────────────────────────────────────────
MAX_POSITION_RISK=0.01
STOP_LOSS_PCT=0.005
TAKE_PROFIT_PCT=0.015

# ── Optional: override runtime directories (leave blank for defaults) ────────
DATA_DIR=
CACHE_DIR=
MODEL_DIR=
LOG_DIR=
"@
    Set-Content -Path $EnvExample -Value $envTemplate -Encoding UTF8
    Write-Host "  .env.example generated." -ForegroundColor Green
}

if (-Not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "  Created .env from template." -ForegroundColor Green
    Write-Host "  ⚠  Edit .env and add your API keys before running!" -ForegroundColor Yellow
} else {
    Write-Host "  .env already exists — leaving untouched." -ForegroundColor Green
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
    try {
        $action   = New-ScheduledTaskAction -Execute $PythonExe -Argument $ScriptPath -WorkingDirectory $ProjectDir
        $trigger  = New-ScheduledTaskTrigger -AtLogOn
        $settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Hours 0)
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
        Write-Host "  Task '$TaskName' registered (runs at logon)." -ForegroundColor Green
    } catch {
        Write-Host "  Task Scheduler skipped (run as Administrator to register)." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  IMPORTANT — install PyTorch next (not in requirements.txt):" -ForegroundColor Yellow
Write-Host "    CPU only:"
Write-Host "      .\venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu"
Write-Host "    NVIDIA GPU (CUDA 12.1):"
Write-Host "      .\venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121"
Write-Host ""
Write-Host "  Then:" -ForegroundColor White
Write-Host "  1. Edit .env  (add FRED_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)"
Write-Host "  2. Activate venv:   .\venv\Scripts\Activate.ps1"
Write-Host "  3. Train:           python scripts\train.py --alpaca"
Write-Host "  4. Evaluate:        python scripts\evaluate.py --alpaca"
Write-Host "  5. Paper trade:     python scripts\run_live.py"
Write-Host "  6. Dashboard:       python -m dashboard.server"
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
