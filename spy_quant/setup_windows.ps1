# setup_windows.ps1
# Windows setup script for SPY Quant
# Run in PowerShell:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\setup_windows.ps1

# Continue so a single step doesn't abort everything.
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
    $pyver = & python --version 2>&1
    Write-Host "  Found: $pyver" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install 3.10-3.12 from https://python.org" -ForegroundColor Red
    exit 1
}

# Warn on Python 3.13+ (numpy 1.x won't compile; requirements.txt now allows 2.x so it should still work)
$verMatch = [regex]::Match($pyver, '(\d+)\.(\d+)')
if ($verMatch.Success) {
    $major = [int]$verMatch.Groups[1].Value
    $minor = [int]$verMatch.Groups[2].Value
    if ($major -eq 3 -and $minor -ge 13) {
        Write-Host ""
        Write-Host "  WARNING: Python $major.$minor detected." -ForegroundColor Yellow
        Write-Host "  Packages support 3.10-3.12 best. If install fails," -ForegroundColor Yellow
        Write-Host "  install Python 3.12 from python.org and re-run." -ForegroundColor Yellow
        Write-Host ""
    }
}

# ── 2. Check for NVIDIA GPU ────────────────────────────────────────────────────
Write-Host "[1b] Checking for NVIDIA GPU..." -ForegroundColor Yellow
$HasCuda = $false
try {
    $nvOut = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0 -and $nvOut) {
        Write-Host "  GPU detected: $nvOut" -ForegroundColor Green
        $HasCuda = $true
    } else {
        Write-Host "  No NVIDIA GPU found - will use CPU" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  nvidia-smi not found - will use CPU" -ForegroundColor Yellow
}

# ── 3. Create virtualenv ───────────────────────────────────────────────────────
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $ProjectDir "venv"
if (-Not (Test-Path $VenvDir)) {
    & python -m venv $VenvDir
    Write-Host "  Created: $VenvDir" -ForegroundColor Green
} else {
    Write-Host "  Already exists: $VenvDir" -ForegroundColor Green
}

$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

# ── 4. Install dependencies ────────────────────────────────────────────────────
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow

# Upgrade pip via python -m pip (avoids the "can't modify pip while running" error)
& $PythonExe -m pip install --upgrade pip --quiet

# Install project deps (PyTorch is NOT included - installed separately below)
& $PythonExe -m pip install -r (Join-Path $ProjectDir "requirements.txt")

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  WARNING: Some packages failed." -ForegroundColor Yellow
    Write-Host "  Fix: install Python 3.12 from python.org, delete the venv" -ForegroundColor Yellow
    Write-Host "  folder, then re-run this script." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  Dependencies installed." -ForegroundColor Green
}

# ── 5. Install PyTorch (auto-detect GPU vs CPU) ────────────────────────────────
Write-Host "[3b] Installing PyTorch..." -ForegroundColor Yellow

# Read DEVICE from .env if it already exists, otherwise use GPU detection
$EnvFilePath = Join-Path $ProjectDir ".env"
$DeviceSetting = "cpu"
if (Test-Path $EnvFilePath) {
    $envContent = Get-Content $EnvFilePath
    foreach ($line in $envContent) {
        if ($line -match "^DEVICE\s*=\s*(.+)$") {
            $DeviceSetting = $Matches[1].Trim()
        }
    }
}

# Auto-detect: if GPU found and DEVICE not explicitly set to cpu, use CUDA
if ($HasCuda -and $DeviceSetting -ne "cpu") {
    Write-Host "  Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Cyan
    & $PythonExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  PyTorch (CUDA) installed." -ForegroundColor Green
    } else {
        Write-Host "  CUDA install failed - falling back to CPU PyTorch..." -ForegroundColor Yellow
        & $PythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
} else {
    Write-Host "  Installing PyTorch (CPU)..." -ForegroundColor Cyan
    & $PythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    Write-Host "  PyTorch (CPU) installed." -ForegroundColor Green
}

# Verify torch sees the GPU
Write-Host "  Verifying PyTorch device..." -ForegroundColor Yellow
$torchCheck = & $PythonExe -c "import torch; cuda=torch.cuda.is_available(); print('CUDA available:', cuda); print('Device count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if cuda else 'N/A')" 2>&1
Write-Host "  $torchCheck" -ForegroundColor Cyan

# ── 6. Set up .env file ────────────────────────────────────────────────────────
Write-Host "[4/6] Setting up environment file..." -ForegroundColor Yellow

# Look for env.example OR .env.example (repo has both names across commits)
$EnvExample = $null
$candidates = @(
    (Join-Path $ProjectDir ".env.example"),
    (Join-Path $ProjectDir "env.example")
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $EnvExample = $c; break }
}

# Generate a template if neither exists
if (-Not $EnvExample) {
    Write-Host "  No env template found - generating one..." -ForegroundColor Yellow
    $EnvExample = Join-Path $ProjectDir ".env.example"

    # Detect correct DEVICE value for the generated template
    $templateDevice = if ($HasCuda) { "cuda" } else { "cpu" }

    # NOTE: here-string @" "@ closing delimiter MUST be at column 0 with no leading spaces
    $templateLines = @(
        "# Alpaca credentials",
        "# Paper trading: https://paper-api.alpaca.markets",
        "# Live trading:  https://api.alpaca.markets",
        "ALPACA_API_KEY=your_alpaca_api_key_here",
        "ALPACA_SECRET_KEY=your_alpaca_secret_key_here",
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
        "",
        "# FRED macro data - free key at fred.stlouisfed.org/docs/api/api_key.html",
        "FRED_API_KEY=your_fred_api_key_here",
        "",
        "# Safety gate - keep false until validated in paper trading",
        "LIVE_TRADING_ENABLED=false",
        "",
        "# Hardware: cuda (NVIDIA GPU), mps (Apple Silicon), or cpu",
        "DEVICE=$templateDevice",
        "",
        "# Model / training",
        "BATCH_SIZE=64",
        "EPOCHS=50",
        "LEARNING_RATE=0.0003",
        "TARGET_HORIZON=6",
        "",
        "# Risk parameters",
        "MAX_POSITION_RISK=0.01",
        "STOP_LOSS_PCT=0.005",
        "TAKE_PROFIT_PCT=0.015",
        "",
        "# Optional: override runtime directories (leave blank for defaults)",
        "DATA_DIR=",
        "CACHE_DIR=",
        "MODEL_DIR=",
        "LOG_DIR="
    )
    $templateLines | Set-Content -Path $EnvExample -Encoding UTF8
    Write-Host "  Template generated at: $EnvExample" -ForegroundColor Green
}

$EnvFile = Join-Path $ProjectDir ".env"
if (-Not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "  Created .env from template." -ForegroundColor Green
    Write-Host "  Edit .env and add your API keys before training!" -ForegroundColor Yellow

    # Auto-set DEVICE in .env based on GPU detection
    if ($HasCuda) {
        (Get-Content $EnvFile) -replace "^DEVICE=.*", "DEVICE=cuda" |
            Set-Content $EnvFile -Encoding UTF8
        Write-Host "  Auto-set DEVICE=cuda in .env (GPU detected)." -ForegroundColor Green
    }
} else {
    Write-Host "  .env already exists - leaving untouched." -ForegroundColor Green

    # Warn if DEVICE=cpu but a GPU is present
    $currentDevice = ""
    foreach ($line in (Get-Content $EnvFile)) {
        if ($line -match "^DEVICE\s*=\s*(.+)$") {
            $currentDevice = $Matches[1].Trim()
        }
    }
    if ($HasCuda -and $currentDevice -eq "cpu") {
        Write-Host ""
        Write-Host "  WARNING: GPU detected but DEVICE=cpu in your .env!" -ForegroundColor Red
        Write-Host "  Change it to DEVICE=cuda to use your GPU for training." -ForegroundColor Red
        Write-Host ""
    } elseif ($HasCuda -and $currentDevice -eq "cuda") {
        Write-Host "  DEVICE=cuda confirmed - GPU will be used for training." -ForegroundColor Green
    }
}

# ── 7. Create runtime directories ─────────────────────────────────────────────
Write-Host "[5/6] Creating runtime directories..." -ForegroundColor Yellow
foreach ($dir in @("data", "models", "cache", "logs", "backups")) {
    $path = Join-Path $ProjectDir $dir
    if (-Not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path | Out-Null
    }
}
Write-Host "  Directories ready." -ForegroundColor Green

# ── 8. Task Scheduler (optional, needs elevation) ─────────────────────────────
Write-Host "[6/6] Task Scheduler setup..." -ForegroundColor Yellow

# Check if running as Administrator - Task Scheduler requires elevation
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-Not $isAdmin) {
    Write-Host "  Skipped: Task Scheduler requires Administrator privileges." -ForegroundColor Yellow
    Write-Host "  To register, right-click PowerShell -> 'Run as administrator'" -ForegroundColor Yellow
    Write-Host "  and re-run this script, OR use the manual command below." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Manual one-liner (run as Admin):" -ForegroundColor White
    $fullPython = Join-Path $VenvDir "Scripts\python.exe"
    $fullScript = Join-Path $ProjectDir "scripts\run_live.py"
    Write-Host "    `$a=New-ScheduledTaskAction -Execute '$fullPython' -Argument '$fullScript'" -ForegroundColor Gray
    Write-Host "    `$t=New-ScheduledTaskTrigger -AtLogOn" -ForegroundColor Gray
    Write-Host "    Register-ScheduledTask -TaskName 'SPYQuantLive' -Action `$a -Trigger `$t -RunLevel Highest -Force" -ForegroundColor Gray
} else {
    $TaskName  = "SPYQuantLive"
    $existing  = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  Task '$TaskName' already registered." -ForegroundColor Green
    } else {
        try {
            $fullPython = Join-Path $VenvDir "Scripts\python.exe"
            $fullScript = Join-Path $ProjectDir "scripts\run_live.py"
            $action   = New-ScheduledTaskAction -Execute $fullPython -Argument $fullScript -WorkingDirectory $ProjectDir
            $trigger  = New-ScheduledTaskTrigger -AtLogOn
            $settings = New-ScheduledTaskSettingsSet `
                -RestartCount 5 `
                -RestartInterval (New-TimeSpan -Minutes 1) `
                -ExecutionTimeLimit (New-TimeSpan -Hours 0)
            Register-ScheduledTask -TaskName $TaskName -Action $action `
                -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
            Write-Host "  Task '$TaskName' registered." -ForegroundColor Green
        } catch {
            Write-Host "  Task Scheduler registration failed: $_" -ForegroundColor Yellow
        }
    }
}

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
if ($HasCuda) {
    Write-Host "  GPU ready - PyTorch CUDA installed. DEVICE=cuda in .env." -ForegroundColor Green
} else {
    Write-Host "  No GPU found - using CPU. Training will take 4-8 hours." -ForegroundColor Yellow
    Write-Host "  If you have an NVIDIA GPU, install CUDA drivers and re-run." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor White
Write-Host "  1. Edit .env - add ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY"
Write-Host "  2. Activate:  .\venv\Scripts\Activate.ps1"
Write-Host "  3. Train:     python scripts\train.py --alpaca"
Write-Host "  4. Evaluate:  python scripts\evaluate.py --alpaca"
Write-Host "  5. Paper:     python scripts\run_live.py"
Write-Host "  6. Dashboard: python -m dashboard.server"
Write-Host ""
Write-Host "  Verify GPU is working after activation:" -ForegroundColor White
Write-Host "  python -c `"import torch; print(torch.cuda.is_available())`""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
