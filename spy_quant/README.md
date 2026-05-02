# SPY Quant — Model-Driven Intraday Trading System

A quantitative trading pipeline for SPY: raw 5-minute OHLCV → feature engineering → diffusion model training → walk-forward backtesting → live execution via Alpaca. Designed to run continuously on a Vultr VPS.

---

## Recent Changes

| File | Change |
|---|---|
| `config.py` | `FEATURE_DIM` 11 → 14 · `TARGET_HORIZON=6` · `DEVICE=""` crash fixed |
| `data/features.py` | 3 new features: `ba_spread`, `trade_imbalance`, `overnight_gap` |
| `data/dataset.py` | `horizon` param — target is now 30-min forward return (6 bars) |
| `trading/inference.py` | Ensemble 32×50 → 64×20 DDIM · returns `(signal, vov)` tuple |
| `trading/live.py` | Position sizing 2× leverage bug fixed · regime gate added |
| `backtest/simulation.py` | Non-overlapping MC branches (min `SEQ_LEN` gap enforced) |
| `scripts/run_live.py` | `TradingSession` created once so regime history persists |
| `deployment/deploy.sh` | Double-nesting fix · auto-generates env template if missing |
| `gpu_utils.py` | Clear diagnostics when `DEVICE=cuda` but CPU wheel is installed |
| `setup_windows.ps1` | Auto-detects GPU · installs correct PyTorch wheel · fixes parse errors |

> **Retraining required after this update**: `FEATURE_DIM` changed 11 → 14. Delete old checkpoints before running `train.py`.

---

## Architecture

```
Raw OHLCV (5-min)
      │
      ▼
 data/loader.py          ←── UST10Y (FRED, cached 12h)
      │   build_raw_dataset()
      ▼
 data/features.py        — 14 engineered features
      │   compute_features()
      ▼
 data/preprocessing.py   — stationarity · RobustScaler (fit on train only)
      │   preprocess()  saves → models/feature_scaler.pkl
      ▼
 data/dataset.py         — lazy rolling windows · horizon=6 (30-min target)
      │   SPYWindowDataset
      ▼
 models/diffusion.py     — MultiTimeframeDiffusion
      │   Fine encoder (5-min) + Coarse encoder (30-min) + Cross-attention + DDPM head
      ▼
 models/trainer.py       — AdamW · CosineAnnealingLR · AMP · dual checkpoint
      │   saves → models/diffusion_latest.pt
      │            models/diffusion_best_trading.pt
      ▼
 backtest/simulation.py  — Regime detection · non-overlapping MC branches · WFO
      ▼
 trading/inference.py    — 64-sample × 20-step DDIM ensemble · SNR signal · returns vov
      ▼
 trading/live.py         — Regime gate · fixed sizing · Alpaca bracket orders
```

---

## Feature Set (14 features)

| # | Name | Description |
|---|------|-------------|
| 0 | `log_return` | Bar-to-bar log return |
| 1 | `realized_vol` | 20-bar rolling σ of log returns |
| 2 | `vol_of_vol` | 20-bar rolling σ of realized_vol — drives the regime gate |
| 3 | `momentum_20` | 20-bar price momentum |
| 4 | `momentum_60` | 60-bar price momentum |
| 5 | `vwap_dev` | Deviation from 20-bar VWAP |
| 6 | `spread_proxy` | (high−low) / close |
| 7 | `volume_z` | Z-score of log volume (20-bar) |
| 8 | `ust10y` | UST 10-yr yield (Δ-differenced) |
| 9 | `yield_change_5d` | 5-day change in UST10Y |
| 10 | `bar_of_day` | Fraction through NYSE session [0, 1] |
| 11 | `ba_spread` | Bid-ask spread / mid (quote data if available, else high-low proxy) |
| 12 | `trade_imbalance` | (close−open) / (high−low) — buy/sell pressure [-1, 1] |
| 13 | `overnight_gap` | log(open / prev_close) — pre-market gap signal |

---

## Model

- **Fine encoder** — 3-layer Transformer over 60 × 5-min bars
- **Coarse encoder** — 3-layer Transformer over 10 × 30-min bars
- **Fusion** — Cross-attention (fine queries, coarse keys/values)
- **Head** — DDPM ε-prediction MLP, cosine noise schedule (1000 steps)
- **Inference** — DDIM (20 steps), 64-sample ensemble, SNR-weighted signal
- **Target** — Sum of log returns over next 6 bars (30-min forward return)

---

## Training on Your Home PC

### Hardware

| Setup | Expected time |
|---|---|
| CPU only, 8 GB RAM | 4–8 hours (50 epochs, 4 years data). Use `--batch-size 32`. |
| CPU only, 16 GB RAM | 3–5 hours. Default batch size (64) is fine. |
| NVIDIA GPU, 6 GB+ VRAM | 30–60 minutes. Set `DEVICE=cuda` in `.env`. |
| Apple Silicon (M1/M2/M3) | 1–2 hours. Set `DEVICE=mps`. Requires PyTorch ≥ 2.0. |

The trainer auto-scales batch size to available RAM. Override with `--batch-size`.

---

### Step 1 — Prerequisites

**Python 3.10, 3.11, or 3.12.** Python 3.13/3.14 works but some packages have limited wheel availability — if you hit build errors, install 3.12 instead.

```bash
python --version
```

Download: https://www.python.org/downloads/

**Git:**
- Windows: https://git-scm.com/download/win
- macOS: `xcode-select --install`
- Linux: `sudo apt install git`

---

### Step 2 — Clone the repo

```bash
git clone <your-repo-url> mytrader
cd mytrader/spy_quant
```

> The repo root is `MyTrader/`. You must `cd` into `spy_quant/` — everything runs from there.

---

### Step 3 — Set up the environment

#### Windows

```powershell
# One-time: allow PowerShell scripts to run
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run setup — creates venv, installs all deps including PyTorch, copies .env
.\setup_windows.ps1
```

`setup_windows.ps1` automatically:
- Detects your GPU via `nvidia-smi`
- Installs the CUDA 12.1 PyTorch wheel if a GPU is found, CPU wheel otherwise
- Generates `.env` from template
- Warns you if `DEVICE=cpu` is set but a GPU is present

**After setup, verify GPU is working:**
```powershell
.\venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
# Should print: True  and a version ending in +cu121
```

If you see `False` and the version ends in `+cpu`, the wrong wheel was installed. Fix:
```powershell
.\venv\Scripts\pip.exe uninstall torch torchvision torchaudio -y
.\venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check your driver's supported CUDA version first with `nvidia-smi` (top-right corner). Use `cu121` for CUDA 12.x drivers, `cu118` for CUDA 11.8.

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp env.example .env      # note: file is named env.example (no dot prefix)
```

PyTorch is included in `requirements.txt` and will install automatically. The default wheel is CUDA 12.1 compatible. On macOS, pip will resolve the correct wheel for your platform.

---

### Step 4 — Configure .env

```powershell
notepad .env        # Windows
```
```bash
nano .env           # Linux
open -e .env        # macOS
```

Required keys:

```env
# Alpaca — paper trading is safe default
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# FRED — free key for Treasury yield data
FRED_API_KEY=your_fred_key_here

# Safety gate — keep false during testing
LIVE_TRADING_ENABLED=false

# Device: cuda (NVIDIA), mps (Apple Silicon), cpu
DEVICE=cuda

# Prediction horizon (6 = 30-min forward return, recommended)
TARGET_HORIZON=6
```

Free API keys:
- **Alpaca**: https://alpaca.markets → sign up → Paper Trading → API Keys
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html

---

### Step 5 — Train

**Windows** (activate venv first):
```powershell
.\venv\Scripts\Activate.ps1
python scripts\train.py --alpaca --start 2020-01-01
```

Or use the launcher (no activation needed):
```powershell
run.bat train
```

**macOS / Linux:**
```bash
source venv/bin/activate
python scripts/train.py --alpaca --start 2020-01-01
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--alpaca` | — | Pull data live from Alpaca API |
| `--data path/file.parquet` | — | Use a local Parquet file instead |
| `--start 2020-01-01` | `2020-01-01` | Start date |
| `--end 2024-12-31` | today | End date |
| `--epochs 50` | from `.env` | Max epochs |
| `--batch-size 32` | auto | Override auto-detected batch size |
| `--device cuda` | from `.env` | Override device |
| `--patience 80` | 80 | Early stopping patience (epochs without improvement) |
| `--no-trading-selection` | off | Save checkpoint by val loss instead of sign accuracy |

**Healthy output:**
```
Epoch  1/50 | train_loss=0.843 | val_loss=0.789 | sign_acc=0.521
Epoch  5/50 | train_loss=0.610 | val_loss=0.583 | sign_acc=0.537
Epoch 10/50 | train_loss=0.534 | val_loss=0.521 | sign_acc=0.549
...
✓ Best trading checkpoint → models/diffusion_best_trading.pt  (sign_acc=0.554)
✓ Latest checkpoint      → models/diffusion_latest.pt
```

Sign accuracy above **0.53** is meaningful signal. Above **0.55** is strong.

**Files saved after training:**

| File | What it is |
|---|---|
| `models/diffusion_best_trading.pt` | Best directional accuracy checkpoint — used by inference |
| `models/diffusion_latest.pt` | Most recent epoch checkpoint |
| `models/feature_scaler.pkl` | Fitted RobustScaler — must be deployed alongside the model |

---

### Step 6 — Backtest

```powershell
# Windows
run.bat evaluate
# or: python scripts\evaluate.py --alpaca
```

```bash
# macOS / Linux
python scripts/evaluate.py --alpaca
```

Runs walk-forward Monte Carlo on the held-out test set (last 15% of data — never seen during training). Aim for Sharpe > 0.5 and max drawdown < 20% before going live.

---

### Step 7 — Paper trade locally

```powershell
# Windows
run.bat live
# or: .\venv\Scripts\Activate.ps1 then python scripts\run_live.py
```

```bash
# macOS / Linux
python scripts/run_live.py
```

Fires every 5 minutes during NYSE hours. With `LIVE_TRADING_ENABLED=false` (default), logs what it *would* do without submitting orders. Check `logs/live_YYYY-MM-DD.log` for signals and regime gate decisions.

Open the dashboard in a second terminal:
```powershell
run.bat dashboard        # Windows — also opens browser automatically
```
```bash
python -m dashboard.server   # macOS / Linux
# visit http://localhost:8080
```

Other launcher commands:
```powershell
run.bat monitor     # terminal status view
run.bat optimize    # walk-forward optimization
run.bat report      # generate HTML session report
run.bat shell       # open an activated venv shell
```

---

### Step 8 — Copy checkpoint to server

Train locally on your GPU, then copy the checkpoint to the VPS. This is much faster than retraining on a CPU-only server.

```bash
# From your home PC — replace <server-ip>
scp models/diffusion_best_trading.pt  root@<server-ip>:/opt/spy_quant/models/
scp models/diffusion_latest.pt        root@<server-ip>:/opt/spy_quant/models/
scp models/feature_scaler.pkl         root@<server-ip>:/opt/spy_quant/models/
```

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` returns `False` | CPU wheel installed. Run: `pip uninstall torch torchvision torchaudio -y` then reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| Version string shows `+cpu` not `+cu121` | Same as above — wrong wheel |
| `ALPACA_API_KEY not set` | Check `.env` is saved and you're running from `spy_quant/` |
| Training very slow | Normal on CPU. Use `--start 2022-01-01` for less data, or run overnight |
| `CUDA out of memory` | Lower batch size: `--batch-size 16` or `--batch-size 8` |
| Checkpoint load error after updating code | `FEATURE_DIM` changed 11→14 — old checkpoints are incompatible, delete and retrain |
| `Parse error` in `setup_windows.ps1` | Pull latest version — heredoc unicode issue is fixed |
| `Access denied` on Task Scheduler | Script must run as Administrator for that step — it now skips gracefully and prints the manual command |

**Delete incompatible old checkpoints:**
```powershell
# Windows
del models\diffusion_best_trading.pt models\diffusion_latest.pt models\feature_scaler.pkl
```
```bash
# macOS / Linux
rm models/diffusion_best_trading.pt models/diffusion_latest.pt models/feature_scaler.pkl
```

---

## Deployment on Vultr

### Option A: Bare-metal (systemd)

```bash
# 1. Clone — cd into spy_quant/ before running deploy.sh
git clone <your-repo-url> /tmp/mytrader
cd /tmp/mytrader/spy_quant

# 2. Deploy as root — installs to /opt/spy_quant, creates quant user,
#    registers both systemd services, generates .env from template
sudo bash deployment/deploy.sh

# 3. Add your API keys
sudo nano /opt/spy_quant/.env

# 4. Copy trained checkpoint from your home PC (recommended)
#    Run these from your home PC:
#    scp models/diffusion_best_trading.pt root@<ip>:/opt/spy_quant/models/
#    scp models/feature_scaler.pkl        root@<ip>:/opt/spy_quant/models/
#
#    Or train on the server directly (30-90 min on CPU):
#    sudo -u quant /opt/spy_quant/venv/bin/python scripts/train.py --alpaca

# 5. Start services — sudo required for all systemctl commands
sudo systemctl start spy_quant
sudo systemctl start spy_quant_dashboard

# 6. Monitor
sudo systemctl status spy_quant
sudo journalctl -u spy_quant -f
```

> `systemctl` always needs `sudo` — it communicates with PID 1 via D-Bus. Running without `sudo` gives *"Failed to connect to bus"*. This is a Linux security requirement, not a bug.

### Option B: Docker Compose

```bash
# From inside spy_quant/
cp env.example .env
nano .env          # add your API keys
cd deployment
docker compose up -d
docker logs -f spy_quant_trading
```

### Recommended Vultr instance

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 40 GB SSD | 80 GB SSD |

Train locally on your GPU and `scp` the checkpoint. Inference is lightweight — a $6/mo CPU instance handles it fine.

---

## Safety System

Five layers in `trading/live.py`, all must pass before any order is submitted:

1. **`LIVE_TRADING_ENABLED=true`** — must be explicitly set. Default is dry-run.
2. **Market hours** — Alpaca clock API checked; no orders outside NYSE session.
3. **Regime gate** — `vol_of_vol` (feature 2) tracked in a 500-bar rolling window. If above the 80th percentile, cycle is skipped (`reason="regime_choppy"`).
4. **Signal threshold** — `|signal| < 0.15` is dropped.
5. **Cooldown** — minimum 5 minutes between signals.

All orders are **bracket orders** with stop-loss and take-profit set before submission.

**Position sizing**: `shares = floor(equity × MAX_POSITION_RISK / (price × STOP_LOSS_PCT))`. At defaults (1% risk, 0.5% stop), a stop hit costs exactly 1% of equity.

---

## Project Structure

```
spy_quant/
├── config.py                    ← all config, reads .env
├── gpu_utils.py                 ← CUDA setup with clear diagnostics
├── requirements.txt             ← includes torch>=2.3.0 (CUDA 12.1)
├── env.example                  ← copy to .env and fill in keys
├── setup_windows.ps1            ← Windows: auto-detects GPU, installs everything
├── run.bat                      ← Windows launcher (train/evaluate/live/dashboard/etc)
│
├── data/
│   ├── loader.py                ← OHLCV + FRED loader (12h cache)
│   ├── features.py              ← 14 features
│   ├── preprocessing.py         ← stationarity · RobustScaler · saves feature_scaler.pkl
│   └── dataset.py               ← lazy SPYWindowDataset (horizon param)
│
├── models/
│   ├── diffusion.py             ← MultiTimeframeDiffusion · save/load checkpoint
│   └── trainer.py               ← training loop · dual checkpoint · AMP
│
├── backtest/
│   └── simulation.py            ← regime detection · non-overlapping MC · WFO
│
├── trading/
│   ├── inference.py             ← 64×20 DDIM ensemble · returns (signal, vov)
│   └── live.py                  ← regime gate · fixed sizing · bracket orders
│
├── scripts/
│   ├── train.py                 ← training entry point
│   ├── evaluate.py              ← backtest entry point
│   ├── optimize.py              ← walk-forward optimization
│   ├── run_live.py              ← live loop (persistent TradingSession)
│   ├── monitor.py               ← terminal status view
│   └── report.py                ← HTML session report
│
├── dashboard/
│   └── server.py                ← FastAPI dashboard (http://localhost:8080)
│
├── deployment/
│   ├── deploy.sh                ← Vultr bare-metal setup
│   ├── spy_quant.service        ← systemd unit (trading loop)
│   ├── spy_quant_dashboard.service
│   ├── Dockerfile               ← python:3.11-slim, CPU torch
│   └── docker-compose.yml
│
├── models/                      ← runtime: diffusion_best_trading.pt · diffusion_latest.pt · feature_scaler.pkl
├── cache/                       ← runtime: FRED data cache
└── logs/                        ← runtime: daily trading logs
```

---

## Risk Disclaimer

This system is for research and educational purposes. Quantitative models carry real financial risk. Always validate thoroughly in paper trading before enabling live execution. Past simulated performance does not guarantee future results. Never trade with money you cannot afford to lose.
