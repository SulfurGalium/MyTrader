# SPY Quant — Model-Driven Intraday Trading System

A production-grade quantitative trading pipeline for SPY that takes raw 5-minute OHLCV data through feature engineering, diffusion model training, walk-forward backtesting, and live deployment via Alpaca — designed to run continuously on a Vultr VPS.

---

## What Changed (recent improvements)

| File | Change |
|---|---|
| `config.py` | `FEATURE_DIM` 11 → 14 · `TARGET_HORIZON=6` added · `DEVICE=""` crash fixed |
| `data/features.py` | 3 new microstructure features: `ba_spread`, `trade_imbalance`, `overnight_gap` |
| `data/dataset.py` | `horizon` param — default target is now 30-min forward return (6 bars) not 1-bar |
| `trading/inference.py` | Ensemble 32×50 → 64×20 DDIM · returns `(signal, vov)` tuple |
| `trading/live.py` | Position sizing 2× leverage bug fixed · regime gate added (skips choppy markets) |
| `backtest/simulation.py` | MC branches now enforced non-overlapping (min `SEQ_LEN` gap between starts) |
| `scripts/run_live.py` | `TradingSession` created once (not per-cycle) so regime history persists |
| `deployment/deploy.sh` | Double-nesting fix · auto-generates `.env.example` if missing |

> **Retraining required**: `FEATURE_DIM` changed from 11 → 14. Old checkpoints are incompatible — delete them before training.

---

## Architecture Overview

```
Raw OHLCV (5-min)
      │
      ▼
 data/loader.py          ←── UST10Y (FRED, cached 12h)
      │   build_raw_dataset()
      ▼
 data/features.py        — 14 engineered features (see table below)
      │   compute_features()
      ▼
 data/preprocessing.py   — stationarity + RobustScaler (train-only fit, no leakage)
      │   preprocess()
      ▼
 data/dataset.py         — lazy rolling windows · horizon=6 (30-min target)
      │   SPYWindowDataset
      ▼
 models/diffusion.py     — MultiTimeframeDiffusion
      │   Fine encoder (5-min) → Coarse encoder (30-min) → Cross-attention → DDPM head
      ▼
 models/trainer.py       — AdamW · CosineAnnealingLR · AMP · dual checkpoint
      │
      ▼
 backtest/simulation.py  — Regime detection · non-overlapping MC branches · WFO
      │
      ▼
 trading/inference.py    — 64-sample DDIM ensemble · SNR-weighted signal · returns vov
      │
      ▼
 trading/live.py         — Regime gate · fixed sizing · Alpaca bracket orders
```

---

## Feature Set (14 features)

| # | Name | Description |
|---|------|-------------|
| 0 | `log_return` | Bar-to-bar log return |
| 1 | `realized_vol` | 20-bar rolling σ of log returns |
| 2 | `vol_of_vol` | 20-bar rolling σ of realized_vol — also drives the regime gate |
| 3 | `momentum_20` | 20-bar price momentum |
| 4 | `momentum_60` | 60-bar price momentum |
| 5 | `vwap_dev` | Deviation from 20-bar VWAP |
| 6 | `spread_proxy` | (high−low) / close |
| 7 | `volume_z` | Z-score of log volume (20-bar) |
| 8 | `ust10y` | UST 10-yr yield (Δ-differenced, stationarised) |
| 9 | `yield_change_5d` | 5-day change in UST10Y |
| 10 | `bar_of_day` | Fraction through NYSE session [0, 1] |
| 11 | `ba_spread` | Bid-ask spread / mid (real quote when available, else high-low proxy) |
| 12 | `trade_imbalance` | (close−open) / (high−low) — buy/sell pressure proxy [-1, 1] |
| 13 | `overnight_gap` | log(open / prev_close) — pre-market gap signal |

---

## Model: MultiTimeframeDiffusion

- **Fine encoder** — 3-layer Transformer over 60 × 5-min bars
- **Coarse encoder** — 3-layer Transformer over 10 × 30-min aggregated bars
- **Fusion** — Cross-attention: fine queries attend to coarse keys/values
- **Diffusion head** — DDPM ε-prediction MLP with cosine noise schedule (1000 steps)
- **Inference** — DDIM fast sampling (20 steps), 64-sample ensemble for SNR-weighted signal
- **Target** — Sum of log returns over next 6 bars (30-min forward return)

---

## Training on Your Home PC

### Hardware requirements

| Setup | What to expect |
|---|---|
| **CPU only, 8 GB RAM** | Works. 4–8 hours for 50 epochs on 4 years of data. Use `--batch-size 32`. |
| **CPU only, 16 GB RAM** | Comfortable. 3–5 hours. Default batch size (64) is fine. |
| **NVIDIA GPU (6 GB+ VRAM)** | 30–60 min for 50 epochs. Set `DEVICE=cuda` in `.env`. |
| **Apple Silicon (M1/M2/M3)** | Use `DEVICE=mps`. 1–2 hours. Requires PyTorch ≥ 2.0. |
| **Minimum disk** | ~2 GB for data + checkpoints. |

The trainer auto-scales batch size to available RAM (16 if < 2 GB free, 32 if < 3 GB free, 64 otherwise). Override with `--batch-size`.

---

### Step 1 — Prerequisites

**Python 3.10, 3.11, or 3.12** required. Python 3.13+ not yet supported.

```bash
python --version   # or python3 --version on Linux/macOS
```

Download: https://www.python.org/downloads/

**Git:**
- Windows: https://git-scm.com/download/win
- macOS: `xcode-select --install`
- Linux: `sudo apt install git`

---

### Step 2 — Get the code

```bash
git clone <your-repo-url> mytrader
cd mytrader/spy_quant
```

> The repo root is `MyTrader/` — you must `cd` into `spy_quant/` before running anything.

---

### Step 3 — Set up the environment

**Windows (PowerShell):**
```powershell
# Allow scripts (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Creates venv, installs deps, copies .env
.\setup_windows.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---

### Step 4 — Install PyTorch

PyTorch isn't pinned in `requirements.txt` because the right wheel depends on your hardware. Install it separately:

```bash
# CPU only (works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU — CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# NVIDIA GPU — CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (M1/M2/M3) — MPS included in the standard wheel
pip install torch
```

Verify:
```bash
python -c "import torch; print(torch.__version__)"
# CUDA:  python -c "import torch; print(torch.cuda.is_available())"
# MPS:   python -c "import torch; print(torch.backends.mps.is_available())"
```

---

### Step 5 — Configure your .env

```bash
notepad .env      # Windows
open -e .env      # macOS
nano .env         # Linux
```

Minimum keys needed:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

FRED_API_KEY=your_fred_key_here

LIVE_TRADING_ENABLED=false

# "cuda" (NVIDIA), "mps" (Apple Silicon), or "cpu"
DEVICE=cpu

# 6 = 30-min forward return (recommended)
TARGET_HORIZON=6
```

Free API keys:
- **Alpaca** (data + paper trading): https://alpaca.markets → sign up → API Keys
- **FRED** (Treasury yield): https://fred.stlouisfed.org/docs/api/api_key.html

---

### Step 6 — Train

**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
python scripts\train.py --alpaca --start 2020-01-01
# or: run.bat train
```

**macOS / Linux:**
```bash
source venv/bin/activate
python scripts/train.py --alpaca --start 2020-01-01
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--alpaca` | — | Pull live data from Alpaca |
| `--data path/file.parquet` | — | Use a local Parquet file instead |
| `--start 2020-01-01` | `2020-01-01` | Data start date |
| `--end 2024-12-31` | today | Data end date |
| `--epochs 50` | 50 | Max epochs (early stopping may end sooner) |
| `--batch-size 32` | auto | Override auto-detected batch size |
| `--device cpu` | from `.env` | Force a specific device |
| `--patience 80` | 80 | Early stopping patience |
| `--no-trading-selection` | off | Checkpoint by val loss instead of sign accuracy |

**Healthy training output:**
```
Epoch  1/50 | train_loss=0.843 | val_loss=0.789 | sign_acc=0.521
Epoch  5/50 | train_loss=0.610 | val_loss=0.583 | sign_acc=0.537
Epoch 10/50 | train_loss=0.534 | val_loss=0.521 | sign_acc=0.549
...
✓ Best trading checkpoint → models/trading_model.pt  (sign_acc=0.554)
✓ Best loss checkpoint   → models/best_model.pt      (val_loss=0.510)
```

Sign accuracy above **0.53** is meaningful. Above **0.55** is strong. `trading_model.pt` is what inference uses.

---

### Step 7 — Backtest

```bash
python scripts/evaluate.py --alpaca     # macOS/Linux
python scripts\evaluate.py --alpaca     # Windows
```

Runs the walk-forward Monte Carlo on the held-out test set (last 15% of data, never seen during training). Aim for Sharpe > 0.5, max drawdown < 20% before going live.

---

### Step 8 — Paper trade locally

```bash
python scripts/run_live.py     # macOS/Linux
python scripts\run_live.py     # Windows
# or: run.bat live
```

Fires every 5 minutes during NYSE hours. With `LIVE_TRADING_ENABLED=false` (default), it logs what it *would* do without submitting orders. Check `logs/live_YYYY-MM-DD.log` to see signals and regime gate decisions.

Open the dashboard in a second terminal:
```bash
python -m dashboard.server
# visit http://localhost:8080
```

---

### Step 9 — Deploy to the server

Copy your trained checkpoint to the VPS instead of retraining there (much faster):

```bash
# From your home PC
scp models/trading_model.pt  root@<server-ip>:/opt/spy_quant/models/
scp models/best_model.pt     root@<server-ip>:/opt/spy_quant/models/
scp models/scaler.pkl        root@<server-ip>:/opt/spy_quant/models/
```

Then on the server:
```bash
sudo systemctl start spy_quant
sudo systemctl start spy_quant_dashboard
sudo journalctl -u spy_quant -f
```

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` | See Step 4 — install the right PyTorch wheel for your hardware manually |
| `ALPACA_API_KEY not set` | Make sure `.env` is saved and you're running from inside `spy_quant/` |
| Training very slow on CPU | Normal. Try `--start 2022-01-01` for less data, or run overnight |
| `CUDA out of memory` | Lower batch size: `--batch-size 16` or `--batch-size 8` |
| Checkpoint errors after code update | `FEATURE_DIM` changed 11→14. Delete old checkpoints and retrain |
| `torch.device("")` RuntimeError | Fixed in new `config.py`. Set `DEVICE=cpu` (not blank) in `.env` |

**Delete old incompatible checkpoints:**
```bash
# Windows
del models\best_model.pt models\trading_model.pt models\scaler.pkl

# macOS / Linux
rm models/best_model.pt models/trading_model.pt models/scaler.pkl
```

---

## Deployment on Vultr

### Option A: Bare-metal (systemd)

```bash
# 1. Clone — repo root is MyTrader/, must cd into spy_quant/ first
git clone <your-repo-url> /tmp/mytrader
cd /tmp/mytrader/spy_quant

# 2. Deploy (installs to /opt/spy_quant, creates quant user, registers services)
sudo bash deployment/deploy.sh

# 3. Edit .env
sudo nano /opt/spy_quant/.env

# 4a. Train on server (30–90 min, CPU)
sudo -u quant /opt/spy_quant/venv/bin/python scripts/train.py --alpaca

# 4b. Or SCP checkpoint from home PC (recommended)
#     scp models/trading_model.pt root@<ip>:/opt/spy_quant/models/
#     scp models/scaler.pkl       root@<ip>:/opt/spy_quant/models/

# 5. Start (sudo required for all systemctl commands)
sudo systemctl start spy_quant
sudo systemctl start spy_quant_dashboard

# 6. Monitor
sudo systemctl status spy_quant
sudo journalctl -u spy_quant -f
```

> **Why `sudo` is always required for `systemctl`**: communicates with PID 1 via D-Bus — a root operation. Running without `sudo` gives *"Failed to connect to bus"*. This is a Linux security boundary, not a bug.

### Option B: Docker Compose

```bash
cp .env.example .env && nano .env
cd deployment
docker compose up -d
docker logs -f spy_quant
```

### Recommended Vultr instance

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 40 GB SSD | 80 GB SSD |
| GPU | — | NVIDIA A16 (server-side training only) |

Train at home on GPU, `scp` the checkpoint to a cheap CPU-only VPS ($6–12/mo) for inference.

---

## Safety System

Five independent safety layers in `trading/live.py`:

1. **`LIVE_TRADING_ENABLED` flag** — must be explicitly `"true"`. Default is dry-run.
2. **Market-hours check** — Alpaca clock API; no orders outside NYSE session.
3. **Regime gate** — `vol_of_vol` tracked in a rolling 500-bar window. If current reading exceeds the 80th percentile → `reason="regime_choppy"`, cycle skipped. Automatically sits out choppy markets.
4. **Signal threshold** — `|signal| < 0.15` dropped.
5. **Cooldown** — minimum 5 minutes between signals.

All orders are **bracket orders** (entry + stop-loss + take-profit) — every position has defined maximum loss before submission.

**Position sizing**: `shares = floor(equity × MAX_POSITION_RISK / (price × STOP_LOSS_PCT))`. At defaults (risk=1%, SL=0.5%), a stop hit costs exactly 1% of equity. No leverage, no signal-strength amplification.

---

## Project Structure

```
spy_quant/
├── config.py                  ← central config (reads .env)
├── requirements.txt
├── .env.example               ← copy to .env, fill in keys
├── setup_windows.ps1          ← Windows one-shot setup
├── run.bat                    ← Windows launcher (train/live/dashboard)
│
├── data/
│   ├── loader.py              ← OHLCV + FRED loader (12h cache)
│   ├── features.py            ← 14 engineered features
│   ├── preprocessing.py       ← stationarity + leak-free RobustScaler
│   └── dataset.py             ← lazy rolling-window Dataset (horizon param)
│
├── models/
│   ├── diffusion.py           ← MultiTimeframeDiffusion architecture
│   └── trainer.py             ← AdamW · CosineAnnealingLR · AMP · dual checkpoint
│
├── backtest/
│   └── simulation.py          ← regime detection · non-overlapping MC · WFO
│
├── trading/
│   ├── inference.py           ← 64×20 DDIM ensemble · returns (signal, vov)
│   └── live.py                ← regime gate · fixed sizing · bracket orders
│
├── scripts/
│   ├── train.py               ← training entry point
│   ├── evaluate.py            ← backtest entry point
│   ├── run_live.py            ← live loop (persistent session, regime history)
│   └── monitor.py             ← terminal health view
│
├── deployment/
│   ├── deploy.sh              ← Vultr setup (fixed)
│   ├── spy_quant.service      ← systemd trading loop
│   ├── spy_quant_dashboard.service
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── models/                    ← trading_model.pt · best_model.pt · scaler.pkl
├── cache/                     ← FRED cache
└── logs/                      ← daily trading logs
```

---

## Risk Disclaimer

This system is for research and educational purposes. Quantitative models carry real financial risk. Always validate thoroughly in paper trading before enabling live execution. Past simulated performance does not guarantee future results. Never trade with money you cannot afford to lose.
