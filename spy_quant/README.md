# SPY Quant — Model-Driven Intraday Trading System

A production-grade quantitative trading pipeline for SPY that takes raw 5-minute OHLCV data through feature engineering, diffusion model training, walk-forward backtesting, and live deployment via Alpaca — designed to run continuously on a Vultr VPS.

---

## Architecture Overview

```
Raw OHLCV (5-min)
      │
      ▼
 data/loader.py          ←── UST10Y (FRED, cached)
      │   build_raw_dataset()
      ▼
 data/features.py        — 11 engineered features
      │   compute_features()
      ▼
 data/preprocessing.py   — stationarity + RobustScaler (train-only fit)
      │   preprocess()
      ▼
 data/dataset.py         — lazy rolling windows (fine 5-min + coarse 30-min)
      │   SPYWindowDataset
      ▼
 models/diffusion.py     — MultiTimeframeDiffusion
      │   Fine encoder → Coarse encoder → Cross-attention Fusion → DDPM head
      ▼
 models/trainer.py       — CUDA + AMP + OneCycleLR + early stopping
      │
      ▼
 backtest/simulation.py  — Regime detection → WalkForward → Monte Carlo
      │
      ▼
 trading/inference.py    — Live feature pipeline + ensemble sampling
      │
      ▼
 trading/live.py         — Alpaca bracket orders + safety gates
```

---

## Feature Set (11 features)

| # | Name | Description |
|---|------|-------------|
| 0 | `log_return` | Bar-to-bar log return |
| 1 | `realized_vol` | 20-bar rolling σ of log returns |
| 2 | `vol_of_vol` | 20-bar rolling σ of realized_vol |
| 3 | `momentum_20` | 20-bar price momentum |
| 4 | `momentum_60` | 60-bar price momentum |
| 5 | `vwap_dev` | Deviation from 20-bar VWAP |
| 6 | `spread_proxy` | (high−low) / close |
| 7 | `volume_z` | Z-score of log volume |
| 8 | `ust10y` | Δ UST 10-yr yield (stationarised) |
| 9 | `yield_change_5d` | 5-day change in UST10Y |
| 10 | `bar_of_day` | Fraction through NYSE session |

---

## Model: MultiTimeframeDiffusion

- **Fine encoder** — 3-layer Transformer over 60 × 5-min bars
- **Coarse encoder** — 3-layer Transformer over 10 × 30-min bars (aggregated)
- **Fusion** — Cross-attention: fine queries attend to coarse keys/values
- **Diffusion head** — DDPM ε-prediction MLP with cosine noise schedule
- **Inference** — DDIM fast sampling (50 steps), 32-sample ensemble for SNR-weighted signal

---

## Quick Start

### Windows

```powershell
# 1. Open PowerShell in the project folder, allow scripts, run setup
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows.ps1

# 2. Edit .env with your API keys (Notepad, VS Code, etc.)
notepad .env

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Train the model
python scripts\train.py --alpaca --start 2020-01-01

# 5. Evaluate (backtest)
python scripts\evaluate.py --alpaca

# 6. Paper trade
python scripts\run_live.py

# 7. Open dashboard in browser
python -m dashboard.server
# then visit http://localhost:8080
```

Or use the one-click launcher for everything:
```
run.bat train      # train
run.bat live       # paper trading loop
run.bat dashboard  # web dashboard
run.bat monitor    # terminal status
```

> **GPU on Windows:** Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and the matching `torch` wheel, then set `DEVICE=cuda` in `.env`. Training will be ~10–20× faster. CPU works fine for inference.

### Linux / macOS

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env
python scripts/train.py --alpaca --start 2020-01-01
python scripts/run_live.py
```

---

## Deployment on Vultr

### Option A: Bare-metal (systemd)

```bash
# On a fresh Ubuntu 22.04 VPS:
git clone <your-repo> /tmp/spy_quant
cd /tmp/spy_quant
sudo bash deployment/deploy.sh

# Edit .env with real credentials
sudo nano /opt/spy_quant/.env

# Train the model on the server
sudo -u quant /opt/spy_quant/venv/bin/python scripts/train.py --alpaca

# Start the service
sudo systemctl start spy_quant
sudo journalctl -u spy_quant -f
```

### Option B: Docker Compose

```bash
cp .env.example .env && nano .env
cd deployment
docker compose up -d

# Logs
docker logs -f spy_quant
```

### Recommended Vultr instance

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 40 GB SSD | 80 GB SSD |
| GPU | — | NVIDIA A16 (optional, for training) |

Train on a GPU instance, then downsize to a CPU-only instance for inference — the live signal generation is lightweight.

---

## Safety System

The live trading module has **four independent safety layers**:

1. **`LIVE_TRADING_ENABLED` env flag** — must be explicitly `"true"` to submit real orders. Default is dry-run.
2. **Market-hours check** — Alpaca clock API; no orders outside NYSE session.
3. **Signal threshold** — signals with `|z-score| < 0.5` are silently dropped.
4. **Cooldown timer** — minimum one 5-min bar between successive signals.

All orders are **bracket orders** (market entry + stop-loss + take-profit), ensuring every position has defined risk before submission.

---

## Project Structure

```
spy_quant/
├── config.py                  ← central config (reads .env, cross-platform)
├── requirements.txt
├── .env.example
├── setup_windows.ps1          ← Windows: one-shot setup + Task Scheduler
├── run.bat                    ← Windows: quick launcher (train/live/dashboard…)
│
├── data/
│   ├── loader.py              ← OHLCV + FRED macro loader (with cache)
│   ├── features.py            ← 11 engineered features
│   ├── preprocessing.py       ← stationarity + leak-free scaling
│   └── dataset.py             ← lazy rolling-window Dataset
│
├── models/
│   ├── diffusion.py           ← MultiTimeframeDiffusion architecture
│   └── trainer.py             ← training loop (CUDA + AMP + early stopping)
│
├── backtest/
│   └── simulation.py          ← regime detection, walk-forward, Monte Carlo
│
├── trading/
│   ├── inference.py           ← live feature pipeline + model inference
│   └── live.py                ← Alpaca client + bracket order execution
│
├── scripts/
│   ├── train.py               ← end-to-end training entry point
│   ├── evaluate.py            ← backtesting entry point
│   ├── run_live.py            ← live trading scheduler (runs on server)
│   └── monitor.py             ← system health dashboard
│
├── deployment/
│   ├── deploy.sh              ← one-shot Vultr server setup
│   ├── spy_quant.service      ← systemd unit file
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                      ← (runtime) local OHLCV parquet files
├── models/                    ← (runtime) saved checkpoints + scaler
├── cache/                     ← (runtime) cached FRED data
└── logs/                      ← (runtime) daily trading logs
```

---

## Risk Disclaimer

This system is for research and educational purposes. Quantitative models carry real financial risk. Always validate thoroughly in paper trading before enabling live execution. Past simulated performance does not guarantee future results.
