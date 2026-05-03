"""
config.py - Centralised runtime configuration loaded from .env
Works on Windows, Linux, and macOS.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# -- Paths (cross-platform via pathlib) ---------------------------------------
# os.getenv returns "" for empty env vars - treat "" as "not set" and use default
def _resolve_dir(env_key: str, default: Path) -> Path:
    val = os.getenv(env_key, "").strip()
    return Path(val) if val else default

DATA_DIR   = _resolve_dir("DATA_DIR",  BASE_DIR / "data")
CACHE_DIR  = _resolve_dir("CACHE_DIR", BASE_DIR / "cache")
MODEL_DIR  = _resolve_dir("MODEL_DIR", BASE_DIR / "models")
LOG_DIR    = _resolve_dir("LOG_DIR",   BASE_DIR / "logs")

for d in (DATA_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -- API credentials -----------------------------------------------------------
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")

# -- Safety gate ---------------------------------------------------------------
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"

# -- Device (auto-detect if not set) -----------------------------------------
# Bug fix: os.getenv("DEVICE") returns "" when the env var is set but empty,
# and torch.device("") raises a RuntimeError.  Always fall back to "cuda" when
# the env var is absent or blank; PyTorch itself will fall back to CPU at
# runtime if no CUDA device is found.
_device_env = os.getenv("DEVICE", "").strip()
DEVICE = _device_env if _device_env else "cuda"

BATCH_SIZE     = int(os.getenv("BATCH_SIZE", 64))
EPOCHS         = int(os.getenv("EPOCHS", 50))
# Peak LR after warmup. 1e-4 is safe for this architecture with AMP.
# 3e-4 caused float16 overflow at epoch 28 (loss spike to 4421 -> weights->0).
# The warmup still ramps up to this peak smoothly over 10% of epochs.
LEARNING_RATE  = float(os.getenv("LEARNING_RATE", 1e-4))
WARMUP_RATIO   = float(os.getenv("WARMUP_RATIO", 0.10))

SEQ_LEN        = 60   # 5-min bars  -  5 hours of history
COARSE_FACTOR  = 6    # coarse = 30-min bars aggregated from 5-min
FEATURE_DIM    = 14   # 11 original + ba_spread + trade_imbalance + overnight_gap
DIFFUSION_STEPS = 1000

# Prediction horizon: how many bars ahead the model targets.
# 1 = original (next 5-min bar only - very noisy).
# 6 = 30-min forward return (sum of 6 bars) - ~3x better SNR, recommended.
TARGET_HORIZON = int(os.getenv("TARGET_HORIZON", 6))

# -- Risk parameters -----------------------------------------------------------
MAX_POSITION_RISK = float(os.getenv("MAX_POSITION_RISK", 0.01))
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT",     0.005))
TAKE_PROFIT_PCT   = float(os.getenv("TAKE_PROFIT_PCT",   0.015))

# -- Backtest / Monte Carlo ----------------------------------------------------
MC_SIMULATIONS  = 2000
WALK_FORWARD_SPLITS = 5
MIN_SHARPE_THRESHOLD = 0.5   # reject regimes below this
