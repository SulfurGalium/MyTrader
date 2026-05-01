"""
trading/inference.py
Real-time inference pipeline.
  1. Pull latest N bars from Alpaca
  2. Run feature engineering + scaling
  3. Feed into loaded model
  4. Return signed signal [-1, 1]
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.loader import load_ohlcv_alpaca, build_raw_dataset
from data.features import compute_features
from data.preprocessing import load_scaler, apply_scaler, make_stationary
from models.diffusion import MultiTimeframeDiffusion, load_checkpoint
from data.dataset import SPYWindowDataset


_model: MultiTimeframeDiffusion | None = None
_device: torch.device | None = None


def get_model() -> tuple[MultiTimeframeDiffusion, torch.device]:
    global _model, _device
    if _model is None:
        _device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        _model  = MultiTimeframeDiffusion()
        ckpt    = load_checkpoint(_model)
        _model  = _model.to(_device)
        _model.eval()
        logger.info(f"Model loaded (epoch {ckpt.get('epoch')}) on {_device}")
    return _model, _device


def generate_signal(
    symbol:   str = "SPY",
    n_bars:   int = config.SEQ_LEN + 80,   # extra buffer for feature warmup
    n_samples: int = 32,                    # diffusion ensemble size
) -> float:
    """
    Pull live data, compute features, run model ensemble, return signal ∈ [-1, 1].
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    # Pull enough recent bars for seq_len + warmup features (need ~500 bars minimum)
    # Use 5 trading days back to ensure enough data regardless of holidays
    from datetime import datetime, timedelta
    start_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    raw = load_ohlcv_alpaca(symbol=symbol, start=start_date)
    raw = raw.tail(n_bars)
    enriched = build_raw_dataset(raw)
    feats    = compute_features(enriched)
    stat     = make_stationary(feats)

    scaler   = load_scaler()
    scaled   = apply_scaler(stat, scaler)

    if len(scaled) < config.SEQ_LEN + 1:
        logger.warning("Not enough bars for inference window.")
        return 0.0

    # ── Build a single window ─────────────────────────────────────────────────
    window   = scaled[-config.SEQ_LEN:]                   # (L, F)
    trim     = (config.SEQ_LEN // config.COARSE_FACTOR) * config.COARSE_FACTOR
    coarse_w = (
        window[:trim]
        .reshape(-1, config.COARSE_FACTOR, window.shape[1])
        .mean(axis=1)
    )

    model, device = get_model()
    fine_t   = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
    coarse_t = torch.from_numpy(coarse_w.astype(np.float32)).unsqueeze(0).to(device)

    # ── Ensemble: run diffusion sampling n_samples times ─────────────────────
    fine_rep   = fine_t.expand(n_samples, -1, -1)
    coarse_rep = coarse_t.expand(n_samples, -1, -1)

    with torch.no_grad():
        preds = model.sample(fine_rep, coarse_rep, steps=50)   # (n_samples,)

    pred_mean = preds.mean().item()
    pred_std  = preds.std().item()

    # Confidence-weighted signal: scale by |mean| / std ratio (snr)
    snr    = abs(pred_mean) / (pred_std + 1e-6)
    signal = float(np.tanh(pred_mean * snr))   # squash to (-1, 1)

    logger.info(
        f"Signal={signal:+.4f}  pred_mean={pred_mean:.5f}  "
        f"pred_std={pred_std:.5f}  SNR={snr:.2f}"
    )
    return signal
