"""
data/features.py
Compute all model-ready features from raw OHLCV + macro data.

Feature set (FEATURE_DIM = 14):
  0  log_return          — bar-to-bar log return
  1  realized_vol        — 20-bar rolling σ of log returns
  2  vol_of_vol          — 20-bar rolling σ of realized_vol
  3  momentum_20         — 20-bar price momentum (normalised)
  4  momentum_60         — 60-bar price momentum (normalised)
  5  vwap_dev            — deviation of close from 20-bar VWAP
  6  spread_proxy        — (high - low) / close  [original intrabar spread proxy]
  7  volume_z            — z-score of log volume (20-bar)
  8  ust10y              — daily 10-yr yield (raw, will be scaled)
  9  yield_change_5d     — 5-day change in UST10Y
 10  bar_of_day          — fraction through trading session [0, 1]
 11  ba_spread           — bid-ask spread as fraction of mid  (microstructure)
                           Falls back to high-low proxy when quote data absent.
 12  trade_imbalance     — (close - open) / (high - low + EPS)  in [-1, 1]
                           Positive = bar closed near high (buy pressure).
 13  overnight_gap       — log(open / prev_close)  — gap risk / momentum signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


_EPS = 1e-8
_TRADING_BARS_PER_DAY = 78   # 6.5 h × 12 bars/h


def _rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).std()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume, ust10y
        with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        One row per original bar, columns are the 11 engineered features.
        Rows with insufficient history (NaN) are dropped.
    """
    out = pd.DataFrame(index=df.index)

    # ── 0  Log return ─────────────────────────────────────────────────────────
    log_c = np.log(df["close"] + _EPS)
    out["log_return"] = log_c.diff()

    # ── 1  Realised volatility (20-bar) ───────────────────────────────────────
    out["realized_vol"] = _rolling_std(out["log_return"], 20)

    # ── 2  Vol-of-vol ─────────────────────────────────────────────────────────
    out["vol_of_vol"] = _rolling_std(out["realized_vol"], 20)

    # ── 3 / 4  Momentum ───────────────────────────────────────────────────────
    out["momentum_20"] = log_c - log_c.shift(20)
    out["momentum_60"] = log_c - log_c.shift(60)

    # ── 5  VWAP deviation (20-bar typical-price VWAP) ─────────────────────────
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]
    vwap = tp_vol.rolling(20).sum() / (df["volume"].rolling(20).sum() + _EPS)
    out["vwap_dev"] = (df["close"] - vwap) / (vwap + _EPS)

    # ── 6  Spread proxy ───────────────────────────────────────────────────────
    out["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + _EPS)

    # ── 7  Volume z-score ─────────────────────────────────────────────────────
    log_vol = np.log(df["volume"] + 1.0)
    vol_mean = log_vol.rolling(20).mean()
    vol_std  = _rolling_std(log_vol, 20)
    out["volume_z"] = (log_vol - vol_mean) / (vol_std + _EPS)

    # ── 8  UST10Y ─────────────────────────────────────────────────────────────
    out["ust10y"] = df["ust10y"]

    # ── 9  5-day yield change (≈ 390 bars @ 5-min) ───────────────────────────
    bars_5d = 5 * _TRADING_BARS_PER_DAY
    out["yield_change_5d"] = df["ust10y"].diff(bars_5d)

    # ── 10  Bar-of-day ────────────────────────────────────────────────────────
    # NYSE session: 09:30 – 16:00 ET   →  0 … 1
    try:
        et = df.index.tz_convert("America/New_York")
    except Exception:
        et = df.index
    open_minutes  = 9 * 60 + 30
    close_minutes = 16 * 60
    session_len   = close_minutes - open_minutes   # 390 min
    bar_minutes   = et.hour * 60 + et.minute
    out["bar_of_day"] = np.clip((bar_minutes - open_minutes) / session_len, 0, 1)

    # ── 11  Bid-ask spread (microstructure) ───────────────────────────────────
    # When Alpaca quote data is available (columns ask / bid or ask_price /
    # bid_price), compute true spread as fraction of mid.  Falls back to the
    # (high - low) / close proxy so the column is always present.
    if "ask" in df.columns and "bid" in df.columns:
        mid = (df["ask"] + df["bid"]) / 2.0 + _EPS
        out["ba_spread"] = (df["ask"] - df["bid"]) / mid
    elif "ask_price" in df.columns and "bid_price" in df.columns:
        mid = (df["ask_price"] + df["bid_price"]) / 2.0 + _EPS
        out["ba_spread"] = (df["ask_price"] - df["bid_price"]) / mid
    else:
        # Proxy: intrabar range / close — wider when spread is wide
        out["ba_spread"] = (df["high"] - df["low"]) / (df["close"] + _EPS)

    # ── 12  Trade imbalance ───────────────────────────────────────────────────
    # (close - open) / (high - low)  normalised to [-1, 1].
    # +1 means bar closed at its high (strong buy pressure).
    # -1 means bar closed at its low  (strong sell pressure).
    # 0  means indecision / balanced.
    bar_range = df["high"] - df["low"]
    out["trade_imbalance"] = (df["close"] - df["open"]) / (bar_range + _EPS)
    out["trade_imbalance"] = out["trade_imbalance"].clip(-1.0, 1.0)

    # ── 13  Overnight gap ─────────────────────────────────────────────────────
    # log(open_t / close_{t-1}) captures pre-market moves, earnings gaps,
    # and macro overnight events — information not in intraday OHLCV alone.
    # For intraday bars this is non-zero only on the first bar of each session.
    out["overnight_gap"] = np.log(df["open"] + _EPS) - log_c.shift(1)

    # ── Drop warm-up rows ─────────────────────────────────────────────────────
    out = out.dropna()
    logger.info(f"Features computed: {len(out):,} bars × {out.shape[1]} features")
    return out
