"""
data/preprocessing.py
Stationary transforms + leak-free train/val/test split + scaling.

The scaler is fit ONLY on the training fold and saved to disk so
inference uses identical parameters with no leakage.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


SCALER_PATH = config.MODEL_DIR / "feature_scaler.pkl"


# ── Stationarity helpers ──────────────────────────────────────────────────────

_ALREADY_STATIONARY = {
    "log_return", "realized_vol", "vol_of_vol",
    "momentum_20", "momentum_60", "vwap_dev",
    "spread_proxy", "volume_z", "yield_change_5d", "bar_of_day",
}

_LEVEL_FEATURES = {"ust10y"}   # take first difference to stationarise


def make_stationary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _LEVEL_FEATURES:
        if col in out.columns:
            out[col] = out[col].diff()
    out = out.dropna()
    return out


# ── Train / val / test split ──────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — NO shuffling, NO leakage."""
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]
    logger.info(
        f"Split → train {len(train):,} | val {len(val):,} | test {len(test):,}"
    )
    return train, val, test


# ── Scaling ───────────────────────────────────────────────────────────────────

def fit_and_save_scaler(train: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(train.values)
    joblib.dump(scaler, SCALER_PATH)
    logger.success(f"Scaler saved → {SCALER_PATH}")
    return scaler


def load_scaler() -> RobustScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run preprocessing first."
        )
    return joblib.load(SCALER_PATH)


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(df.values).astype(np.float32)


# ── High-level pipeline ───────────────────────────────────────────────────────

def preprocess(
    features: pd.DataFrame,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    train_arr, val_arr, test_arr : np.ndarray  (T × F)
    feature_names                : list[str]
    """
    stat = make_stationary(features)
    train_df, val_df, test_df = time_split(stat)

    if fit_scaler:
        scaler = fit_and_save_scaler(train_df)
    else:
        scaler = load_scaler()

    return (
        apply_scaler(train_df, scaler),
        apply_scaler(val_df,   scaler),
        apply_scaler(test_df,  scaler),
        list(stat.columns),
    )
