"""
data/loader.py
Load 5-min SPY OHLCV from a local Parquet file (or Alpaca historical API)
and enrich it with UST10Y from FRED.  FRED data is cached so subsequent
runs are instant.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# FRED / macro helpers
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = config.CACHE_DIR / "ust10y.parquet"
CACHE_META  = config.CACHE_DIR / "ust10y_meta.json"
FRED_SERIES = "DGS10"
CACHE_TTL_HOURS = 12          # refresh at most twice a day


def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists() or not CACHE_META.exists():
        return False
    meta = json.loads(CACHE_META.read_text())
    age = time.time() - meta.get("fetched_at", 0)
    return age < CACHE_TTL_HOURS * 3600


def fetch_ust10y(start: str = "2010-01-01") -> pd.Series:
    """Return daily UST 10-year yield as a pandas Series indexed by date.
    Hits the local cache first; falls back to FRED API when stale."""
    if _cache_is_fresh():
        logger.debug("UST10Y: loading from cache")
        df = pd.read_parquet(CACHE_FILE)
        return df["yield"]

    logger.info("UST10Y: fetching from FRED …")
    try:
        from fredapi import Fred
        fred = Fred(api_key=config.FRED_API_KEY)
        s = fred.get_series(FRED_SERIES, observation_start=start)
        s.name = "yield"
        df = s.to_frame()
        df.index.name = "date"
        df.to_parquet(CACHE_FILE)
        CACHE_META.write_text(json.dumps({"fetched_at": time.time()}))
        logger.success(f"UST10Y: cached {len(df)} observations")
        return df["yield"]
    except Exception as exc:
        logger.warning(f"FRED fetch failed ({exc}). Returning zeros.")
        return pd.Series(dtype=float, name="yield")


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file with columns: open, high, low, close, volume
    and a DatetimeIndex (UTC or tz-aware).  Validates schema."""
    df = pd.read_parquet(path)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"OHLCV file missing columns: {missing}")
    df.columns = df.columns.str.lower()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    logger.info(f"OHLCV loaded: {len(df):,} bars  [{df.index[0]} → {df.index[-1]}]")
    return df


def load_ohlcv_alpaca(
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str | None = None,
    timeframe: str = "5Min",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Pull historical bars from Alpaca and return a standard OHLCV frame.
    Caches to a local Parquet file so subsequent runs are instant."""
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        raise ImportError("alpaca-trade-api not installed")

    end = end or datetime.utcnow().strftime("%Y-%m-%d")

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_file = config.CACHE_DIR / f"ohlcv_{symbol}_{timeframe.lower()}.parquet"
    cache_meta = config.CACHE_DIR / f"ohlcv_{symbol}_{timeframe.lower()}_meta.json"
    cache_ttl  = 6 * 3600   # refresh at most every 6 hours

    if use_cache and cache_file.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        age  = time.time() - meta.get("fetched_at", 0)
        if age < cache_ttl:
            logger.info(f"OHLCV: loading from cache ({age/3600:.1f}h old) …")
            df = pd.read_parquet(cache_file)
            logger.success(f"OHLCV cache hit: {len(df):,} bars")
            return df

    # ── Fetch from Alpaca ─────────────────────────────────────────────────────
    api = tradeapi.REST(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.ALPACA_BASE_URL,
    )
    logger.info(f"Alpaca: fetching {symbol} {timeframe} bars {start} → {end}")

    bars = api.get_bars(
        symbol, timeframe,
        start=start, end=end,
        adjustment="all",
        feed="iex",
    ).df

    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars[["open", "high", "low", "close", "volume"]].sort_index()

    # ── Save to cache ─────────────────────────────────────────────────────────
    if use_cache:
        bars.to_parquet(cache_file)
        cache_meta.write_text(json.dumps({
            "fetched_at": time.time(),
            "symbol": symbol,
            "start": start,
            "end": end,
            "bars": len(bars),
        }))
        logger.success(f"Alpaca: {len(bars):,} bars downloaded and cached → {cache_file.name}")
    else:
        logger.success(f"Alpaca: {len(bars):,} bars downloaded")

    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Merge & enrich
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_dataset(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Merge intraday OHLCV with UST10Y macro data (forward-filled daily → bar)."""
    ust = fetch_ust10y(start=str(ohlcv.index[0].date()))

    # Align to bar timestamps: forward-fill daily yield onto intraday index
    ohlcv = ohlcv.copy()
    ohlcv["date"] = ohlcv.index.normalize().tz_localize(None)
    ust.index = pd.to_datetime(ust.index)
    yield_map = ust.reindex(ohlcv["date"].values, method="ffill").values
    ohlcv["ust10y"] = yield_map
    ohlcv = ohlcv.drop(columns=["date"])

    # Fill any remaining NaN yields with 0 (weekend/holiday edge cases)
    ohlcv["ust10y"] = ohlcv["ust10y"].ffill().fillna(0.0)
    logger.info(f"Dataset built: {len(ohlcv):,} bars with macro enrichment")
    return ohlcv
