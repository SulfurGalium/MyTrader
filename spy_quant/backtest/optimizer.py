"""
backtest/optimizer.py
Walk-forward hyperparameter search over signal quality threshold,
position sizing scale, and stop/take-profit ratios.

Uses a grid search within each expanding training window so parameters
are always selected on past data only — never on the test fold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backtest.simulation import (
    detect_regime, signal_quality, size_positions,
    compute_metrics, BacktestMetrics,
)


@dataclass
class ParamSet:
    signal_threshold: float   # |z-score| cutoff for signal quality
    risk_scale:       float   # multiplier on MAX_POSITION_RISK
    sl_pct:           float   # stop-loss percentage
    tp_pct:           float   # take-profit percentage
    sharpe:           float = 0.0

    def as_dict(self) -> dict:
        return {
            "signal_threshold": self.signal_threshold,
            "risk_scale":       self.risk_scale,
            "sl_pct":           self.sl_pct,
            "tp_pct":           self.tp_pct,
        }


# ── Grid definition ────────────────────────────────────────────────────────────

PARAM_GRID = {
    "signal_threshold": [0.3, 0.5, 0.7, 1.0],
    "risk_scale":       [0.5, 1.0, 1.5],
    "sl_pct":           [0.003, 0.005, 0.008],
    "tp_pct":           [0.010, 0.015, 0.020],
}


def _simulate(
    preds: np.ndarray,
    rets:  np.ndarray,
    vols:  np.ndarray,
    p:     ParamSet,
) -> BacktestMetrics:
    quality = signal_quality(preds, threshold=p.signal_threshold)
    sizes   = size_positions(
        preds, vols,
        max_risk=config.MAX_POSITION_RISK * p.risk_scale,
    )
    # Simplified PnL using fixed SL/TP ratio as a cost proxy
    sl_tp_factor = p.tp_pct / (p.sl_pct + 1e-9)   # reward/risk
    adjusted_rets = np.where(rets > 0, rets * sl_tp_factor, rets)
    pnl = np.where(quality, sizes * adjusted_rets, 0.0)
    pnl[quality] -= 0.00005   # slippage
    return compute_metrics(pnl, detect_regime(vols))


def grid_search(
    preds: np.ndarray,
    rets:  np.ndarray,
    vols:  np.ndarray,
) -> ParamSet:
    """Find the ParamSet with highest Sharpe on the given data."""
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    best   = ParamSet(0.5, 1.0, 0.005, 0.015, sharpe=-1e9)

    for combo in combos:
        p = ParamSet(**dict(zip(keys, combo)))
        m = _simulate(preds, rets, vols, p)
        if m.sharpe > best.sharpe:
            best = ParamSet(**dict(zip(keys, combo)), sharpe=m.sharpe)

    return best


# ── Walk-forward optimizer ────────────────────────────────────────────────────

@dataclass
class WFOResult:
    fold:        int
    best_params: ParamSet
    oos_metrics: BacktestMetrics


def walk_forward_optimize(
    predictions:  np.ndarray,
    log_returns:  np.ndarray,
    realized_vol: np.ndarray,
    n_splits:     int = config.WALK_FORWARD_SPLITS,
) -> list[WFOResult]:
    """
    Expanding-window WFO:
      IS window  → grid search for best params
      OOS window → evaluate with those params
    """
    N    = len(predictions)
    fold = N // (n_splits + 1)
    results: list[WFOResult] = []

    logger.info(f"Walk-forward optimization: {n_splits} folds, grid size={len(list(itertools.product(*PARAM_GRID.values())))}")

    for k in range(1, n_splits + 1):
        is_end   = k * fold
        oos_end  = min(is_end + fold, N)
        if oos_end <= is_end:
            break

        # In-sample
        p_is = predictions[:is_end]
        r_is = log_returns[:is_end]
        v_is = realized_vol[:is_end]

        # Out-of-sample
        p_oos = predictions[is_end:oos_end]
        r_oos = log_returns[is_end:oos_end]
        v_oos = realized_vol[is_end:oos_end]

        best = grid_search(p_is, r_is, v_is)
        oos  = _simulate(p_oos, r_oos, v_oos, best)

        results.append(WFOResult(fold=k, best_params=best, oos_metrics=oos))
        logger.info(
            f"  Fold {k}: IS best params={best.as_dict()}  "
            f"OOS Sharpe={oos.sharpe:.3f}  DD={oos.max_drawdown:.2%}"
        )

    return results


def summarize_wfo(results: list[WFOResult]) -> dict:
    sharpes  = [r.oos_metrics.sharpe for r in results]
    dds      = [r.oos_metrics.max_drawdown for r in results]
    win_rates = [r.oos_metrics.win_rate for r in results]

    return {
        "n_folds":        len(results),
        "mean_sharpe":    float(np.mean(sharpes)),
        "std_sharpe":     float(np.std(sharpes)),
        "min_sharpe":     float(np.min(sharpes)),
        "max_sharpe":     float(np.max(sharpes)),
        "mean_max_dd":    float(np.mean(dds)),
        "mean_win_rate":  float(np.mean(win_rates)),
        "passing_folds":  int(sum(s > config.MIN_SHARPE_THRESHOLD for s in sharpes)),
        "param_stability": _param_stability(results),
    }


def _param_stability(results: list[WFOResult]) -> dict[str, Any]:
    """Check whether the same parameters win consistently."""
    out = {}
    for key in ("signal_threshold", "risk_scale", "sl_pct", "tp_pct"):
        vals = [getattr(r.best_params, key) for r in results]
        out[key] = {"values": vals, "unique": len(set(vals))}
    return out
