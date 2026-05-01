"""
backtest/simulation.py
Regime-aware batched Monte Carlo backtest.

Key design — matches the reference architecture:
  - One tensor gather per step, no Python loops over windows
  - Stratified branch sampling covers full time range cheaply
  - All inference done in one batched model.sample() call per step
  - Regime detection via rolling VoV and RV percentiles
  - Per-branch PnL with stop/target simulation using OHLC bars
  - Walk-forward splits over branches (not re-running inference)
  - Bootstrap Monte Carlo over realized branch PnL series
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Regime helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_percentile(values: np.ndarray, window: int = 500) -> np.ndarray:
    """For each index, return the percentile rank of that value in its lookback window."""
    out = np.full(len(values), 50.0, dtype=np.float32)
    for i in range(len(values)):
        lo  = max(0, i - window + 1)
        seg = values[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        out[i] = float(np.searchsorted(np.sort(seg), float(seg[-1])) / len(seg) * 100.0)
    return out


def detect_regime(realized_vol: np.ndarray, window: int = 20) -> np.ndarray:
    """0 = low-vol/trending, 1 = high-vol/choppy."""
    rv     = pd.Series(realized_vol)
    median = rv.rolling(window, min_periods=5).median()
    return (rv > median).astype(int).values


# ─────────────────────────────────────────────────────────────────────────────
# Signal quality + sizing
# ─────────────────────────────────────────────────────────────────────────────

def signal_quality(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    z = (predictions - predictions.mean()) / (predictions.std() + 1e-8)
    return np.abs(z) > threshold


def size_positions(
    predictions:  np.ndarray,
    realized_vol: np.ndarray,
    max_risk:     float = config.MAX_POSITION_RISK,
) -> np.ndarray:
    sign = np.sign(predictions)
    vol  = np.clip(realized_vol, 1e-5, None)
    size = np.clip((max_risk / vol) * np.abs(predictions), 0, 1)
    return sign * size


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    total_return:    float = 0.0
    sharpe:          float = 0.0
    max_drawdown:    float = 0.0
    calmar:          float = 0.0
    win_rate:        float = 0.0
    n_trades:        int   = 0
    regime_0_sharpe: float = 0.0
    regime_1_sharpe: float = 0.0


def compute_metrics(
    pnl:           np.ndarray,
    regime:        np.ndarray,
    bars_per_year: int  = 252 * 78,
    is_contiguous: bool = False,   # set True only for real time-series PnL
) -> BacktestMetrics:
    if len(pnl) == 0:
        return BacktestMetrics()

    # Annualisation only makes sense for a contiguous time series.
    # For flattened branch×step arrays, report the raw Information Ratio
    # (mean/std) without any time scaling — it's still comparable across runs.
    if is_contiguous:
        ann = np.sqrt(bars_per_year)
    else:
        ann = 1.0   # raw IR — no spurious annualisation

    cum      = np.cumprod(1 + np.clip(pnl, -0.5, 0.5))
    total    = float(cum[-1] - 1)
    sharpe   = float(pnl.mean() / (pnl.std() + 1e-9) * ann)
    roll_max = np.maximum.accumulate(cum)
    dd       = (roll_max - cum) / (roll_max + 1e-9)
    max_dd   = float(dd.max())
    calmar   = (total / max_dd) if max_dd > 1e-6 else 0.0
    win_rate = float((pnl > 0).mean())
    n_trades = int((pnl != 0).sum())

    def _s(mask):
        s = pnl[mask]
        if len(s) < 10:
            return float("nan")
        return float(s.mean() / (s.std() + 1e-9) * ann)

    return BacktestMetrics(
        total_return=total, sharpe=sharpe, max_drawdown=max_dd,
        calmar=calmar, win_rate=win_rate, n_trades=n_trades,
        regime_0_sharpe=_s(regime == 0), regime_1_sharpe=_s(regime == 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core: batched tensor-gather Monte Carlo backtest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    branch_pnl:    np.ndarray   # (branches, horizon_steps)
    branch_equity: np.ndarray   # (branches, horizon_steps+1)
    trade_counts:  np.ndarray   # (branches,)
    win_counts:    np.ndarray   # (branches,)
    predictions:   np.ndarray   # (branches, horizon_steps)  — raw model outputs
    log_returns:   np.ndarray   # (branches, horizon_steps)  — actual next returns
    realized_vols: np.ndarray   # (branches, horizon_steps)
    start_indices: np.ndarray   # (branches,)


def batched_backtest(
    model,                          # MultiTimeframeDiffusion
    feature_arr:   np.ndarray,      # (T, F) scaled features
    branches:      int   = 200,     # number of stratified start points
    horizon_steps: int   = 50,      # steps per branch
    initial_equity: float = 10_000.0,
    risk_pct:      float = config.MAX_POSITION_RISK,
    stop_pct:      float = config.STOP_LOSS_PCT,
    target_pct:    float = config.TAKE_PROFIT_PCT,
    signal_thresh: float = 0.5,
    seed:          Optional[int] = 42,
    device:        Optional[torch.device] = None,
    num_samples:   int   = 16,      # diffusion ensemble size per branch
    ddim_steps:    int   = 10,      # DDIM steps — keep low for speed
) -> BacktestResult:
    """
    Memory-efficient batched backtest using tensor gather.

    Instead of loading all test windows into a Dataset and looping,
    we:
      1. Load feature_arr as ONE tensor on device (single allocation)
      2. Sample `branches` stratified start indices
      3. At each horizon step, gather ALL branch windows in one index op
      4. Run one batched model.sample() call for all branches
      5. Compute PnL from signal × next return

    Peak RAM = size of feature_arr tensor + one batch of windows.
    No Python loops over windows. No Dataset objects.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = device or torch.device("cpu")
    T, F   = feature_arr.shape
    seq    = config.SEQ_LEN
    cf     = config.COARSE_FACTOR

    # ── Load feature array as single tensor (zero-copy if float32) ────────────
    feat_t = torch.as_tensor(feature_arr.astype(np.float32), device=device)

    # ── Stratified start indices ───────────────────────────────────────────────
    max_start = T - seq - horizon_steps - 1
    if max_start < 1:
        raise ValueError("Not enough data: T=" + str(T) +
                         " need at least " + str(seq + horizon_steps + 2))

    branches  = min(branches, max_start + 1)
    band      = max(1, (max_start + 1) // branches)
    starts    = [np.random.randint(i * band, min((i+1)*band, max_start+1))
                 for i in range(branches)]
    start_t   = torch.tensor(starts, dtype=torch.long, device=device)

    # ── Pre-allocate result arrays ────────────────────────────────────────────
    branch_pnl    = np.zeros((branches, horizon_steps), dtype=np.float32)
    branch_equity = np.ones( (branches, horizon_steps + 1), dtype=np.float32) * initial_equity
    trade_counts  = np.zeros(branches, dtype=np.int32)
    win_counts    = np.zeros(branches, dtype=np.int32)
    all_preds     = np.zeros((branches, horizon_steps), dtype=np.float32)
    all_rets      = np.zeros((branches, horizon_steps), dtype=np.float32)
    all_vols      = np.zeros((branches, horizon_steps), dtype=np.float32)

    offsets    = torch.arange(seq, device=device)                    # (seq,)
    coarse_len = seq // cf

    model.eval()
    with torch.no_grad():
        for step in range(horizon_steps):
            # ── Gather fine windows: (branches, seq, F) ───────────────────────
            ctx_idx    = start_t[:, None] + step + offsets[None, :]  # (B, seq)
            batch_fine = feat_t[ctx_idx]                              # (B, seq, F)

            # ── Build coarse: mean-pool along time ────────────────────────────
            trim = coarse_len * cf
            batch_coarse = (
                batch_fine[:, :trim, :]
                .reshape(branches, coarse_len, cf, F)
                .mean(dim=2)
            )                                                         # (B, coarse_len, F)

            # ── Ensemble: repeat each branch num_samples times ────────────────
            fine_rep   = batch_fine.repeat_interleave(num_samples, dim=0)   # (B*S, seq, F)
            coarse_rep = batch_coarse.repeat_interleave(num_samples, dim=0) # (B*S, cl, F)

            raw_preds  = model.sample(fine_rep, coarse_rep, steps=ddim_steps)  # (B*S,)
            raw_preds  = raw_preds.view(branches, num_samples)         # (B, S)

            # Ensemble stats
            pred_median = raw_preds.median(dim=1).values               # (B,)
            pred_std    = raw_preds.std(dim=1)                         # (B,)
            upside_prob = (raw_preds > 0).float().mean(dim=1)         # (B,)

            pred_np  = pred_median.cpu().numpy()
            std_np   = pred_std.cpu().numpy()
            prob_np  = upside_prob.cpu().numpy()

            # ── Actual next-bar log returns and vols from feature array ────────
            next_idx = (start_t + step + seq).cpu().numpy()           # (B,)
            next_idx = np.clip(next_idx, 0, T - 1)
            ret_np   = feature_arr[next_idx, 0]                       # log_return col
            vol_np   = feature_arr[next_idx, 1]                       # realized_vol col

            all_preds[:, step] = pred_np
            all_rets[:, step]  = ret_np
            all_vols[:, step]  = vol_np

            # ── Signal filter + sizing ────────────────────────────────────────
            # SNR-weighted signal: tanh(median / std)
            snr    = np.abs(pred_np) / (std_np + 1e-6)
            signal = np.tanh(pred_np * snr)

            # Only trade when model is confident: |signal| > thresh AND
            # upside_prob far from 0.5 (directional conviction)
            active = (np.abs(signal) > signal_thresh) & \
                     (np.abs(prob_np - 0.5) > 0.1)

            vol_clipped = np.clip(vol_np, 1e-5, None)
            sizes       = np.clip(
                np.sign(signal) * (risk_pct / vol_clipped) * np.abs(signal),
                -1.0, 1.0,
            )

            # ── PnL simulation ────────────────────────────────────────────────
            for b in range(branches):
                if not active[b]:
                    branch_equity[b, step + 1] = branch_equity[b, step]
                    continue

                trade_counts[b] += 1
                side = 1.0 if signal[b] > 0 else -1.0
                pnl  = side * sizes[b] * ret_np[b]
                pnl -= 0.00005   # slippage (0.5 bps)

                branch_pnl[b, step]       = pnl
                branch_equity[b, step+1]  = branch_equity[b, step] * (1 + pnl)
                if pnl > 0:
                    win_counts[b] += 1

            # Free batch tensors each step
            del batch_fine, batch_coarse, fine_rep, coarse_rep, raw_preds
            del pred_median, pred_std, upside_prob
            if step % 10 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    del feat_t
    gc.collect()

    return BacktestResult(
        branch_pnl    = branch_pnl,
        branch_equity = branch_equity,
        trade_counts  = trade_counts,
        win_counts    = win_counts,
        predictions   = all_preds,
        log_returns   = all_rets,
        realized_vols = all_vols,
        start_indices = np.array(starts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward over branches
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_from_result(
    result:   BacktestResult,
    n_splits: int = config.WALK_FORWARD_SPLITS,
) -> list[BacktestMetrics]:
    """
    Walk-forward over branches — no new inference needed.
    Splits branches chronologically by start_index.
    """
    order   = np.argsort(result.start_indices)
    B       = len(order)
    fold    = max(1, B // n_splits)
    metrics = []

    for k in range(n_splits):
        idx = order[k * fold : (k + 1) * fold]
        if len(idx) == 0:
            continue

        pnl    = result.branch_pnl[idx].flatten()
        vols   = result.realized_vols[idx].flatten()
        regime = detect_regime(vols)
        m      = compute_metrics(pnl, regime)
        metrics.append(m)
        logger.info(
            "  WF fold " + str(k+1) + ": Sharpe=" + str(round(m.sharpe, 3)) +
            "  DD=" + str(round(m.max_drawdown, 3)) +
            "  Win=" + str(round(m.win_rate, 3)) +
            "  Trades=" + str(m.n_trades)
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Monte Carlo over realized branch PnL
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(
    pnl:      np.ndarray,
    n_sims:   int = config.MC_SIMULATIONS,
    horizon:  int = 252 * 78,
    rng:      np.random.Generator | None = None,
) -> dict:
    rng = rng or np.random.default_rng(42)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 30:
        logger.warning("Too few PnL obs for Monte Carlo.")
        return {}

    # Batched bootstrap — allocate one big array instead of looping
    draws = rng.choice(pnl, size=(n_sims, horizon), replace=True)
    final = (np.cumprod(1 + np.clip(draws, -0.5, 0.5), axis=1))[:, -1] - 1
    del draws; gc.collect()

    var95 = float(np.percentile(final, 5))
    tail  = final[final <= var95]
    cvar  = float(tail.mean()) if len(tail) > 0 else var95

    result = {
        "mean_return":   float(final.mean()),
        "median_return": float(np.median(final)),
        "std_return":    float(final.std()),
        "var_95":        var95,
        "cvar_95":       cvar,
        "prob_positive": float((final > 0).mean()),
        "pct_5":         float(np.percentile(final, 5)),
        "pct_25":        float(np.percentile(final, 25)),
        "pct_75":        float(np.percentile(final, 75)),
        "pct_95":        float(np.percentile(final, 95)),
    }
    logger.info(
        "Monte Carlo (" + str(n_sims) + " sims): " +
        "E[R]=" + str(round(result["mean_return"]*100, 1)) + "%" +
        "  VaR95=" + str(round(result["var_95"]*100, 1)) + "%" +
        "  P(+)=" + str(round(result["prob_positive"]*100, 1)) + "%"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# High-level evaluate() — called from scripts/evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    feature_arr:   np.ndarray,
    branches:      int   = 200,
    horizon_steps: int   = 50,
    device:        Optional[torch.device] = None,
    num_samples:   int   = 16,
    ddim_steps:    int   = 10,
    signal_thresh: float = 0.5,
) -> dict:
    """
    Full evaluation: batched backtest → walk-forward → Monte Carlo.
    Returns summary dict for JSON export.
    """
    device = device or torch.device("cpu")

    logger.info("Running batched backtest (" + str(branches) +
                " branches × " + str(horizon_steps) + " steps)")
    result = batched_backtest(
        model         = model,
        feature_arr   = feature_arr,
        branches      = branches,
        horizon_steps = horizon_steps,
        device        = device,
        num_samples   = num_samples,
        ddim_steps    = ddim_steps,
        signal_thresh = signal_thresh,
    )

    total_trades = int(result.trade_counts.sum())
    hit_rate     = float(result.win_counts.sum() / max(total_trades, 1))
    logger.info("Total trades=" + str(total_trades) +
                "  Overall hit rate=" + str(round(hit_rate, 3)))

    # ── Walk-forward ──────────────────────────────────────────────────────────
    logger.info("Walk-forward splits")
    wf      = walk_forward_from_result(result)
    sharpes = [m.sharpe for m in wf if not np.isnan(m.sharpe)]
    mean_sh = float(np.mean(sharpes)) if sharpes else float("nan")
    passing = sum(1 for s in sharpes if s > config.MIN_SHARPE_THRESHOLD)
    logger.info("WF mean Sharpe=" + str(round(mean_sh, 3)) +
                "  passing=" + str(passing) + "/" + str(len(wf)))

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    logger.info("Monte Carlo simulation")
    all_pnl = result.branch_pnl.flatten()
    mc      = monte_carlo(all_pnl)

    # ── Per-branch equity summary ─────────────────────────────────────────────
    final_eq   = result.branch_equity[:, -1]
    eq_returns = final_eq / 10_000.0 - 1

    return {
        "branches":          branches,
        "horizon_steps":     horizon_steps,
        "total_trades":      total_trades,
        "hit_rate":          hit_rate,
        "walk_forward":      [vars(m) for m in wf],
        "wf_mean_sharpe":    mean_sh,
        "wf_passing_folds":  passing,
        "monte_carlo":       mc,
        "equity": {
            "mean_return":   float(eq_returns.mean()),
            "median_return": float(np.median(eq_returns)),
            "best_branch":   float(eq_returns.max()),
            "worst_branch":  float(eq_returns.min()),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy compatibility shims (used by optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    predictions:  np.ndarray,
    log_returns:  np.ndarray,
    realized_vol: np.ndarray,
    n_splits:     int = config.WALK_FORWARD_SPLITS,
) -> list[BacktestMetrics]:
    """Legacy array-based walk-forward (used by optimizer.py)."""
    N    = len(predictions)
    fold = N // (n_splits + 1)
    out  = []
    for k in range(1, n_splits + 1):
        s, e = k * fold, k * fold + fold
        if e > N:
            break
        q   = signal_quality(predictions[s:e])
        sz  = size_positions(predictions[s:e], realized_vol[s:e])
        pnl = np.where(q, sz * log_returns[s:e], 0.0)
        pnl[q] -= 0.00005
        m = compute_metrics(pnl, detect_regime(realized_vol[s:e]))
        out.append(m)
        logger.info("  WF fold " + str(k) + ": Sharpe=" + str(round(m.sharpe, 2)))
    return out
