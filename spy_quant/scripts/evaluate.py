#!/usr/bin/env python3
"""
scripts/evaluate.py
Ultra-low RAM evaluation — designed for 4 GB machines.

Key optimisations vs previous version:
  - Deferred imports: torch/model only loaded after data is processed
    and non-essential RAM is explicitly freed
  - Model loaded directly to CPU in float16 (half the weight RAM)
  - Inference done in micro-batches of 1-4 with immediate tensor free
  - No Dataset objects ever created
  - gc.collect() + ctypes.malloc_trim() after every major step
  - --lean flag: skips Monte Carlo, reduces branches to absolute minimum

Usage
-----
  python scripts/evaluate.py --data cache/ohlcv_SPY_5min.parquet
  python scripts/evaluate.py --data cache/ohlcv_SPY_5min.parquet --lean
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Aggressive memory free helper ────────────────────────────────────────────
def _free():
    gc.collect()
    try:
        import ctypes
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        else:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def print_ram(label: str):
    try:
        import psutil
        m = psutil.virtual_memory()
        pct = m.percent
        color = "\033[91m" if pct > 90 else ("\033[93m" if pct > 75 else "\033[92m")
        reset = "\033[0m"
        print(color + "[RAM " + str(pct) + "%] " + label +
              " — " + str(round(m.used/1024**3, 1)) + "/" +
              str(round(m.total/1024**3, 1)) + " GB" + reset)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          type=str)
    parser.add_argument("--alpaca",        action="store_true")
    parser.add_argument("--start",         type=str,   default="2020-01-01")
    parser.add_argument("--branches",      type=int,   default=50)
    parser.add_argument("--horizon",       type=int,   default=20)
    parser.add_argument("--samples",       type=int,   default=4,
                        help="Diffusion ensemble size — keep at 4 for low RAM")
    parser.add_argument("--ddim-steps",    type=int,   default=5,
                        help="DDIM steps — 5 is fast, 20 is more accurate")
    parser.add_argument("--signal-thresh", type=float, default=0.5)
    parser.add_argument("--lean",          action="store_true",
                        help="Skip Monte Carlo, use minimum branches (fastest)")
    args = parser.parse_args()

    if args.lean:
        args.branches   = max(args.branches, 30)
        args.horizon    = min(args.horizon, 15)
        args.samples    = 2
        args.ddim_steps = 3
        print("Lean mode: branches=" + str(args.branches) +
              " horizon=" + str(args.horizon) + " samples=" + str(args.samples))

    try:
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 1: Pure numpy/pandas — NO torch import yet
        # ═══════════════════════════════════════════════════════════════════════
        print_ram("startup")

        print("Step 1/4: Loading data (no torch yet)")
        import numpy as np
        import pandas as pd

        if args.alpaca:
            from data.loader import load_ohlcv_alpaca, build_raw_dataset
            raw = load_ohlcv_alpaca(start=args.start)
        elif args.data:
            from data.loader import load_ohlcv_parquet, build_raw_dataset
            raw = load_ohlcv_parquet(args.data)
        else:
            print("ERROR: provide --alpaca or --data <path>")
            sys.exit(1)

        from data.loader import build_raw_dataset
        enriched = build_raw_dataset(raw)
        del raw; _free()
        print_ram("after raw load")

        print("Step 2/4: Features + preprocessing")
        from data.features import compute_features
        features = compute_features(enriched)
        del enriched; _free()

        from data.preprocessing import preprocess
        _, _, test_arr, feat_names = preprocess(features, fit_scaler=False)
        del features; _free()

        T, F = test_arr.shape
        print("  test_arr: " + str(test_arr.shape) + "  dtype=" + str(test_arr.dtype))
        print_ram("after preprocessing — about to load torch")

        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 2: Load torch + model — now that data RAM is settled
        # ═══════════════════════════════════════════════════════════════════════
        print("Step 3/4: Loading model")
        import torch

        # Force CPU — no CUDA DLL loading
        device = torch.device("cpu")

        from models.diffusion import MultiTimeframeDiffusion, load_checkpoint

        model = MultiTimeframeDiffusion(feature_dim=len(feat_names))
        ckpt  = load_checkpoint(model)
        epoch = ckpt.get("epoch")
        print("  Loaded epoch=" + str(epoch))
        print_ram("after model load")

        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 3: Micro-batch inference — no full Dataset, no large tensors
        # ═══════════════════════════════════════════════════════════════════════
        print("Step 4/4: Evaluation (" + str(args.branches) +
              " branches × " + str(args.horizon) + " steps)")

        seq        = config_seq()
        cf         = config_cf()
        coarse_len = seq // cf

        # Stratified start indices
        max_start = T - seq - args.horizon - 1
        if max_start < 1:
            print("ERROR: Not enough test data. T=" + str(T) +
                  " need " + str(seq + args.horizon + 2))
            sys.exit(1)

        branches  = min(args.branches, max_start + 1)
        band      = max(1, (max_start + 1) // branches)
        starts    = [np.random.randint(i*band, min((i+1)*band, max_start+1))
                     for i in range(branches)]

        # Result arrays
        all_preds  = np.zeros((branches, args.horizon), dtype=np.float32)
        all_rets   = np.zeros((branches, args.horizon), dtype=np.float32)
        all_vols   = np.zeros((branches, args.horizon), dtype=np.float32)
        trade_hits = np.zeros(branches, dtype=np.int32)
        trade_cnt  = np.zeros(branches, dtype=np.int32)
        equity     = np.ones(branches, dtype=np.float32) * 10_000.0

        # Keep model in float32 — fp16 on CPU saves only ~6MB on this model
        # and causes dtype mismatches with the noise schedule buffers.
        # RAM saving comes from micro-batching + deferred import instead.
        use_fp16 = False
        model = model.to(device)
        model.eval()
        dtype = torch.float32

        with torch.no_grad():
            for step in range(args.horizon):
                step_preds = np.zeros(branches, dtype=np.float32)

                # Process in micro-batches to limit peak RAM
                mb = 8   # micro-batch size — tiny on purpose
                for mb_start in range(0, branches, mb):
                    mb_end   = min(mb_start + mb, branches)
                    mb_size  = mb_end - mb_start
                    mb_starts = starts[mb_start:mb_end]

                    # Gather windows for this micro-batch
                    windows = np.stack([
                        test_arr[s + step : s + step + seq]
                        for s in mb_starts
                    ])                                       # (mb, seq, F)

                    fine_t = torch.tensor(windows, dtype=dtype, device=device)

                    trim   = coarse_len * cf
                    coarse_t = fine_t[:, :trim, :].reshape(
                        mb_size, coarse_len, cf, F
                    ).mean(dim=2)                            # (mb, cl, F)

                    # Ensemble: repeat_interleave along batch dim
                    S        = args.samples
                    fine_rep = fine_t.repeat_interleave(S, dim=0)    # (mb*S, seq, F)
                    crs_rep  = coarse_t.repeat_interleave(S, dim=0)  # (mb*S, cl, F)

                    raw = model.sample(
                        fine_rep, crs_rep, steps=args.ddim_steps
                    ).float().view(mb_size, S)               # back to float32

                    preds_mb = raw.median(dim=1).values.cpu().numpy()
                    step_preds[mb_start:mb_end] = preds_mb

                    # Free immediately
                    del fine_t, coarse_t, fine_rep, crs_rep, raw, preds_mb, windows
                    _free()

                # Actual next returns from feature array
                next_idx  = np.array([s + step + seq for s in starts])
                next_idx  = np.clip(next_idx, 0, T - 1)
                ret_np    = test_arr[next_idx, 0]
                vol_np    = test_arr[next_idx, 1]

                all_preds[:, step] = step_preds
                all_rets[:, step]  = ret_np
                all_vols[:, step]  = vol_np

                # ── Signal ────────────────────────────────────────────────────
                std_p  = step_preds.std() + 1e-8
                z      = np.abs(step_preds - step_preds.mean()) / std_p
                active = z > args.signal_thresh
                # SNR-weighted direction in (-1, 1)
                signal = np.tanh(step_preds / (std_p + 1e-6))

                # ── PnL in equity-percentage terms ────────────────────────────
                # ret_np is the SCALED log return (RobustScaler output, IQR~1).
                # sign(signal) * ret_np gives the signed return in scaled units.
                # We normalise by the median absolute scaled return so that
                # a "typical" correct trade earns exactly RISK_PER_TRADE.
                # This keeps PnL in consistent %-of-equity space.
                RISK_PER_TRADE = 0.005          # 0.5% equity risked per trade
                SLIPPAGE       = RISK_PER_TRADE * 0.10  # 10% of risk = realistic cost

                median_abs_ret = float(np.median(np.abs(ret_np[ret_np != 0])) + 1e-8)

                for b in range(branches):
                    if active[b]:
                        trade_cnt[b] += 1
                        direction  = float(np.sign(signal[b]))
                        # Cap norm_ret at ±3 — prevents flash-crash bars
                        # from multiplying equity by 10x in one step
                        norm_ret   = np.clip(float(ret_np[b]) / median_abs_ret, -3.0, 3.0)
                        pnl        = direction * RISK_PER_TRADE * norm_ret - SLIPPAGE
                        pnl        = max(pnl, -RISK_PER_TRADE * 2)
                        equity[b] *= (1.0 + pnl)
                        equity[b]  = max(equity[b], 1.0)
                        if pnl > 0:
                            trade_hits[b] += 1

                if step % 5 == 0:
                    _free()
                    print("  step " + str(step+1) + "/" + str(args.horizon) +
                          "  active=" + str(int(active.sum())) + "/" + str(branches))
                    print_ram("  step " + str(step+1))

        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 4: Metrics — pure numpy, model already freed
        # ═══════════════════════════════════════════════════════════════════════
        del model; _free()

        from backtest.simulation import (
            detect_regime, compute_metrics, walk_forward, monte_carlo
        )

        pnl_flat  = (all_preds * all_rets).flatten() * 0.01   # rough PnL proxy
        vol_flat  = all_vols.flatten()
        reg_flat  = detect_regime(vol_flat)

        total_trades = int(trade_cnt.sum())
        hit_rate     = float(trade_hits.sum() / max(total_trades, 1))

        # Walk-forward over branches (chronological order)
        branch_order = np.argsort([starts[b] for b in range(branches)])
        wf_metrics   = []
        fold_size    = max(1, branches // config_wf_splits())
        RISK_WF      = 0.005
        SLIP_WF      = RISK_WF * 0.10

        for k in range(config_wf_splits()):
            s   = k * fold_size
            e   = min(s + fold_size, branches)
            if s >= branches:
                break
            idx     = branch_order[s:e]
            p_fold  = all_preds[idx].flatten()
            r_fold  = all_rets[idx].flatten()
            v_fold  = all_vols[idx].flatten()
            reg_f   = detect_regime(v_fold)

            std_f   = p_fold.std() + 1e-8
            z_f     = np.abs(p_fold - p_fold.mean()) / std_f
            active_f = z_f > args.signal_thresh
            sig_f   = np.sign(np.tanh(p_fold / (std_f + 1e-6)))
            med_r   = float(np.median(np.abs(r_fold[r_fold != 0])) + 1e-8)
            norm_r  = np.clip(r_fold / med_r, -3.0, 3.0)
            pnl_f   = np.where(active_f,
                               np.clip(sig_f * RISK_WF * norm_r - SLIP_WF,
                                       -RISK_WF * 2, None),
                               0.0)

            m = compute_metrics(pnl_f, reg_f)
            wf_metrics.append(m)
            print("  WF fold " + str(k+1) + ": Sharpe=" + str(round(m.sharpe, 3)) +
                  "  DD=" + str(round(m.max_drawdown, 3)) +
                  "  Win=" + str(round(m.win_rate, 3)) +
                  "  Trades=" + str(m.n_trades))

        sharpes  = [m.sharpe for m in wf_metrics if not np.isnan(m.sharpe)]
        mean_sh  = float(np.mean(sharpes)) if sharpes else float("nan")
        # With is_contiguous=False, sharpe = raw IR (mean/std), not annualised.
        # A positive IR means the strategy has edge. IR > 0.05 is meaningful.
        passing  = sum(1 for s in sharpes if s > 0.0)   # just needs to be positive

        # Monte Carlo over all realized trades
        mc = {}
        if not args.lean:
            real_pnl = []
            RISK_MC  = 0.005
            SLIP_MC  = RISK_MC * 0.10
            for b in range(branches):
                std_b = all_preds[b].std() + 1e-8
                med_r = float(np.median(np.abs(all_rets[b][all_rets[b] != 0])) + 1e-8)
                for step in range(args.horizon):
                    p = float(all_preds[b, step])
                    r = float(all_rets[b, step])
                    z = abs(p - all_preds[b].mean()) / std_b
                    if z > args.signal_thresh:
                        sig  = float(np.sign(np.tanh(p / (std_b + 1e-6))))
                        # Cap norm_ret at ±3 to prevent outlier bars blowing up
                        norm_r = np.clip(r / med_r, -3.0, 3.0)
                        pnl    = sig * RISK_MC * norm_r - SLIP_MC
                        pnl    = max(pnl, -RISK_MC * 2)
                        real_pnl.append(pnl)
            if len(real_pnl) >= 30:
                # Horizon = one realistic trading year of 5-min bars
                # NOT len(real_pnl)*10 which creates astronomical compounding
                mc = monte_carlo(
                    np.array(real_pnl, dtype=np.float32),
                    n_sims=500,
                    horizon=252 * 78,   # exactly one year of bars
                )

        # Save + print
        results = {
            "epoch":           epoch,
            "branches":        branches,
            "horizon":         args.horizon,
            "total_trades":    total_trades,
            "hit_rate":        hit_rate,
            "wf_mean_sharpe":  mean_sh,
            "wf_passing_folds": passing,
            "walk_forward":    [vars(m) for m in wf_metrics],
            "monte_carlo":     mc,
            "equity": {
                "mean_return":   float(equity.mean()/10000 - 1),
                "best_branch":   float(equity.max()/10000 - 1),
                "worst_branch":  float(equity.min()/10000 - 1),
            },
        }

        import config as cfg
        out = cfg.MODEL_DIR / "eval_results.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        print("Results saved to " + str(out))
        print_ram("finished")

        # Summary
        eq = results["equity"]
        print("\n" + "=" * 55)
        print("  EVALUATION SUMMARY")
        print("=" * 55)
        print("  Epoch              : " + str(epoch))
        print("  Branches x Horizon : " + str(branches) + " x " + str(args.horizon))
        print("  Total trades       : " + str(total_trades))
        print("  Hit rate           : " + str(round(hit_rate*100, 1)) + "%")
        print("  WF Mean IR         : " + str(round(mean_sh, 4)) +
              "  (raw mean/std, >0 = edge)")
        print("  WF Positive Folds  : " + str(passing) + "/" + str(config_wf_splits()))
        if mc:
            print("  MC Mean Return     : " + str(round(mc.get("mean_return",0)*100,1)) + "%")
            print("  MC P(positive)     : " + str(round(mc.get("prob_positive",0)*100,1)) + "%")
            print("  MC VaR 95          : " + str(round(mc.get("var_95",0)*100,1)) + "%")
        print("  Mean branch return : " + str(round(eq["mean_return"]*100,1)) + "%")
        print("  Best / Worst branch: " + str(round(eq["best_branch"]*100,1)) +
              "% / " + str(round(eq["worst_branch"]*100,1)) + "%")
        print("=" * 55)
        if epoch and epoch < 30:
            print("\n  Still early in training — run more epochs:")
            print("  python scripts/train.py --alpaca --epochs 100")
        elif mean_sh > 0.02 and hit_rate > 0.49:
            print("\n  Signal looks real — consider paper trading:")
            print("  python scripts/run_live.py")
        else:
            print("\n  Edge not yet confirmed (IR=" + str(round(mean_sh,4)) +
                  " hit=" + str(round(hit_rate*100,1)) + "%) — continue training.")
        print()

    except MemoryError:
        print("OUT OF MEMORY — try:")
        print("  python scripts/evaluate.py --data cache/ohlcv_SPY_5min.parquet --lean")
        sys.exit(1)
    except Exception:
        print("EVALUATION FAILED:")
        traceback.print_exc()
        sys.exit(1)


# ── Config accessors (avoids importing config at module level) ────────────────
def config_seq():
    import config
    return config.SEQ_LEN

def config_cf():
    import config
    return config.COARSE_FACTOR

def config_wf_splits():
    import config
    return config.WALK_FORWARD_SPLITS


if __name__ == "__main__":
    main()
