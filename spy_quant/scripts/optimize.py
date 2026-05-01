#!/usr/bin/env python3
"""
scripts/optimize.py
Walk-forward hyperparameter optimization over the test set.

Usage
─────
  python scripts/optimize.py --data path/to/spy_5min.parquet
  python scripts/optimize.py --alpaca --start 2020-01-01
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from loguru import logger

import config
from data.loader import load_ohlcv_parquet, load_ohlcv_alpaca, build_raw_dataset
from data.features import compute_features
from data.preprocessing import preprocess
from data.dataset import SPYWindowDataset
from models.diffusion import MultiTimeframeDiffusion, load_checkpoint
from backtest.optimizer import walk_forward_optimize, summarize_wfo


def main():
    parser = argparse.ArgumentParser(description="Walk-forward optimization")
    parser.add_argument("--data",   type=str)
    parser.add_argument("--alpaca", action="store_true")
    parser.add_argument("--start",  type=str, default="2020-01-01")
    parser.add_argument("--device", type=str, default=config.DEVICE)
    args = parser.parse_args()

    if args.alpaca:
        raw = load_ohlcv_alpaca(start=args.start)
    else:
        raw = load_ohlcv_parquet(args.data)

    enriched = build_raw_dataset(raw)
    features = compute_features(enriched)
    _, _, test_arr, feat_names = preprocess(features, fit_scaler=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = MultiTimeframeDiffusion(feature_dim=len(feat_names))
    ckpt   = load_checkpoint(model)
    model  = model.to(device).eval()
    logger.info(f"Loaded checkpoint epoch={ckpt.get('epoch')} on {device}")

    ds          = SPYWindowDataset(test_arr)
    predictions, log_returns = [], []

    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            end = min(start + batch_size, len(ds))
            batch = [ds[i] for i in range(start, end)]
            fine   = torch.stack([b[0] for b in batch]).to(device)
            coarse = torch.stack([b[1] for b in batch]).to(device)
            preds  = model.sample(fine, coarse, steps=20).cpu().numpy()
            predictions.extend(preds.tolist())
            log_returns.extend([b[2].item() for b in batch])

    preds_np = np.array(predictions)
    rets_np  = np.array(log_returns)
    vols_np  = test_arr[config.SEQ_LEN : config.SEQ_LEN + len(preds_np), 1]

    logger.info(f"Running WFO on {len(preds_np):,} test predictions …")
    results = walk_forward_optimize(preds_np, rets_np, vols_np)
    summary = summarize_wfo(results)

    out = {
        "summary": summary,
        "folds": [
            {
                "fold":        r.fold,
                "best_params": r.best_params.as_dict(),
                "oos_sharpe":  r.oos_metrics.sharpe,
                "oos_max_dd":  r.oos_metrics.max_drawdown,
                "oos_win_rate":r.oos_metrics.win_rate,
                "oos_trades":  r.oos_metrics.n_trades,
            }
            for r in results
        ],
    }

    out_path = config.MODEL_DIR / "wfo_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.success(f"WFO results saved → {out_path}")

    print("\n" + "═" * 55)
    print("WALK-FORWARD OPTIMIZATION SUMMARY")
    print("═" * 55)
    print(f"  Mean OOS Sharpe  : {summary['mean_sharpe']:.3f}  ±{summary['std_sharpe']:.3f}")
    print(f"  Min / Max Sharpe : {summary['min_sharpe']:.3f} / {summary['max_sharpe']:.3f}")
    print(f"  Mean Max DD      : {summary['mean_max_dd']:.1%}")
    print(f"  Passing Folds    : {summary['passing_folds']} / {summary['n_folds']}")
    print(f"  Mean Win Rate    : {summary['mean_win_rate']:.1%}")
    print("\n  Parameter stability:")
    for k, v in summary["param_stability"].items():
        print(f"    {k:<22}: {v['values']}  (unique={v['unique']})")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
