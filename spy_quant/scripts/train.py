#!/usr/bin/env python3
"""
scripts/train.py
End-to-end training pipeline — optimised for low RAM (4 GB).

Usage
─────
  python scripts\train.py --alpaca
  python scripts\train.py --alpaca --start 2022-01-01
  python scripts\train.py --alpaca --no-trading-selection  # loss-only checkpoints
"""
import argparse
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import config
from data.loader import load_ohlcv_parquet, load_ohlcv_alpaca, build_raw_dataset
from data.features import compute_features
from data.preprocessing import preprocess
from data.dataset import make_loader
from models.diffusion import MultiTimeframeDiffusion
from models.trainer import train


def print_ram(label: str):
    try:
        import psutil
        m = psutil.virtual_memory()
        logger.info(label + " | RAM: " + str(round(m.used/1024**3, 1)) +
                    "/" + str(round(m.total/1024**3, 1)) + " GB (" +
                    str(m.percent) + "%)")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str)
    parser.add_argument("--alpaca",     action="store_true")
    parser.add_argument("--start",      type=str, default="2020-01-01")
    parser.add_argument("--end",        type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=config.EPOCHS)
    parser.add_argument("--device",     type=str, default=config.DEVICE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience",   type=int, default=80)
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Compute trading score every N epochs")
    parser.add_argument("--no-trading-selection", action="store_true",
                        help="Select checkpoints by val loss instead of trading score")
    args = parser.parse_args()

    # ── Auto batch size ───────────────────────────────────────────────────────
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        if args.batch_size is None:
            args.batch_size = 16 if avail < 2.0 else (32 if avail < 3.0 else config.BATCH_SIZE)
            logger.info("Auto batch_size=" + str(args.batch_size) +
                        " (" + str(round(avail, 1)) + " GB free)")
    except ImportError:
        args.batch_size = args.batch_size or 32

    print_ram("startup")

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Step 1/5: Loading OHLCV")
    if args.alpaca:
        raw = load_ohlcv_alpaca(start=args.start, end=args.end)
    elif args.data:
        raw = load_ohlcv_parquet(args.data)
    else:
        logger.error("Provide --data or --alpaca")
        sys.exit(1)
    print_ram("after load")

    # ── Features ──────────────────────────────────────────────────────────────
    logger.info("Step 2/5: Feature engineering")
    enriched = build_raw_dataset(raw)
    del raw; gc.collect()
    features = compute_features(enriched)
    del enriched; gc.collect()
    print_ram("after features")

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Step 3/5: Preprocessing")
    train_arr, val_arr, test_arr, feat_names = preprocess(features, fit_scaler=True)
    del features; del test_arr; gc.collect()
    logger.info("train=" + str(train_arr.shape) + "  val=" + str(val_arr.shape))
    print_ram("after preprocess")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Step 4/5: DataLoaders (batch=" + str(args.batch_size) + ")")
    train_loader = make_loader(train_arr, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = make_loader(val_arr,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=False)
    print_ram("after loaders")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Step 5/5: Model")
    model   = MultiTimeframeDiffusion(feature_dim=len(feat_names))
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: " + str(n_param))
    print_ram("after model init")

    # ── Train ─────────────────────────────────────────────────────────────────
    use_trading_sel = not args.no_trading_selection
    history = train(
        model,
        train_loader,
        val_loader,
        val_arr        = val_arr if use_trading_sel else None,
        epochs         = args.epochs,
        lr             = config.LEARNING_RATE,
        patience       = args.patience,
        device         = args.device,
        eval_interval  = args.eval_interval,
        use_trading_selection = use_trading_sel,
    )

    hist_path = config.MODEL_DIR / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    logger.success("History saved to " + str(hist_path))
    print_ram("finished")


if __name__ == "__main__":
    main()
