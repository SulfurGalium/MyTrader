#!/usr/bin/env python3
"""
scripts/monitor.py
System health check — prints a status dashboard.
Safe to run at any time without affecting the live loop.

Usage
─────
  python scripts/monitor.py
  python scripts/monitor.py --watch     # refresh every 30s
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil
import config


def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _tail(path: Path, n: int = 10) -> list[str]:
    if not path.exists():
        return ["(no log file)"]
    with open(path) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def _model_info() -> dict:
    ckpt = config.MODEL_DIR / "diffusion_latest.pt"
    if not ckpt.exists():
        return {"status": "NOT FOUND"}
    stat = ckpt.stat()
    return {
        "path":     str(ckpt),
        "size":     _fmt_bytes(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _eval_summary() -> dict:
    p = config.MODEL_DIR / "eval_results.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        mc = d.get("monte_carlo", {})
        return {
            "wf_mean_sharpe":    round(d.get("wf_mean_sharpe", float("nan")), 3),
            "wf_passing_folds":  d.get("wf_passing_folds"),
            "mc_mean_return":    f"{mc.get('mean_return', 0):.1%}",
            "mc_prob_positive":  f"{mc.get('prob_positive', 0):.1%}",
            "mc_var95":          f"{mc.get('var_95', 0):.1%}",
        }
    except Exception:
        return {"error": "Could not parse eval_results.json"}


def _cache_info() -> dict:
    meta = config.CACHE_DIR / "ust10y_meta.json"
    if not meta.exists():
        return {"ust10y": "not cached"}
    d = json.loads(meta.read_text())
    age_h = (time.time() - d.get("fetched_at", 0)) / 3600
    return {"ust10y_age_hours": round(age_h, 1)}


def _live_mode() -> str:
    return "🔴 LIVE" if config.LIVE_TRADING_ENABLED else "🟡 DRY-RUN / PAPER"


def _recent_log() -> list[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    log   = config.LOG_DIR / f"live_{today}.log"
    return _tail(log, 8)


def show_dashboard():
    w = 60
    sep = "─" * w

    print()
    print("╔" + "═" * (w-2) + "╗")
    print(f"║{'  SPY QUANT — STATUS DASHBOARD':^{w-2}}║")
    print(f"║{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^{w-2}}║")
    print("╚" + "═" * (w-2) + "╝")

    # ── System ────────────────────────────────────────────────────────────────
    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    print(f"\n  {'SYSTEM':}")
    print(f"  {sep}")
    print(f"  CPU          : {cpu:.1f}%")
    print(f"  Memory       : {mem.percent:.1f}%  ({_fmt_bytes(mem.used)} / {_fmt_bytes(mem.total)})")
    print(f"  Disk         : {disk.percent:.1f}%  ({_fmt_bytes(disk.used)} / {_fmt_bytes(disk.total)})")

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            dev  = torch.cuda.get_device_name(0)
            vmem = torch.cuda.memory_allocated(0) / 1024**2
            print(f"  GPU          : {dev}  ({vmem:.0f} MB allocated)")
        else:
            print("  GPU          : not available (CPU mode)")
    except ImportError:
        print("  GPU          : torch not found")

    # ── Trading mode ──────────────────────────────────────────────────────────
    print(f"\n  {'TRADING'}")
    print(f"  {sep}")
    print(f"  Mode         : {_live_mode()}")
    print(f"  Alpaca URL   : {config.ALPACA_BASE_URL}")

    # ── Model ─────────────────────────────────────────────────────────────────
    mi = _model_info()
    print(f"\n  {'MODEL'}")
    print(f"  {sep}")
    for k, v in mi.items():
        print(f"  {k:<13}: {v}")

    # ── Evaluation results ────────────────────────────────────────────────────
    ev = _eval_summary()
    if ev:
        print(f"\n  {'LAST EVALUATION'}")
        print(f"  {sep}")
        for k, v in ev.items():
            print(f"  {k:<20}: {v}")

    # ── Cache ─────────────────────────────────────────────────────────────────
    ci = _cache_info()
    print(f"\n  {'CACHE'}")
    print(f"  {sep}")
    for k, v in ci.items():
        print(f"  {k:<20}: {v}")

    # ── Recent log ────────────────────────────────────────────────────────────
    print(f"\n  {'RECENT LOG (today)'}")
    print(f"  {sep}")
    for line in _recent_log():
        print(f"  {line[:w-2]}")

    print()


def main():
    parser = argparse.ArgumentParser(description="SPY Quant status dashboard")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                show_dashboard()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
