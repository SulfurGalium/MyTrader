#!/usr/bin/env python3
"""
scripts/run_live.py
Continuous trading loop -- runs on the Vultr server.

  ? Generates a signal every 5 min during market hours
  ? Uses `schedule` library for cron-like timing
  ? Logs all activity to logs/live_YYYY-MM-DD.log
  ? Graceful shutdown on SIGINT / SIGTERM

Usage
-----
  python scripts/run_live.py
  python scripts/run_live.py --dry-run     (ignores LIVE_TRADING_ENABLED)
"""
import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import schedule
from loguru import logger

import config

# -- Logging setup -------------------------------------------------------------
today     = datetime.now().strftime("%Y-%m-%d")
log_file  = config.LOG_DIR / f"live_{today}.log"
logger.add(log_file, rotation="00:00", retention="30 days", level="DEBUG")
logger.add(sys.stdout, level="INFO")


def run_one_cycle(session: "TradingSession", dry_run: bool = False):
    """One 5-min cycle: generate signal -> update regime history -> execute."""
    try:
        from trading.inference import generate_signal

        # generate_signal now returns (signal, vov) -- vol-of-vol for regime gate
        signal_val, current_vov = generate_signal(symbol="SPY")

        # Keep regime history up to date before deciding to trade
        session.update_vov(current_vov)

        result = session.run(signal_val, symbol="SPY", current_vov=current_vov)
        logger.info(f"Cycle complete | {result}")
    except Exception as exc:
        logger.exception(f"Cycle failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Override: disable live order submission regardless of env")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["LIVE_TRADING_ENABLED"] = "false"
        # Reload config after env change
        import importlib
        importlib.reload(config)

    mode = "LIVE" if config.LIVE_TRADING_ENABLED else "DRY-RUN"
    logger.info(f"Starting trading loop -- mode: {mode}")
    logger.info(f"Schedule: every 5 minutes  |  Log: {log_file}")

    # Instantiate session once so _vov_history accumulates across all cycles.
    # Creating a new TradingSession each cycle (the old behaviour) reset the
    # regime history to empty on every tick, making the regime gate useless.
    from trading.live import TradingSession
    session = TradingSession()

    # -- Signal handlers for graceful shutdown ---------------------------------
    _running = [True]

    def _shutdown(signum, frame):
        logger.warning("Shutdown signal received. Stopping after current cycle ?")
        _running[0] = False

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    # -- Schedule every 5 min --------------------------------------------------
    schedule.every(5).minutes.do(run_one_cycle, session=session, dry_run=args.dry_run)

    # Run once immediately on startup
    run_one_cycle(session=session, dry_run=args.dry_run)

    while _running[0]:
        schedule.run_pending()
        time.sleep(10)

    logger.info("Trading loop exited cleanly.")


if __name__ == "__main__":
    main()
