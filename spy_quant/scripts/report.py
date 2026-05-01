#!/usr/bin/env python3
"""
scripts/report.py
Generate the HTML session report and print summary stats.

Usage
─────
  python scripts/report.py                    # today's log
  python scripts/report.py --date 2024-03-01
  python scripts/report.py --out /tmp/my_report.html
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from trading.report import build_report
import config


def main():
    parser = argparse.ArgumentParser(description="Generate session HTML report")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--out",  type=str, default=None)
    args = parser.parse_args()

    html, stats = build_report(args.date)

    from datetime import datetime
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    out_path = Path(args.out) if args.out else config.LOG_DIR / f"report_{date_str}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.success(f"Report → {out_path}")

    print(f"\n  Cycles   : {stats.get('total_cycles', 0)}")
    print(f"  Submitted: {stats.get('submitted', 0)}")
    print(f"  Longs    : {stats.get('long_signals', 0)}")
    print(f"  Shorts   : {stats.get('short_signals', 0)}")
    print(f"  Mean SNR : {stats.get('mean_snr', 0):.3f}")


if __name__ == "__main__":
    main()
