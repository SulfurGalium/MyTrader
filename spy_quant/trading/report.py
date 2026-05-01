"""
trading/report.py
Generates a self-contained HTML performance report from the live trading log.

Reads today's (or a specified date's) log file, parses all cycle records,
and produces:
  • Summary statistics table
  • Equity curve (ASCII + HTML chart)
  • Signal distribution histogram
  • Trade-by-trade log table

Usage (standalone):
  python -m trading.report
  python -m trading.report --date 2024-01-15
  python -m trading.report --out /tmp/report.html
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

_CYCLE_RE = re.compile(
    r"Cycle complete \| (\{.*\})"
)
_SIGNAL_RE = re.compile(
    r"Signal=([+-]?\d+\.\d+).*pred_mean=([+-]?\d+\.\d+).*SNR=(\d+\.\d+)"
)


def parse_log(log_path: Path) -> list[dict]:
    records = []
    if not log_path.exists():
        logger.warning(f"Log not found: {log_path}")
        return records

    with open(log_path) as f:
        lines = f.readlines()

    current: dict[str, Any] = {}
    for line in lines:
        # Signal line
        m = _SIGNAL_RE.search(line)
        if m:
            current = {
                "signal":    float(m.group(1)),
                "pred_mean": float(m.group(2)),
                "snr":       float(m.group(3)),
                "timestamp": _extract_ts(line),
            }

        # Cycle completion line
        m2 = _CYCLE_RE.search(line)
        if m2 and current:
            try:
                cycle = json.loads(m2.group(1).replace("'", '"'))
                current.update(cycle)
                records.append(current)
                current = {}
            except Exception:
                pass

    return records


def _extract_ts(line: str) -> str:
    # Loguru format: "2024-01-15 09:35:00.123 | INFO | ..."
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return m.group(1) if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_session_stats(records: list[dict]) -> dict:
    if not records:
        return {}

    signals  = [r.get("signal", 0.0) for r in records]
    actions  = [r.get("action", "none") for r in records]
    snrs     = [r.get("snr", 0.0) for r in records]

    submitted = [r for r in records if r.get("action") == "submitted"]
    dry_runs  = [r for r in records if r.get("action") == "dry_run"]
    skipped   = len(records) - len(submitted) - len(dry_runs)

    long_sigs  = [s for s in signals if s > 0.15]
    short_sigs = [s for s in signals if s < -0.15]

    return {
        "total_cycles":   len(records),
        "submitted":      len(submitted),
        "dry_run":        len(dry_runs),
        "skipped":        skipped,
        "long_signals":   len(long_sigs),
        "short_signals":  len(short_sigs),
        "mean_signal":    float(np.mean(signals)) if signals else 0.0,
        "mean_abs_signal":float(np.mean(np.abs(signals))) if signals else 0.0,
        "mean_snr":       float(np.mean(snrs)) if snrs else 0.0,
        "max_snr":        float(np.max(snrs)) if snrs else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML report renderer
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPY Quant — Trading Report {date}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; padding: 2rem; }}
  h1 {{ color: var(--accent); font-size: 1.4rem; margin-bottom: 0.3rem; }}
  h2 {{ color: var(--muted); font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; margin: 1.8rem 0 0.8rem; }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; }}
  .card .label {{ color: var(--muted); font-size: 0.75rem; margin-bottom: 0.4rem; }}
  .card .value {{ font-size: 1.5rem; font-weight: bold; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th {{ background: var(--surface); color: var(--muted); padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
  td {{ padding: 0.4rem 0.8rem; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: var(--surface); }}
  .bar-chart {{ display: flex; align-items: flex-end; gap: 2px; height: 80px; margin: 1rem 0; }}
  .bar {{ background: var(--accent); min-width: 4px; border-radius: 2px 2px 0 0; opacity: 0.8; transition: opacity 0.2s; }}
  .bar:hover {{ opacity: 1; }}
  .bar.neg {{ background: var(--red); }}
  .signal-line {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }}
  .signal-bar {{ height: 10px; border-radius: 2px; background: var(--accent); }}
  .signal-bar.neg {{ background: var(--red); }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.72rem; font-weight: bold; }}
  .badge-green {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }}
  .badge-red {{ background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid rgba(248,81,73,0.3); }}
  .badge-yellow {{ background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid rgba(210,153,34,0.3); }}
  .badge-blue {{ background: rgba(88,166,255,0.15); color: var(--accent); border: 1px solid rgba(88,166,255,0.3); }}
</style>
</head>
<body>
<h1>⚡ SPY Quant — Session Report</h1>
<p class="meta">Generated: {generated}  ·  Session date: {date}  ·  Mode: {mode}</p>

<h2>Session Summary</h2>
<div class="grid">
  <div class="card"><div class="label">TOTAL CYCLES</div><div class="value">{total_cycles}</div></div>
  <div class="card"><div class="label">ORDERS SUBMITTED</div><div class="value {submitted_color}">{submitted}</div></div>
  <div class="card"><div class="label">SKIPPED</div><div class="value yellow">{skipped}</div></div>
  <div class="card"><div class="label">LONG SIGNALS</div><div class="value green">{long_signals}</div></div>
  <div class="card"><div class="label">SHORT SIGNALS</div><div class="value red">{short_signals}</div></div>
  <div class="card"><div class="label">MEAN |SIGNAL|</div><div class="value">{mean_abs_signal:.4f}</div></div>
  <div class="card"><div class="label">MEAN SNR</div><div class="value">{mean_snr:.2f}</div></div>
  <div class="card"><div class="label">PEAK SNR</div><div class="value">{max_snr:.2f}</div></div>
</div>

<h2>Signal Distribution</h2>
{signal_chart}

<h2>Trade Log</h2>
{trade_table}

</body>
</html>"""


def _signal_bars_html(records: list[dict]) -> str:
    if not records:
        return "<p style='color:var(--muted)'>No records.</p>"

    signals = [r.get("signal", 0.0) for r in records[-80:]]   # last 80 cycles
    max_abs = max(abs(s) for s in signals) or 1.0

    bars = []
    for s in signals:
        pct   = int(abs(s) / max_abs * 70)
        cls   = "bar neg" if s < 0 else "bar"
        title = f"{s:+.4f}"
        bars.append(f'<div class="{cls}" style="height:{max(pct,2)}px" title="{title}"></div>')

    return f'<div class="bar-chart">{"".join(bars)}</div>'


def _trade_table_html(records: list[dict]) -> str:
    if not records:
        return "<p style='color:var(--muted)'>No trades recorded.</p>"

    rows = []
    for r in records[-50:]:    # last 50
        action = r.get("action", "none")
        sig    = r.get("signal", 0.0)
        snr    = r.get("snr", 0.0)
        ts     = r.get("timestamp", "")
        reason = r.get("reason", "")
        qty    = r.get("qty", "—")

        if action == "submitted":
            badge = '<span class="badge badge-green">LIVE</span>'
        elif action == "dry_run":
            badge = '<span class="badge badge-blue">DRY-RUN</span>'
        elif action == "none":
            badge = '<span class="badge badge-yellow">SKIP</span>'
        else:
            badge = f'<span class="badge badge-yellow">{action}</span>'

        sig_color = "green" if sig > 0.15 else ("red" if sig < -0.15 else "")
        rows.append(
            f"<tr>"
            f"<td>{ts}</td>"
            f"<td>{badge}</td>"
            f'<td class="{sig_color}">{sig:+.4f}</td>'
            f"<td>{snr:.2f}</td>"
            f"<td>{qty}</td>"
            f"<td style='color:var(--muted)'>{reason}</td>"
            f"</tr>"
        )

    header = (
        "<thead><tr>"
        "<th>Time</th><th>Action</th><th>Signal</th>"
        "<th>SNR</th><th>Qty</th><th>Reason</th>"
        "</tr></thead>"
    )
    return f"<table>{header}<tbody>{''.join(reversed(rows))}</tbody></table>"


def build_report(date_str: str | None = None) -> tuple[str, dict]:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    log_path = config.LOG_DIR / f"live_{date_str}.log"

    records = parse_log(log_path)
    stats   = compute_session_stats(records)

    if not stats:
        stats = {k: 0 for k in (
            "total_cycles", "submitted", "dry_run", "skipped",
            "long_signals", "short_signals", "mean_signal",
            "mean_abs_signal", "mean_snr", "max_snr",
        )}

    mode = "🔴 LIVE" if config.LIVE_TRADING_ENABLED else "🟡 PAPER / DRY-RUN"
    sub_color = "green" if stats.get("submitted", 0) > 0 else "muted"

    html = _HTML_TEMPLATE.format(
        date=date_str,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mode=mode,
        submitted_color=sub_color,
        signal_chart=_signal_bars_html(records),
        trade_table=_trade_table_html(records),
        **{k: stats.get(k, 0) for k in stats},
    )
    return html, stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate session performance report")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--out",  type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    html, stats = build_report(args.date)

    out_path = Path(args.out) if args.out else (
        config.LOG_DIR / f"report_{args.date or datetime.now().strftime('%Y-%m-%d')}.html"
    )
    out_path.write_text(html, encoding="utf-8")
    logger.success(f"Report written → {out_path}")

    print(json.dumps(stats, indent=2))
