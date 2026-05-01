"""
dashboard/server.py
Lightweight FastAPI server providing a live monitoring dashboard
for the SPY Quant system running on Vultr.

Endpoints
─────────
  GET /              → HTML dashboard (auto-refreshes every 30s)
  GET /api/status    → JSON system status
  GET /api/metrics   → JSON latest eval + WFO metrics
  GET /api/log       → JSON last N log lines
  GET /api/signal    → JSON most recent signal record
  GET /health        → plain 200 OK for load balancer probes

Run:
  uvicorn dashboard.server:app --host 0.0.0.0 --port 8080
  or via: python -m dashboard.server
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

import re


app = FastAPI(title="SPY Quant Dashboard", docs_url=None, redoc_url=None) if _FASTAPI_OK else None


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _tail_log(n: int = 50) -> list[str]:
    today    = datetime.now().strftime("%Y-%m-%d")
    log_path = config.LOG_DIR / f"live_{today}.log"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def _system_stats() -> dict:
    stats: dict[str, Any] = {
        "platform": platform.system(),
        "python":   platform.python_version(),
        "uptime":   _uptime(),
    }
    if _PSUTIL_OK:
        mem  = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        stats.update({
            "cpu_pct":    psutil.cpu_percent(interval=0.5),
            "mem_pct":    mem.percent,
            "mem_used_gb":round(mem.used / 1024**3, 2),
            "mem_total_gb":round(mem.total / 1024**3, 2),
            "disk_pct":   disk.percent,
            "disk_free_gb":round(disk.free / 1024**3, 1),
        })
    return stats


def _uptime() -> str:
    if not _PSUTIL_OK:
        return "N/A"
    try:
        boot = psutil.boot_time()
        secs = int(time.time() - boot)
        d, r = divmod(secs, 86400)
        h, r = divmod(r, 3600)
        m, _ = divmod(r, 60)
        return f"{d}d {h:02d}h {m:02d}m"
    except Exception:
        return "N/A"


def _model_info() -> dict:
    ckpt = config.MODEL_DIR / "diffusion_latest.pt"
    if not ckpt.exists():
        return {"exists": False}
    stat = ckpt.stat()
    return {
        "exists":   True,
        "size_mb":  round(stat.st_size / 1024**2, 1),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _latest_signal() -> dict:
    lines = _tail_log(200)
    signal_re = re.compile(r"Signal=([+-]?\d+\.\d+).*pred_mean=([+-]?\d+\.\d+).*SNR=(\d+\.\d+)")
    ts_re     = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    for line in reversed(lines):
        m = signal_re.search(line)
        if m:
            ts = ts_re.match(line)
            return {
                "signal":    float(m.group(1)),
                "pred_mean": float(m.group(2)),
                "snr":       float(m.group(3)),
                "timestamp": ts.group(1) if ts else "",
            }
    return {}


def _eval_metrics() -> dict:
    ev  = _read_json(config.MODEL_DIR / "eval_results.json")
    wfo = _read_json(config.MODEL_DIR / "wfo_results.json")
    mc  = ev.get("monte_carlo", {})
    return {
        "wf_mean_sharpe":   ev.get("wf_mean_sharpe"),
        "wf_passing_folds": ev.get("wf_passing_folds"),
        "mc_mean_return":   mc.get("mean_return"),
        "mc_prob_positive": mc.get("prob_positive"),
        "mc_var95":         mc.get("var_95"),
        "wfo_mean_sharpe":  wfo.get("summary", {}).get("mean_sharpe"),
        "wfo_passing":      wfo.get("summary", {}).get("passing_folds"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────────────────────────────────────

if _FASTAPI_OK:
    @app.get("/health")
    def health():
        return {"status": "ok", "ts": datetime.utcnow().isoformat()}

    @app.get("/api/status")
    def api_status():
        return JSONResponse({
            "system":  _system_stats(),
            "model":   _model_info(),
            "trading": {
                "live_enabled": config.LIVE_TRADING_ENABLED,
                "alpaca_url":   config.ALPACA_BASE_URL,
            },
            "cache": {
                "ust10y_age_h": _ust10y_age(),
            },
        })

    @app.get("/api/metrics")
    def api_metrics():
        return JSONResponse(_eval_metrics())

    @app.get("/api/log")
    def api_log(n: int = 50):
        return JSONResponse({"lines": _tail_log(n)})

    @app.get("/api/signal")
    def api_signal():
        return JSONResponse(_latest_signal())

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return HTMLResponse(_render_dashboard())


def _ust10y_age() -> float | None:
    meta = config.CACHE_DIR / "ust10y_meta.json"
    if not meta.exists():
        return None
    try:
        d = json.loads(meta.read_text())
        return round((time.time() - d["fetched_at"]) / 3600, 1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Server-side rendered dashboard HTML
# ─────────────────────────────────────────────────────────────────────────────

def _render_dashboard() -> str:
    sys_s  = _system_stats()
    model  = _model_info()
    sig    = _latest_signal()
    ev     = _eval_metrics()
    logs   = _tail_log(30)
    mode   = "🔴 LIVE" if config.LIVE_TRADING_ENABLED else "🟡 PAPER"
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sig_val   = sig.get("signal", 0.0)
    sig_color = "#3fb950" if sig_val > 0.15 else ("#f85149" if sig_val < -0.15 else "#8b949e")
    sig_dir   = "LONG ▲" if sig_val > 0.15 else ("SHORT ▼" if sig_val < -0.15 else "FLAT —")

    def _pct(v):
        return f"{v:.1%}" if v is not None else "—"

    def _f(v, dec=3):
        return f"{v:.{dec}f}" if v is not None else "—"

    log_rows = "".join(
        f'<div class="log-line {"err" if "ERROR" in l or "FAIL" in l else ("warn" if "WARNING" in l else "")}">{l}</div>'
        for l in reversed(logs)
    )

    model_badge = (
        f'<span class="badge green">{model["modified"]}</span>'
        if model.get("exists") else
        '<span class="badge red">NOT FOUND</span>'
    )

    sharpe_val = ev.get("wf_mean_sharpe")
    sharpe_color = "#3fb950" if (sharpe_val or 0) > 0.5 else ("#f85149" if (sharpe_val or 0) < 0 else "#d29922")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>SPY Quant Dashboard</title>
  <style>
    :root {{
      --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;
      --text:#c9d1d9;--muted:#8b949e;--accent:#58a6ff;
      --green:#3fb950;--red:#f85149;--yellow:#d29922;
    }}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:var(--bg);color:var(--text);font-family:'JetBrains Mono','Fira Code',monospace;font-size:12px;min-height:100vh}}
    header{{background:var(--surface);border-bottom:1px solid var(--border);padding:1rem 1.5rem;display:flex;justify-content:space-between;align-items:center}}
    .logo{{font-size:1rem;font-weight:700;color:var(--accent);letter-spacing:0.05em}}
    .meta{{color:var(--muted);font-size:0.75rem}}
    main{{padding:1.5rem;display:grid;grid-template-columns:repeat(12,1fr);gap:1rem}}
    .col-4{{grid-column:span 4}}.col-6{{grid-column:span 6}}.col-8{{grid-column:span 8}}.col-12{{grid-column:span 12}}
    @media(max-width:900px){{.col-4,.col-6,.col-8{{grid-column:span 12}}}}
    .panel{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem}}
    .panel-title{{color:var(--muted);font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem}}
    .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem}}
    .stat{{background:var(--surface2);border-radius:4px;padding:0.6rem 0.8rem}}
    .stat .lbl{{color:var(--muted);font-size:0.68rem;margin-bottom:0.2rem}}
    .stat .val{{font-size:1.1rem;font-weight:700}}
    .big-signal{{text-align:center;padding:1rem 0}}
    .big-signal .val{{font-size:2.8rem;font-weight:900;color:{sig_color}}}
    .big-signal .dir{{font-size:0.9rem;color:{sig_color};margin-top:0.2rem}}
    .big-signal .ts{{color:var(--muted);font-size:0.72rem;margin-top:0.5rem}}
    .gauge-bar{{height:8px;background:var(--surface2);border-radius:4px;margin-top:0.3rem;overflow:hidden}}
    .gauge-fill{{height:100%;border-radius:4px;transition:width 0.4s}}
    .row-stat{{display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #21262d}}
    .row-stat:last-child{{border-bottom:none}}
    .row-stat .k{{color:var(--muted)}}
    .badge{{display:inline-block;padding:0.15rem 0.5rem;border-radius:3px;font-size:0.7rem;font-weight:700}}
    .badge.green{{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}}
    .badge.red{{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}}
    .badge.yellow{{background:rgba(210,153,34,.15);color:var(--yellow);border:1px solid rgba(210,153,34,.3)}}
    .badge.blue{{background:rgba(88,166,255,.15);color:var(--accent);border:1px solid rgba(88,166,255,.3)}}
    .log-box{{height:280px;overflow-y:auto;font-size:0.7rem;line-height:1.6;background:var(--surface2);border-radius:4px;padding:0.6rem}}
    .log-line{{white-space:pre-wrap;word-break:break-all;padding:1px 0}}
    .log-line.err{{color:var(--red)}}
    .log-line.warn{{color:var(--yellow)}}
    .refresh-note{{color:var(--muted);font-size:0.68rem;text-align:right;margin-top:0.4rem}}
  </style>
</head>
<body>
<header>
  <div class="logo">⚡ SPY QUANT</div>
  <div class="meta">{mode} &nbsp;·&nbsp; {now} &nbsp;·&nbsp; <span style="color:var(--accent)">auto-refresh 30s</span></div>
</header>
<main>

  <!-- Signal card -->
  <div class="panel col-4">
    <div class="panel-title">Latest Signal</div>
    <div class="big-signal">
      <div class="val">{sig_val:+.4f}</div>
      <div class="dir">{sig_dir}</div>
      <div class="ts">{sig.get("timestamp","—")} &nbsp;·&nbsp; SNR {_f(sig.get("snr"),2)}</div>
    </div>
  </div>

  <!-- System stats -->
  <div class="panel col-4">
    <div class="panel-title">System</div>
    <div class="row-stat"><span class="k">CPU</span><span>{sys_s.get("cpu_pct","—")}%</span></div>
    <div class="gauge-bar"><div class="gauge-fill" style="width:{sys_s.get('cpu_pct',0)}%;background:var(--accent)"></div></div>
    <div class="row-stat" style="margin-top:0.5rem"><span class="k">Memory</span><span>{sys_s.get("mem_pct","—")}%</span></div>
    <div class="gauge-bar"><div class="gauge-fill" style="width:{sys_s.get('mem_pct',0)}%;background:var(--yellow)"></div></div>
    <div class="row-stat" style="margin-top:0.5rem"><span class="k">Disk</span><span>{sys_s.get("disk_pct","—")}%</span></div>
    <div class="gauge-bar"><div class="gauge-fill" style="width:{sys_s.get('disk_pct',0)}%;background:var(--green)"></div></div>
    <div class="row-stat" style="margin-top:0.7rem"><span class="k">Uptime</span><span>{sys_s.get("uptime","—")}</span></div>
    <div class="row-stat"><span class="k">Python</span><span>{sys_s.get("python","—")}</span></div>
  </div>

  <!-- Model info -->
  <div class="panel col-4">
    <div class="panel-title">Model</div>
    <div class="row-stat"><span class="k">Checkpoint</span>{model_badge}</div>
    <div class="row-stat"><span class="k">Size</span><span>{model.get("size_mb","—")} MB</span></div>
    <div class="row-stat"><span class="k">WF Sharpe</span><span style="color:{sharpe_color}">{_f(sharpe_val)}</span></div>
    <div class="row-stat"><span class="k">WF Passing</span><span>{ev.get("wf_passing_folds","—")} / {config.WALK_FORWARD_SPLITS}</span></div>
    <div class="row-stat"><span class="k">MC E[R]</span><span>{_pct(ev.get("mc_mean_return"))}</span></div>
    <div class="row-stat"><span class="k">MC P(+)</span><span>{_pct(ev.get("mc_prob_positive"))}</span></div>
    <div class="row-stat"><span class="k">MC VaR95</span><span style="color:var(--red)">{_pct(ev.get("mc_var95"))}</span></div>
    <div class="row-stat"><span class="k">UST10Y cache</span><span>{_ust10y_age() or "—"} h old</span></div>
  </div>

  <!-- Trading config -->
  <div class="panel col-4">
    <div class="panel-title">Trading Config</div>
    <div class="row-stat"><span class="k">Mode</span>
      <span class="badge {'green' if config.LIVE_TRADING_ENABLED else 'yellow'}">
        {'LIVE' if config.LIVE_TRADING_ENABLED else 'PAPER'}
      </span>
    </div>
    <div class="row-stat"><span class="k">Endpoint</span><span style="color:var(--muted);font-size:0.68rem">{config.ALPACA_BASE_URL.replace("https://","")}</span></div>
    <div class="row-stat"><span class="k">Max Risk</span><span>{config.MAX_POSITION_RISK:.1%}</span></div>
    <div class="row-stat"><span class="k">Stop Loss</span><span style="color:var(--red)">{config.STOP_LOSS_PCT:.2%}</span></div>
    <div class="row-stat"><span class="k">Take Profit</span><span style="color:var(--green)">{config.TAKE_PROFIT_PCT:.2%}</span></div>
    <div class="row-stat"><span class="k">Seq Len</span><span>{config.SEQ_LEN} bars</span></div>
    <div class="row-stat"><span class="k">Coarse Factor</span><span>{config.COARSE_FACTOR}×</span></div>
    <div class="row-stat"><span class="k">Diffusion Steps</span><span>{config.DIFFUSION_STEPS}</span></div>
  </div>

  <!-- Live log -->
  <div class="panel col-8">
    <div class="panel-title">Live Log (last 30 lines)</div>
    <div class="log-box">{log_rows or '<span style="color:var(--muted)">No log entries today.</span>'}</div>
    <div class="refresh-note">Auto-refreshes every 30 seconds</div>
  </div>

</main>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _FASTAPI_OK:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)
    import uvicorn
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", 8080))
    print(f"Dashboard → http://{host}:{port}")
    uvicorn.run("dashboard.server:app", host=host, port=port, reload=False)
