# ─────────────────────────────────────────────────────────────────────────────
#  SPY Quant — Hardened Production Dockerfile
#  Multi-stage build, CPU-only torch, non-root user
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (largest package, benefits most from layer cache)
# Then install all other deps — both into /usr/local so no COPY tricks needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL description="SPY Quant — hardened production image"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy ALL installed Python packages from builder in one layer
# This avoids the broken selective COPY that missed transitive deps
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy project source
COPY config.py    .
COPY data/        ./data/
COPY models/      ./models/
COPY backtest/    ./backtest/
COPY trading/     ./trading/
COPY dashboard/   ./dashboard/
COPY scripts/     ./scripts/

# Runtime directories — bind-mounted from host in production
RUN mkdir -p data models cache logs backups

# Non-root user
RUN groupadd -r -g 1001 quant && \
    useradd -r -u 1001 -g quant -d /app -s /sbin/nologin quant && \
    chown -R quant:quant /app && \
    # Source read-only, runtime dirs writable
    chmod -R 550 /app && \
    chmod -R 770 /app/data /app/models /app/cache /app/logs /app/backups

USER quant

HEALTHCHECK --interval=60s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "scripts/run_live.py"]
