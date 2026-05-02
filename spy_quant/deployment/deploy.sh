#!/usr/bin/env bash
# deployment/deploy.sh  — fixed version
# Resolves three bugs from the original:
#   1. Double-nesting: git clone <repo> produced MyTrader/spy_quant/; running
#      rsync from SCRIPT_DIR/../ (= MyTrader/) copied spy_quant/ as a subdirectory
#      into /opt/spy_quant/spy_quant/ instead of flattening the contents.
#   2. Missing .env.example: the repo had no .env.example file, so
#      "cp .env.example .env" silently failed and left no .env at all.
#   3. systemctl permission: README bare-metal steps omitted sudo before every
#      systemctl command; non-root users get "Failed to connect to bus: No such
#      file or directory" or permission denied.
#
# Usage (must be run as root or via sudo):
#   git clone <your-repo> /tmp/mytrader
#   cd /tmp/mytrader/spy_quant        # enter the inner spy_quant/ directory
#   sudo bash deployment/deploy.sh

set -euo pipefail

# ── Verify we are root ────────────────────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: deploy.sh must be run as root."
    echo "       Re-run with: sudo bash deployment/deploy.sh"
    exit 1
fi

PROJECT_DIR="/opt/spy_quant"
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_NAME="spy_quant"
PYTHON_BIN="python3.11"

echo "═══════════════════════════════════════════════════════"
echo "  SPY Quant — Vultr Server Deployment (fixed)"
echo "═══════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip build-essential git curl wget \
    htop tmux logrotate \
    libssl-dev libffi-dev cython3

# ── 2. Create quant user ──────────────────────────────────────────────────────
if ! id -u quant &>/dev/null; then
    useradd -r -m -s /bin/bash -d /home/quant quant
    echo "Created user: quant"
fi

# ── 3. Locate source directory (fix for double-nesting) ───────────────────────
# SCRIPT_DIR = .../spy_quant/deployment/
# SOURCE_DIR = .../spy_quant/   (the actual project root with config.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$SOURCE_DIR/config.py" ]; then
    echo "ERROR: Cannot find config.py in $SOURCE_DIR"
    echo "       Make sure you cd into spy_quant/ before running this script:"
    echo "       cd /tmp/mytrader/spy_quant && sudo bash deployment/deploy.sh"
    exit 1
fi

echo "Source : $SOURCE_DIR"
echo "Target : $PROJECT_DIR"

# ── 4. Sync files (trailing slash on SOURCE_DIR copies contents, not folder) ──
mkdir -p "$PROJECT_DIR"
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='venv/' \
    --exclude='.env' \
    "$SOURCE_DIR/" "$PROJECT_DIR/"

chown -R quant:quant "$PROJECT_DIR"
echo "Files synced."

# ── 5. Generate .env.example if missing, then create .env ────────────────────
ENV_EXAMPLE="$PROJECT_DIR/.env.example"
ENV_FILE="$PROJECT_DIR/.env"

if [ ! -f "$ENV_EXAMPLE" ]; then
    echo "No .env.example in repo — generating one now..."
    cat > "$ENV_EXAMPLE" << 'ENVEOF'
# ── Alpaca credentials ─────────────────────────────────────────────────────
# Paper trading (default): https://paper-api.alpaca.markets
# Live trading:            https://api.alpaca.markets
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# ── FRED (macro data) ─────────────────────────────────────────────────────
# Free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# ── Safety gate ────────────────────────────────────────────────────────────
# Set "true" ONLY after thorough paper-trading validation
LIVE_TRADING_ENABLED=false

# ── Model / training ──────────────────────────────────────────────────────
DEVICE=cpu
BATCH_SIZE=64
EPOCHS=50
LEARNING_RATE=0.0003

# ── Risk parameters ───────────────────────────────────────────────────────
MAX_POSITION_RISK=0.01
STOP_LOSS_PCT=0.005
TAKE_PROFIT_PCT=0.015

# ── Optional: override runtime directories ────────────────────────────────
DATA_DIR=
CACHE_DIR=
MODEL_DIR=
LOG_DIR=
ENVEOF
    chown quant:quant "$ENV_EXAMPLE"
    chmod 644 "$ENV_EXAMPLE"
fi

if [ ! -f "$ENV_FILE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    chown quant:quant "$ENV_FILE"
    echo ""
    echo "⚠  .env created from template. Edit it before starting the service:"
    echo "   sudo nano $ENV_FILE"
    echo ""
else
    echo ".env already exists — leaving it untouched (credentials preserved)."
fi

# ── 6. Virtualenv + dependencies ──────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    sudo -u quant $PYTHON_BIN -m venv "$VENV_DIR"
    echo "Virtualenv created."
fi

sudo -u quant "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q
sudo -u quant "$VENV_DIR/bin/pip" install --no-build-isolation \
    -r "$PROJECT_DIR/requirements.txt"
echo "Dependencies installed."

# ── 7. Systemd services (fix: these require root — document that clearly) ─────
cp "$PROJECT_DIR/deployment/spy_quant.service" \
   "/etc/systemd/system/${SERVICE_NAME}.service"
cp "$PROJECT_DIR/deployment/spy_quant_dashboard.service" \
   "/etc/systemd/system/${SERVICE_NAME}_dashboard.service"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl enable "${SERVICE_NAME}_dashboard"
echo "Systemd services installed and enabled."

# ── 8. Runtime directories ────────────────────────────────────────────────────
for dir in logs cache models backups; do
    mkdir -p "$PROJECT_DIR/$dir"
    chown quant:quant "$PROJECT_DIR/$dir"
done

# ── 9. Log rotation ──────────────────────────────────────────────────────────
cat > /etc/logrotate.d/spy_quant << 'EOF'
/opt/spy_quant/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 quant quant
}
EOF

# ── 10. Backup cron ───────────────────────────────────────────────────────────
CRON_LINE="0 2 * * * quant tar -czf /opt/spy_quant/backups/models_\$(date +\%Y\%m\%d).tar.gz /opt/spy_quant/models/ 2>/dev/null || true"
if ! grep -q "spy_quant" /etc/cron.d/spy_quant 2>/dev/null; then
    echo "$CRON_LINE" > /etc/cron.d/spy_quant
    chmod 644 /etc/cron.d/spy_quant
    echo "Backup cron installed (daily 02:00)."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  Next steps (all commands need sudo):"
echo ""
echo "  1. Add your API keys:"
echo "     sudo nano $PROJECT_DIR/.env"
echo ""
echo "  2. Train the model:"
echo "     sudo -u quant $VENV_DIR/bin/python scripts/train.py --alpaca"
echo ""
echo "  3. Start services:"
echo "     sudo systemctl start $SERVICE_NAME"
echo "     sudo systemctl start ${SERVICE_NAME}_dashboard"
echo ""
echo "  4. Monitor logs:"
echo "     sudo journalctl -u $SERVICE_NAME -f"
echo "     sudo journalctl -u ${SERVICE_NAME}_dashboard -f"
echo "═══════════════════════════════════════════════════════"
