#!/usr/bin/env bash
# deployment/deploy.sh
# One-shot server setup + deployment script for a fresh Vultr Ubuntu 22.04 VPS.
# NOTE: This script is for the Vultr Linux server only.
#       On Windows, use setup_windows.ps1 instead.
#
# Run as root:
#   bash deploy.sh
#
# What it does:
#   1. System update + Python 3.11 + CUDA toolkit (optional)
#   2. Creates a dedicated `quant` user
#   3. Clones / copies the project to /opt/spy_quant
#   4. Creates a Python virtualenv and installs dependencies
#   5. Installs and enables the systemd service
#   6. Sets up daily log rotation + model backup cron

set -euo pipefail

PROJECT_DIR="/opt/spy_quant"
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_NAME="spy_quant"
PYTHON_BIN="python3.11"

echo "═══════════════════════════════════════════════════════"
echo "  SPY Quant — Vultr Server Deployment"
echo "═══════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip build-essential git curl wget \
    htop tmux logrotate

# ── 2. Create quant user ──────────────────────────────────────────────────────
if ! id -u quant &>/dev/null; then
    useradd -r -m -s /bin/bash -d /home/quant quant
    echo "Created user: quant"
fi

# ── 3. Set up project directory ───────────────────────────────────────────────
mkdir -p "$PROJECT_DIR"
chown -R quant:quant "$PROJECT_DIR"

# Copy project files (assumes this script is run from the project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='venv' --exclude='.env' \
    "$SCRIPT_DIR/../" "$PROJECT_DIR/"

chown -R quant:quant "$PROJECT_DIR"

# ── 4. Create virtualenv + install deps ───────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    sudo -u quant $PYTHON_BIN -m venv "$VENV_DIR"
fi

sudo -u quant "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u quant "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "Dependencies installed."

# ── 5. Environment file ───────────────────────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
    chown quant:quant "$PROJECT_DIR/.env"
    echo ""
    echo "⚠  .env created from template. Edit it before starting the service:"
    echo "   nano $PROJECT_DIR/.env"
    echo ""
fi

# ── 6. Systemd service ────────────────────────────────────────────────────────
cp "$PROJECT_DIR/deployment/spy_quant.service" \
   "/etc/systemd/system/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "Systemd service installed: $SERVICE_NAME"
echo "  Start : systemctl start $SERVICE_NAME"
echo "  Status: systemctl status $SERVICE_NAME"
echo "  Logs  : journalctl -u $SERVICE_NAME -f"

# ── 7. Log rotation ───────────────────────────────────────────────────────────
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

# ── 8. Model backup cron (daily at 02:00) ────────────────────────────────────
CRON_LINE="0 2 * * * quant tar -czf /opt/spy_quant/backups/models_\$(date +\%Y\%m\%d).tar.gz /opt/spy_quant/models/ 2>/dev/null"
mkdir -p /opt/spy_quant/backups
chown quant:quant /opt/spy_quant/backups

# Add cron if not present
if ! grep -q "spy_quant" /etc/cron.d/spy_quant 2>/dev/null; then
    echo "$CRON_LINE" > /etc/cron.d/spy_quant
    chmod 644 /etc/cron.d/spy_quant
    echo "Model backup cron installed (daily 02:00)."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit /opt/spy_quant/.env  (add API keys)"
echo "  2. Train the model:  sudo -u quant $VENV_DIR/bin/python scripts/train.py --alpaca"
echo "  3. Start the service: systemctl start $SERVICE_NAME"
echo "═══════════════════════════════════════════════════════"
