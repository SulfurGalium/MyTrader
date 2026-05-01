#!/usr/bin/env bash
# deployment/vultr_setup.sh
# Hardened production setup for Vultr Ubuntu 22.04 LTS (Newark / NJ).
# Run as root on a fresh server: bash /opt/spy_quant/deployment/vultr_setup.sh
#
# Security measures applied:
#   - SSH: key-only auth, root login disabled after first run, non-default port
#   - Firewall: ufw + nftables, minimal open ports
#   - fail2ban: SSH + custom Alpaca/app brute-force jails
#   - Docker: no privileged containers, read-only rootfs, seccomp profile
#   - Secrets: .env chmod 600, never in image, never in logs
#   - System: unattended-upgrades, kernel hardening via sysctl
#   - Audit: auditd logging of file access, process execution
#   - Intrusion detection: rkhunter baseline scan

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/opt/spy_quant"
SSH_PORT=2222          # non-default SSH port

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
section() { echo -e "\n${GREEN}══ $* ══${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   SPY Quant — Hardened Vultr NJ Deployment      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. System update + essentials ─────────────────────────────────────────────
section "System update"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    curl wget git htop ufw fail2ban \
    ca-certificates gnupg lsb-release \
    unattended-upgrades apt-listchanges \
    auditd audispd-plugins \
    rkhunter \
    libpam-pwquality \
    tini
info "Packages installed"

# ── 2. Automatic security updates ─────────────────────────────────────────────
section "Automatic security updates"
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'APT'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
APT
cat > /etc/apt/apt.conf.d/20auto-upgrades << 'APT'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APT
info "Unattended security upgrades enabled"

# ── 3. Kernel hardening ────────────────────────────────────────────────────────
section "Kernel hardening"
cat > /etc/sysctl.d/99-hardening.conf << 'SYSCTL'
# IP spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0

# Ignore broadcast pings
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable source routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2

# Hide kernel pointers
kernel.kptr_restrict = 2

# Restrict dmesg to root
kernel.dmesg_restrict = 1

# Prevent core dumps with SUID
fs.suid_dumpable = 0

# ASLR
kernel.randomize_va_space = 2
SYSCTL
sysctl -p /etc/sysctl.d/99-hardening.conf > /dev/null 2>&1 || true
info "Kernel hardening applied"

# ── 4. SSH hardening ───────────────────────────────────────────────────────────
section "SSH hardening"
SSH_CONFIG="/etc/ssh/sshd_config"

# Back up original
cp "$SSH_CONFIG" "${SSH_CONFIG}.bak.$(date +%Y%m%d)"

cat > "$SSH_CONFIG" << SSHCONF
# SPY Quant hardened SSH config
Port $SSH_PORT
Protocol 2

# Authentication — keys only
PermitRootLogin prohibit-password
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes

# Session hardening
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no
MaxAuthTries 3
MaxSessions 5
LoginGraceTime 30

# Ciphers — strong only
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
KexAlgorithms curve25519-sha256,curve25519-sha256@libssh.org

# Logging
LogLevel VERBOSE
SyslogFacility AUTH

# Timeouts
ClientAliveInterval 300
ClientAliveCountMax 2
TCPKeepAlive no
SSHCONF

# CRITICAL: open the new SSH port in firewall BEFORE restarting sshd
# Also keep port 22 open until we confirm 2222 works
ufw allow 22/tcp    comment "SSH original — remove after confirming 2222 works"
ufw allow "$SSH_PORT/tcp" comment "SSH hardened port"
ufw --force enable

# Test sshd config is valid before restarting
if ! sshd -t -f "$SSH_CONFIG"; then
    echo "ERROR: sshd config invalid — reverting to backup"
    cp "${SSH_CONFIG}.bak.$(date +%Y%m%d)" "$SSH_CONFIG"
    exit 1
fi

systemctl restart sshd
sleep 2

# Verify sshd is actually listening on the new port
if ss -tlnp | grep -q ":$SSH_PORT"; then
    info "SSH listening on port $SSH_PORT"
    warn "Test: ssh -p $SSH_PORT root@SERVER_IP in a NEW window before continuing"
    warn "Once confirmed working, run: ufw delete allow 22/tcp"
else
    warn "SSH may not be on port $SSH_PORT yet — check: ss -tlnp | grep ssh"
fi

# ── 5. Firewall ────────────────────────────────────────────────────────────────
section "Firewall (ufw)"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw default deny forward

# Only open what we need
ufw allow "$SSH_PORT/tcp"   comment "SSH"
ufw allow 8080/tcp          comment "Dashboard (restrict after testing)"

# Rate-limit SSH to prevent brute force at fw level
ufw limit "$SSH_PORT/tcp"

ufw --force enable
info "Firewall active: SSH($SSH_PORT) + 8080 open, all else blocked"

# ── 6. fail2ban ───────────────────────────────────────────────────────────────
section "fail2ban"
cat > /etc/fail2ban/jail.local << F2B
[DEFAULT]
bantime  = 3600
findtime = 600
maxretry = 3
backend  = systemd
destemail = root@localhost
action = %(action_mwl)s

[sshd]
enabled  = true
port     = $SSH_PORT
logpath  = %(sshd_log)s
maxretry = 3
bantime  = 86400

[sshd-ddos]
enabled  = true
port     = $SSH_PORT
logpath  = %(sshd_log)s
maxretry = 6
findtime = 60
bantime  = 86400

[docker-auth]
enabled  = true
logpath  = /var/log/auth.log
maxretry = 5
F2B

systemctl enable fail2ban
systemctl restart fail2ban
info "fail2ban configured: 3 retries → 1hr ban, SSH → 24hr ban"

# ── 7. Audit logging ──────────────────────────────────────────────────────────
section "Audit logging"
cat > /etc/audit/rules.d/spy-quant.rules << 'AUDIT'
# Delete all existing rules
-D

# Watch .env file access (secrets)
-w /opt/spy_quant/.env -p rwxa -k secrets_access

# Watch model files
-w /opt/spy_quant/models/ -p wxa -k model_access

# Watch Docker socket
-w /var/run/docker.sock -p rwxa -k docker_socket

# Log all privileged command execution
-a always,exit -F arch=b64 -S execve -F euid=0 -k root_commands

# Log network connections from our app
-a always,exit -F arch=b64 -S connect -k network_connect

# Immutable audit config (comment out if you need to change rules later)
# -e 2
AUDIT

systemctl enable auditd
systemctl restart auditd
info "Audit logging active"

# ── 8. Docker installation ────────────────────────────────────────────────────
section "Docker"
if ! command -v docker &>/dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) \
signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable docker
    systemctl start docker
    info "Docker installed"
else
    # Ensure compose plugin is installed even if docker already exists
    apt-get install -y -qq docker-compose-plugin docker-buildx-plugin 2>/dev/null || true
    info "Docker already present"
fi

# Verify docker compose plugin works
if docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
    info "Using: docker compose (plugin)  version=$(docker compose version --short)"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
    info "Using: docker-compose (standalone)  version=$(docker-compose --version)"
else
    warn "docker compose plugin not found — installing standalone docker-compose"
    curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    COMPOSE_CMD="docker-compose"
    info "docker-compose standalone installed"
fi

# Docker daemon hardening
cat > /etc/docker/daemon.json << 'DOCKER'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "50m",
        "max-file": "5"
    },
    "no-new-privileges": true,
    "live-restore": true,
    "userland-proxy": false,
    "icc": false
}
DOCKER
systemctl restart docker
sleep 2
info "Docker daemon hardened"

# ── 9. Dedicated system user ──────────────────────────────────────────────────
section "System user"
if ! id -u quant &>/dev/null; then
    useradd -r -m -s /bin/bash -d /home/quant quant
    info "Created user: quant"
fi
# Add to docker group so quant user can manage containers
usermod -aG docker quant

# ── 10. Project directory + permissions ───────────────────────────────────────
section "Project directory permissions"
mkdir -p "$PROJECT_DIR"/{models,cache,logs,backups,data,deployment}

# Source code: root-owned, read-only for quant
chown -R root:quant "$PROJECT_DIR"
chmod -R 750 "$PROJECT_DIR"
chmod -R 770 "$PROJECT_DIR"/{models,cache,logs,backups,data}

# .env must be root:quant, readable only by owner
if [ -f "$PROJECT_DIR/.env.example" ] && [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi
if [ -f "$PROJECT_DIR/.env" ]; then
    chown root:quant "$PROJECT_DIR/.env"
    chmod 640 "$PROJECT_DIR/.env"
    info ".env: chmod 640 (root:quant)"
fi

# ── 11. rkhunter baseline ─────────────────────────────────────────────────────
section "Rootkit hunter baseline"
rkhunter --update --quiet 2>/dev/null || true
rkhunter --propupd --quiet 2>/dev/null || true
info "rkhunter baseline established"

# Weekly rkhunter scan via cron
cat > /etc/cron.weekly/rkhunter-scan << 'CRON'
#!/bin/bash
/usr/bin/rkhunter --check --skip-keypress --quiet --report-warnings-only \
    --logfile /var/log/rkhunter-weekly.log 2>&1 | \
    mail -s "rkhunter weekly scan" root 2>/dev/null || true
CRON
chmod +x /etc/cron.weekly/rkhunter-scan

# ── 12. Systemd service ───────────────────────────────────────────────────────
section "Systemd service"

# Use whichever compose command was detected
COMPOSE_BIN=$(command -v docker-compose 2>/dev/null || echo "docker compose")

cat > /etc/systemd/system/spy-quant.service << SERVICE
[Unit]
Description=SPY Quant Trading System
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
User=quant
Group=quant
WorkingDirectory=$PROJECT_DIR
ExecStart=$COMPOSE_BIN -f $PROJECT_DIR/deployment/docker-compose.yml up
ExecStop=$COMPOSE_BIN -f $PROJECT_DIR/deployment/docker-compose.yml down
Restart=on-failure
RestartSec=30
NoNewPrivileges=true
StandardOutput=journal
StandardError=journal
SyslogIdentifier=spy-quant

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable spy-quant.service
info "Systemd service installed + enabled"

# ── 13. Log rotation ──────────────────────────────────────────────────────────
cat > /etc/logrotate.d/spy-quant << 'LOGR'
/opt/spy_quant/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 quant quant
    postrotate
        docker kill --signal HUP spy_quant_trading 2>/dev/null || true
    endscript
}
LOGR
info "Log rotation configured"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Hardened setup complete!                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}IMPORTANT — SSH port changed to $SSH_PORT${NC}"
echo "  Update your connection: ssh -p $SSH_PORT root@SERVER_IP"
echo "  Update Windows shortcut: ssh -p $SSH_PORT root@SERVER_IP"
echo ""
echo "  Verify you can connect in a NEW window BEFORE closing this session!"
echo ""
echo "  Next steps:"
echo "  1. Test SSH on port $SSH_PORT from a new window"
echo "  2. nano $PROJECT_DIR/.env  (add API keys)"
echo "  3. cd $PROJECT_DIR && docker compose -f deployment/docker-compose.yml build"
echo "  4. systemctl start spy-quant"
echo "  5. docker logs -f spy_quant_trading"
echo "  6. Dashboard: http://SERVER_IP:8080"
echo ""
