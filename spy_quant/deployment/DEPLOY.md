# SPY Quant — Secure Deployment to Vultr NJ

Complete step-by-step guide: Windows → Vultr Newark server.

---

## Step 1 — Create the Vultr server

1. Go to https://my.vultr.com → **Deploy New Server**
2. Settings:
   - **Type**: Cloud Compute — Shared CPU
   - **Region**: New Jersey (Newark) — ~8ms latency to NYSE
   - **Image**: Ubuntu 22.04 LTS x64
   - **Plan**: 2 vCPU / 4 GB RAM / 80 GB SSD (~$24/month)
   - **SSH Key**: paste your public key (see below)
   - **Hostname**: `spy-quant-nj`
3. Click **Deploy Now**, wait ~60 seconds, copy the IP.

### Generate SSH key (Windows PowerShell)
```powershell
ssh-keygen -t ed25519 -C "spy-quant-vultr"
# Press Enter for all prompts

# Print your public key — paste this into Vultr SSH key field
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

---

## Step 2 — Copy project files to server

Run in PowerShell. Replace `YOUR_SERVER_IP` with your actual IP.

```powershell
# Set these once at the top
$IP  = "YOUR_SERVER_IP"
$SRC = "C:\Users\gangu\OneDrive\Desktop\Lancelot1"

# Create directories on server
ssh "root@$IP" "mkdir -p /opt/spy_quant/{models,cache,logs,backups,data,deployment}"

# Source code files
scp "$SRC\config.py"           "root@${IP}:/opt/spy_quant/"
scp "$SRC\requirements.txt"    "root@${IP}:/opt/spy_quant/"
scp "$SRC\.env.example"        "root@${IP}:/opt/spy_quant/"

# Source directories (use -r for directories)
scp -r "$SRC\data"             "root@${IP}:/opt/spy_quant/"
scp -r "$SRC\backtest"         "root@${IP}:/opt/spy_quant/"
scp -r "$SRC\trading"          "root@${IP}:/opt/spy_quant/"
scp -r "$SRC\dashboard"        "root@${IP}:/opt/spy_quant/"
scp -r "$SRC\scripts"          "root@${IP}:/opt/spy_quant/"
scp -r "$SRC\deployment"       "root@${IP}:/opt/spy_quant/"

# Model Python files only (not the .pt checkpoint — copy that separately)
scp "$SRC\models\__init__.py"  "root@${IP}:/opt/spy_quant/models/"
scp "$SRC\models\diffusion.py" "root@${IP}:/opt/spy_quant/models/"
scp "$SRC\models\trainer.py"   "root@${IP}:/opt/spy_quant/models/"

# Trained model checkpoint — use the best trading score checkpoint
scp "$SRC\models\diffusion_best_trading.pt" "root@${IP}:/opt/spy_quant/models/diffusion_latest.pt"
scp "$SRC\models\feature_scaler.pkl"        "root@${IP}:/opt/spy_quant/models/"
scp "$SRC\models\training_history.json"     "root@${IP}:/opt/spy_quant/models/"

# Data cache — saves re-downloading on the server
scp "$SRC\cache\ohlcv_SPY_5min.parquet"     "root@${IP}:/opt/spy_quant/cache/"
scp "$SRC\cache\ohlcv_SPY_5min_meta.json"   "root@${IP}:/opt/spy_quant/cache/"
scp "$SRC\cache\ust10y.parquet"             "root@${IP}:/opt/spy_quant/cache/"
scp "$SRC\cache\ust10y_meta.json"           "root@${IP}:/opt/spy_quant/cache/"
```

---

## Step 3 — Run the hardened setup script

```bash
# Connect to server (port 22 at this point)
ssh "root@YOUR_SERVER_IP"

# Run setup
bash /opt/spy_quant/deployment/vultr_setup.sh
```

The script will:
- Update system + enable automatic security patches
- Harden kernel (SYN flood protection, ASLR, IP spoofing prevention)
- **Change SSH to port 2222** with key-only auth
- Configure ufw firewall
- Install fail2ban (3 failures = 1hr ban, SSH = 24hr ban)
- Install audit logging
- Install Docker with hardened config
- Set up systemd auto-start

### ⚠️ Critical — test port 2222 before closing current session

The script keeps port 22 open until you confirm 2222 works.
Open a **new PowerShell window** and test:

```powershell
ssh -p 2222 "root@YOUR_SERVER_IP"
```

If that connects successfully, close port 22:
```bash
# In your original server session
ufw delete allow 22/tcp
ufw status
```

If port 2222 does NOT connect, you can still use port 22 to fix it:
```bash
# Check what happened
ss -tlnp | grep ssh
cat /etc/ssh/sshd_config | grep Port
journalctl -u sshd --no-pager -n 20
```

All future connections use:
```powershell
ssh -p 2222 "root@YOUR_SERVER_IP"
```

---

## Step 4 — Configure credentials

```bash
ssh -p 2222 "root@YOUR_SERVER_IP"
nano /opt/spy_quant/.env
```

Set these values:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FRED_API_KEY=your_fred_key_here
LIVE_TRADING_ENABLED=false
DEVICE=cpu
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
```

Save: `Ctrl+O` then `Enter`, exit: `Ctrl+X`

---

## Step 5 — Generate TLS certificate

```bash
mkdir -p /opt/spy_quant/deployment/ssl

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/spy_quant/deployment/ssl/server.key \
    -out    /opt/spy_quant/deployment/ssl/server.crt \
    -subj   "/CN=spy-quant-nj"

chmod 600 /opt/spy_quant/deployment/ssl/server.key
chmod 644 /opt/spy_quant/deployment/ssl/server.crt
```

---

## Step 6 — Update nginx.conf to mount the SSL cert

The docker-compose mounts `./nginx.conf` into the nginx container.
Add the SSL volume to docker-compose.yml under the nginx service volumes:

```bash
nano /opt/spy_quant/deployment/docker-compose.yml
```

Under `nginx → volumes`, add:
```yaml
      - ./ssl:/etc/nginx/ssl:ro
```

---

## Step 7 — Build and start

```bash
cd /opt/spy_quant

# Build Docker image (5-10 min first time — downloading CPU torch)
docker compose -f deployment/docker-compose.yml build

# Start everything
systemctl start spy-quant

# Watch logs
docker logs -f spy_quant_trading
```

---

## Step 8 — Verify

```bash
# All containers running?
docker ps

# Trading loop alive?
docker logs spy_quant_trading --tail 20

# Dashboard healthy?
curl -k https://localhost/health

# Firewall status
ufw status verbose

# fail2ban status
fail2ban-client status sshd
```

From your Windows browser: `https://YOUR_SERVER_IP`
(Accept the self-signed cert warning — click Advanced → Proceed)

---

## Pushing model updates from Windows

When you train a better model, push it without rebuilding the image:

```powershell
$IP  = "YOUR_SERVER_IP"
$SRC = "C:\Users\gangu\OneDrive\Desktop\Lancelot1"

# Push new checkpoint
scp -P 2222 "$SRC\models\diffusion_best_trading.pt" "root@${IP}:/opt/spy_quant/models/diffusion_latest.pt"

# Restart trading container to load new model
ssh -p 2222 "root@$IP" "docker restart spy_quant_trading"
```

Note: `-P 2222` (capital P) for scp vs `-p 2222` (lowercase) for ssh.

---

## Day-to-day commands

```bash
# View trading logs
docker logs -f spy_quant_trading

# View all container status
docker ps

# Restart after .env change
docker restart spy_quant_trading

# Check who's been banned
fail2ban-client status sshd

# Check audit log for .env access
ausearch -k secrets_access

# Check recent logins
last | head -20

# Disk usage
df -h

# Container resource usage
docker stats
```

---

## Enable live trading (when ready)

```bash
ssh -p 2222 "root@YOUR_SERVER_IP"
nano /opt/spy_quant/.env

# Change these two lines:
# LIVE_TRADING_ENABLED=true
# ALPACA_BASE_URL=https://api.alpaca.markets

docker restart spy_quant_trading
docker logs -f spy_quant_trading
# Confirm you see "LIVE" not "DRY-RUN" in the logs
```

---

## Security architecture summary

```
Your Browser (HTTPS)
        │
        ▼ port 443
    ┌───────┐
    │ nginx │  TLS 1.2+, rate limiting, security headers
    └───────┘
        │ 172.21.x.x internal
        ▼
    ┌───────────┐
    │ dashboard │  read-only mounts, no-new-privileges
    └───────────┘
        │ 172.20.x.x isolated (no internet)
        ▼
    ┌─────────┐
    │ trading │  no exposed ports, 2GB RAM cap
    └─────────┘
        │ bind mounts
    ┌──────────────────────────┐
    │ /opt/spy_quant/          │
    │   models/ cache/ logs/   │  host filesystem
    └──────────────────────────┘
```

| Layer | Protects against |
|---|---|
| SSH port 2222 + keys only | Automated scanners, brute force |
| fail2ban | Persistent login attempts |
| ufw firewall | All unexpected inbound traffic |
| Kernel sysctl | SYN floods, IP spoofing, ASLR |
| TLS 1.2+ nginx | MITM, credential interception |
| nginx rate limiting | API abuse, DDoS |
| Docker no-new-privileges | Container privilege escalation |
| read_only container rootfs | Malware writing to container |
| cap_drop ALL | Root capabilities inside container |
| Isolated Docker networks | Container-to-container attacks |
| .env chmod 640 | Unauthorized API key access |
| auditd | Detects unexpected file access |
| rkhunter weekly scan | Rootkits and backdoors |
| Unattended upgrades | CVE patching without manual work |

---

## Monthly cost

| Item | Cost |
|---|---|
| Vultr 2 vCPU / 4 GB NJ | ~$24 |
| Alpaca (paper trading) | Free |
| FRED API | Free |
| **Total** | **~$24/month** |
