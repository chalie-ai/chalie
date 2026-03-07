# Chalie Deployment Guide — Local Development, Systemd, Docker Compose, and Reverse Proxy Setup

This deployment guide covers multiple methods for running Chalie in different environments. Whether you're developing locally, setting up a production server with systemd, containerizing with Docker Compose, or configuring reverse proxies for secure access, this guide provides step-by-step instructions for each approach.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Systemd Service (Production Linux)](#systemd-service-production-linux)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Reverse Proxy Configuration](#reverse-proxy-configuration)
5. [Environment Variables Reference](#environment-variables-reference)

---

## Local Development

### Prerequisites

- **Python 3.9+** (tested with 3.10, 3.11, 3.12)
- **Git** for cloning the repository
- **Docker Desktop** (optional, required only for sandboxed tool execution)
- **Ollama** (recommended for local LLM inference) or API keys for cloud providers

### Installation Steps

#### Step 1: Clone the Repository

```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
```

#### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
```

#### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r backend/requirements.txt

# Optional: Voice features (STT/TTS)
pip install -r backend/requirements-voice.txt
```

#### Step 4: Run Chalie

```bash
# Default configuration (port 8081, localhost only)
python backend/run.py

# Custom port and host binding
python backend/run.py --port=9000 --host=0.0.0.0

# With verbose logging for debugging
python backend/run.py --verbose
```

The web interface will be available at **http://localhost:8081** (or your custom port).

#### Step 5: Complete Onboarding

1. Open http://localhost:8081/on-boarding/ in your browser
2. Create an account with a password
3. Configure your LLM provider (Ollama, Anthropic, OpenAI, or Gemini)
4. Start using Chalie!

### Development Mode Tips

```bash
# Run tests
cd backend && pytest

# Watch for code changes (manual restart required currently)
python backend/run.py --port=8081

# Check logs in real-time
chalie logs  # If installed via installer script
tail -f ~/.chalie/data/chalie.log  # Alternative log location
```

---

## Systemd Service (Production Linux)

### Overview

Systemd provides process management, automatic startup on boot, logging integration, and service monitoring for production deployments. This method is recommended for servers running Chalie continuously.

### Prerequisites

- **Linux distribution with systemd** (Ubuntu 16.04+, Debian 8+, CentOS 7+)
- **Python 3.9+** installed system-wide or in a virtual environment
- **Root/sudo access** to create service files

### Installation Steps

#### Step 1: Install Chalie as a System Service

```bash
# Create the application directory
sudo mkdir -p /opt/chalie
cd /opt/chalie

# Clone the repository (or copy your existing installation)
git clone https://github.com/chalie-ai/chalie.git .

# Create virtual environment in service directory
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

#### Step 2: Create the Systemd Service File

Create `/etc/systemd/system/chalie.service`:

```ini
[Unit]
Description=Chalie Cognitive Assistant Server
Documentation=https://github.com/chalie-ai/chalie
After=network.target

[Service]
Type=simple
User=chalie
Group=chalie
WorkingDirectory=/opt/chalie
Environment="PATH=/opt/chalie/venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/chalie/venv/bin/python backend/run.py --port=8081 --host=0.0.0.0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/chalie/data

[Install]
WantedBy=multi-user.target
```

#### Step 3: Create Service User and Directories

```bash
# Create dedicated user for running Chalie
sudo useradd -r -s /bin/false chalie

# Create data directory with proper permissions
sudo mkdir -p /opt/chalie/data
sudo chown -R chalie:chalie /opt/chalie
chmod 750 /opt/chalie/data
```

#### Step 4: Enable and Start the Service

```bash
# Reload systemd to recognize new service file
sudo systemctl daemon-reload

# Enable Chalie to start on boot
sudo systemctl enable chalie

# Start the service
sudo systemctl start chalie

# Check status
sudo systemctl status chalie
```

#### Step 5: View Logs and Manage Service

```bash
# View real-time logs
sudo journalctl -u chalie -f

# View last 100 lines of logs
sudo journalctl -u chalie -n 100

# Stop the service
sudo systemctl stop chalie

# Restart after configuration changes
sudo systemctl restart chalie
```

### Security Considerations for Systemd

- The `chalie` user runs with minimal privileges
- `ProtectSystem=strict` prevents access to system directories
- `ReadWritePaths` limits write access to only the data directory
- Consider adding firewall rules to restrict network access:

```bash
# Allow only specific ports (UFW example)
sudo ufw allow 8081/tcp
sudo ufw enable
```

---

## Docker Compose Deployment

### Overview

Docker Compose provides a containerized deployment option with isolated dependencies, easy scaling, and simplified configuration management. This is ideal for development environments or when you need consistent deployments across different systems.

### Prerequisites

- **Docker Engine** 20.10+
- **Docker Compose** v2.0+ (or docker-compose plugin)

### Installation Steps

#### Step 1: Create Docker Compose Configuration

Create `docker-compose.yml` in your project root:

```yaml
version: '3.8'

services:
  chalie:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: chalie
    ports:
      - "8081:8081"
    volumes:
      - chalie_data:/app/data
      - /var/run/docker.sock:/var/run/docker.sock  # For sandboxed tools
    environment:
      - CHALIE_DB_PATH=/app/data/chalie.db
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    mem_limit: 2g
    cpus: 2.0

volumes:
  chalie_data:
    driver: local
```

#### Step 2: Create Dockerfile

Create `Dockerfile` in your project root:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directory
RUN mkdir -p /app/data && chmod 755 /app/data

# Expose port
EXPOSE 8081

# Run the application
CMD ["python", "backend/run.py", "--host=0.0.0.0"]
```

#### Step 3: Build and Run

```bash
# Build and start containers in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f chalie

# Stop all containers
docker-compose down

# Restart after changes
docker-compose restart
```

### Docker Deployment Tips

- **Persistent data:** The `chalie_data` volume ensures your database survives container restarts
- **Resource limits:** Adjust `mem_limit` and `cpus` based on your hardware
- **Docker socket mounting:** Required for sandboxed tool execution; consider security implications
- **Production hardening:** Remove Docker socket access if not using tools, use network policies

---

## Reverse Proxy Configuration

### Overview

A reverse proxy provides SSL/TLS termination, authentication, rate limiting, and centralized logging. This section covers Nginx and Caddy configurations for securing Chalie deployments.

### Prerequisites

- **Nginx** or **Caddy** installed on your server
- Valid domain name pointing to your server's IP address
- Root/sudo access to configure the proxy

---

### Option A: Nginx Configuration

#### Step 1: Install and Configure Nginx

```bash
# Install Nginx (Ubuntu/Debian)
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx

# Create site configuration
sudo nano /etc/nginx/sites-available/chalie
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name chalie.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name chalie.example.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/chalie.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chalie.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        
        # WebSocket support (if needed)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

#### Step 2: Enable Site and Obtain SSL Certificate

```bash
# Create symbolic link to enable site
sudo ln -s /etc/nginx/sites-available/chalie /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx

# Obtain Let's Encrypt certificate
sudo certbot --nginx -d chalie.example.com
```

---

### Option B: Caddy Configuration (Simpler Alternative)

#### Step 1: Install Caddy

```bash
# Download and install Caddy
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

#### Step 2: Create Caddyfile

Create `/etc/caddy/Caddyfile`:

```caddy
chalie.example.com {
    reverse_proxy localhost:8081 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {http.request.header.X-Forwarded-For}
        header_up X-Forwarded-Proto {scheme}
    }

    # Automatic HTTPS with Let's Encrypt
    encode gzip
    
    # Security headers (automatic in Caddy)
    
    # Rate limiting
    rate_limit {
        zone chalie_ip {
            key remote_addr()
            requests 100/s
            burst 200
        }
    }
}
```

#### Step 3: Start Caddy

```bash
# Start Caddy service
sudo systemctl enable caddy --now
sudo systemctl start caddy

# View logs
sudo journalctl -u caddy -f
```

---

## Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CHALIE_DB_PATH` | Path to SQLite database file | `~/.chalie/data/chalie.db` | `/opt/chalie/data/chalie.db` |
| `CHALIE_LOG_LEVEL` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) | `INFO` | `DEBUG` |
| `CHALIE_HOST` | Network interface to bind to | `127.0.0.1` | `0.0.0.0` |
| `CHALIE_PORT` | Port number for the server | `8081` | `9000` |
| `PYTHONUNBUFFERED` | Disable Python output buffering | Not set | `1` |

### Setting Environment Variables

```bash
# For systemd service, edit /etc/systemd/system/chalie.service and add:
Environment="CHALIE_DB_PATH=/opt/chalie/data/chalie.db"
Environment="CHALIE_LOG_LEVEL=DEBUG"

# For Docker Compose, add to environment section in docker-compose.yml
environment:
  - CHALIE_DB_PATH=/app/data/chalie.db
  - CHALIE_LOG_LEVEL=INFO

# For local development, export before running
export CHALIE_PORT=9000
python backend/run.py
```

---

## Related Documentation

- **[19-TROUBLESHOOTING.md](19-TROUBLESHOOTING.md)** — Common problems and solutions for deployment issues
- **[02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md)** — LLM provider configuration after deployment
- **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — System architecture for understanding component interactions

---

*Last updated: 2026-03-07 | Version: Phase 3 Documentation Overhaul*
