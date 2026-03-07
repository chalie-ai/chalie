# Chalie Troubleshooting Guide — Common Problems, Symptoms, and Solutions

This comprehensive troubleshooting guide helps you diagnose and resolve common issues when running Chalie. Organized by symptom category, each problem includes the observable symptoms, likely causes, and concrete solutions to get your cognitive assistant back online quickly.

---

## Port Conflicts & Startup Issues

### Problem 1: "Address already in use" on port 8081

**Symptom:** When starting Chalie with `python backend/run.py` or `chalie`, you see an error like `OSError: [Errno 98] Address already in use` or the server fails to start.

**Cause:** Another process is already listening on port 8081 (the default), such as a previous Chalie instance, another web service, or Docker container.

**Solution:**
```bash
# Option A: Find and stop the conflicting process
lsof -i :8081          # macOS/Linux
netstat -tlnp | grep 8081   # Linux alternative

# Kill the process (replace PID with actual number)
kill <PID>

# Or use Chalie's built-in command if installed via installer
chalie stop
```

```bash
# Option B: Start on a different port
python backend/run.py --port=9000
chalie --port=9000
```

---

### Problem 2: Server starts but web interface is unreachable

**Symptom:** Chalie reports it's running, but http://localhost:8081 shows "Connection refused" or times out.

**Cause:** The server may be bound to localhost only (127.0.0.1) while you're accessing from another device, or a firewall is blocking the port.

**Solution:**
```bash
# Bind to all interfaces for network access
python backend/run.py --host=0.0.0.0 --port=8081

# Check if process is actually running
ps aux | grep "run.py"

# Verify listening ports
netstat -tlnp | grep 8081
```

---

## Provider & LLM Connection Failures

### Problem 3: Ollama provider connection fails

**Symptom:** When testing or using the Ollama provider, you receive "Connection refused" or timeout errors. The web UI shows provider test failures.

**Cause:** Ollama service is not running on your machine, or the host URL is incorrect (default is `http://localhost:11434`).

**Solution:**
```bash
# Start Ollama daemon
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags

# Check configured endpoint in Chalie web UI → Providers
# Should be: http://localhost:11434 (not 8080 or other ports)
```

---

### Problem 4: Cloud provider API key rejected (Anthropic/OpenAI/Gemini)

**Symptom:** Provider test fails with "API key invalid", "Authentication failed", or HTTP 401/403 errors.

**Cause:** The API key is incorrect, expired, revoked, or has insufficient permissions for the requested model.

**Solution:**
```bash
# Verify your API keys from official sources:
# Anthropic: https://console.anthropic.com → API Keys
# OpenAI: https://platform.openai.com/api-keys
# Google Gemini: https://aistudio.google.com/app/apikey

# In Chalie web UI, navigate to Providers and re-enter the key
# Ensure no leading/trailing whitespace in the key field
```

---

### Problem 5: "Model not found" error for Ollama models

**Symptom:** Provider is configured correctly but requests fail with "model qwen:8b not found" or similar.

**Cause:** The specified model hasn't been downloaded to your local Ollama instance yet.

**Solution:**
```bash
# Download the required model before using it in Chalie
ollama pull qwen:8b
ollama pull mistral:latest
ollama pull llama3.2:latest

# Verify available models
ollama list

# Update provider configuration in Chalie to use an available model name
```

---

### Problem 6: LLM requests timeout or hang indefinitely

**Symptom:** Requests to the LLM take extremely long (>60 seconds) or never complete. The UI shows loading spinners forever.

**Cause:** Network connectivity issues, provider rate limiting, oversized prompts exceeding context limits, or model processing delays on slower hardware.

**Solution:**
```bash
# Check network connectivity to provider endpoint
curl -v https://api.anthropic.com  # For Anthropic
curl -v http://localhost:11434     # For Ollama

# Reduce timeout in provider configuration (web UI → Providers)
# Default is 120 seconds; try reducing to 60 for testing

# Check Chalie logs for specific error messages
chalie logs | tail -50
```

---

## Docker & Tool Execution Issues

### Problem 7: Tools fail with "Docker not available" or container errors

**Symptom:** When using sandboxed tools, you see errors about Docker daemon being unavailable or containers failing to start.

**Cause:** Docker Desktop is not installed, not running, or the Chalie process lacks permissions to access the Docker socket.

**Solution:**
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/

# Start Docker (if installed but stopped)
dockerd  # Linux daemon mode
# Or start Docker Desktop application on macOS/Windows

# Verify Docker is running
docker ps

# Check socket permissions (Linux only)
sudo usermod -aG docker $USER
newgrp docker  # Apply group change

# Restart Chalie after fixing Docker access
```

---

### Problem 8: Tool execution times out during sandbox creation

**Symptom:** Tools that require sandboxed execution hang for >30 seconds before failing with timeout errors.

**Cause:** Slow container image pulls, resource constraints on the host machine, or network issues preventing Docker Hub access.

**Solution:**
```bash
# Pre-pull required images to avoid delays during tool execution
docker pull python:3.12-slim

# Check available system resources
free -h          # Memory usage
df -h            # Disk space
nproc            # CPU cores

# Increase tool timeout in provider configuration if needed
```

---

## Memory & Data Persistence Issues

### Problem 9: Conversations and memories not persisting between sessions

**Symptom:** After restarting Chalie, previous conversations, saved memories, and user preferences are lost.

**Cause:** The SQLite database path is misconfigured, the data directory lacks write permissions, or the database file is corrupted.

**Solution:**
```bash
# Check where Chalie stores its data (default: ~/.chalie/data/)
ls -la ~/.chalie/data/

# Verify database file exists and has content
file ~/.chalie/data/chalie.db
sqlite3 ~/.chalie/data/chalie.db ".tables"

# If using custom path, ensure CHALIE_DB_PATH environment variable is set correctly
export CHALIE_DB_PATH=/path/to/custom/db.sqlite

# Check write permissions
chmod 644 ~/.chalie/data/chalie.db
```

---

### Problem 10: "Database locked" or connection errors

**Symptom:** Chalie fails to start with SQLite database lock errors, or you see concurrent access warnings in logs.

**Cause:** Multiple Chalie instances are running simultaneously and accessing the same database file, or a previous instance didn't close connections properly.

**Solution:**
```bash
# Find all running Chalie processes
ps aux | grep "run.py"
pgrep -f chalie

# Stop all instances
chalie stop
pkill -f "python.*run.py"

# If database is corrupted, backup and reset (WARNING: data loss)
cp ~/.chalie/data/chalie.db ~/.chalie/data/chalie.db.backup
rm ~/.chalie/data/chalie.db
```

---

## Authentication & Session Issues

### Problem 11: Login fails or session expires immediately

**Symptom:** You can't log in to the web interface, or you're logged out automatically after a few minutes of activity.

**Cause:** Password hashing configuration issues, expired session tokens, or browser cookie/LocalStorage problems.

**Solution:**
```bash
# Clear browser cache and cookies for localhost:8081
# Or try incognito/private browsing mode

# Reset your password via the web UI (if accessible)
# Navigate to /on-boarding/ to create a new account if needed

# Check Chalie logs for authentication errors
chalie logs | grep -i "auth\|login"
```

---

### Problem 12: Onboarding wizard fails or loops indefinitely

**Symptom:** The onboarding flow at http://localhost:8081/on-boarding/ doesn't complete, shows blank screens, or redirects in a loop.

**Cause:** JavaScript errors in the browser, incomplete database schema migrations, or missing required configuration steps.

**Solution:**
```bash
# Open browser developer tools (F12) and check Console for JS errors

# Verify all database tables exist
sqlite3 ~/.chalie/data/chalie.db ".tables"

# Run pending migrations manually if needed
python backend/scripts/reset_db.py  # WARNING: resets database

# Try accessing onboarding with ?reset=true parameter
http://localhost:8081/on-boarding/?reset=true
```

---

## Voice Feature Issues

### Problem 13: Voice features not appearing in the UI

**Symptom:** The voice input/output options are missing from the chat interface, even though you expected them to be available.

**Cause:** Native voice dependencies (soundfile, pywebpush) weren't installed during setup, or the `--disable-voice` flag was used.

**Solution:**
```bash
# Install voice dependencies manually
pip install -r backend/requirements-voice.txt

# Verify native libraries are available
python3 -c "import soundfile; print('soundfile OK')"
python3 -c "import pywebpush; print('pywebpush OK')"

# Restart Chalie to detect newly installed dependencies
chalie restart
```

---

### Problem 14: Voice input/output produces no audio or garbled output

**Symptom:** Voice features appear in the UI but produce silence, distorted audio, or fail with cryptic errors.

**Cause:** Missing system-level audio libraries (libsndfile), incorrect audio device configuration, or incompatible browser for Web Speech API.

**Solution:**
```bash
# Install required system libraries
sudo apt-get install libsndfile1      # Debian/Ubuntu
brew install libsamplerate            # macOS

# Check available audio devices
arecord -l    # Linux ALSA input devices
aplay -l      # Linux ALSA output devices

# Try a different browser (Chrome has best Web Speech API support)
```

---

## Default Tools Installation Issues

### Problem 15: Default tools fail to auto-install on first run

**Symptom:** After fresh installation, default tools are missing or show as "failed to install" in the tools list.

**Cause:** Network connectivity issues preventing GitHub release downloads, firewall blocking outbound connections, or insufficient disk space.

**Solution:**
```bash
# Check network connectivity to GitHub
curl -I https://api.github.com/repos/chalie-ai/communicate-tool/releases/latest

# Verify available disk space
df -h ~/.chalie/tools

# Manually trigger tool installation by restarting Chalie
chalie restart

# Or disable default tools entirely (if not needed)
touch ~/.chalie/data/.no-default-tools
```

---

## Related Documentation

- **[01-QUICK-START.md](01-QUICK-START.md)** — Installation and basic usage instructions
- **[02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md)** — Detailed provider configuration guide with troubleshooting tips
- **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — System architecture reference for understanding data flow and components
- **[09-TOOLS.md](09-TOOLS.md)** — Tools system documentation including sandbox requirements
- **[20-DEPLOYMENT.md](20-DEPLOYMENT.md)** — Production deployment options and configuration

---

## Getting More Help

If you've tried all solutions above and still need assistance:

1. **Check the logs:** `chalie logs` or inspect `~/.chalie/data/chalie.log`
2. **Review recent changes:** Check if updates broke something (`git log --oneline -5`)
3. **Search existing issues:** https://github.com/chalie-ai/chalie/issues
4. **Create a new issue:** Include your logs, Chalie version, and steps to reproduce

---

*Last updated: 2026-03-07 | Version: Phase 3 Documentation Overhaul*
