# Chalie Quick Start Guide

This comprehensive guide covers Chalie documentation, technical guide, providing essential information for developers and users. For related topics, see: 


## Install Chalie

```bash
curl -fsSL https://chalie.ai/install | bash
```

The installer:
1. Checks prerequisites (Python 3.9+, Docker optional)
2. Downloads the latest release and builds in place (~2 min)
3. Installs the `chalie` CLI and opens Chalie at **http://localhost:8081**

No root access required. Everything lives in `~/.chalie/`.

---

## After Install

```bash
chalie                 # Start Chalie → http://localhost:8081
chalie --port=9000     # Start on a custom port
chalie stop            # Stop the process
chalie restart         # Restart
chalie update          # Update to the latest release
chalie status          # Check if running
chalie logs            # Follow the log
```

---

## Onboarding

Open **http://localhost:8081/on-boarding/** and:

1. **Create an account** — set a password
2. **Configure an LLM provider** — choose from the options below
3. **Begin** — you'll be redirected to the chat interface

---

## LLM Providers

### Ollama (local, recommended)

Free, private, runs entirely on your machine.

```bash
# Install from https://ollama.ai, then:
ollama pull qwen:8b
```

In onboarding, select **Ollama** and set the endpoint to `http://localhost:11434`.

### OpenAI

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. In onboarding, select **OpenAI** and paste your key

### Anthropic

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. In onboarding, select **Anthropic** and paste your key

### Google Gemini

1. Get an API key from [ai.google.dev](https://ai.google.dev)
2. In onboarding, select **Gemini** and paste your key

---

## Configuration

All configuration (LLM providers, API keys, settings) is done via the web UI after first run. The only runtime option is the port:

```bash
chalie --port=9000     # Start on a custom port (default: 8081)
```

Voice features auto-detect native dependencies — no Docker needed. When voice deps are installed (via the installer or `pip install -r backend/requirements-voice.txt`), voice appears automatically. When they're not, voice is silently hidden. Use `--disable-voice` during install to skip voice dependencies entirely.

---

## Updating

```bash
chalie update
```

Re-runs the installer with `CHALIE_UPDATE=1`: stops the running process, downloads the latest source, reinstalls dependencies. Your database and memory in `~/.chalie/data/` are never touched.

---

## Uninstalling

```bash
chalie stop
rm -rf ~/.chalie ~/.local/bin/chalie
```

Remove the `export PATH="$HOME/.local/bin:$PATH"` line from `~/.bashrc` or `~/.zshrc` if it was added by the installer.

---

## For Hackers & Contributors

Want to run from source, patch internals, or contribute?

**Prerequisites:** Python 3.9+, git

**Steps:**

**1. Clone the repo**
```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
```

**2. Create a virtual environment and install dependencies**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

**3. Run Chalie**
```bash
python backend/run.py
# Opens at http://localhost:8081
```

**4. Run tests**
```bash
cd backend && pytest
```

**Port override** (optional):

```bash
python backend/run.py --port=9000
```

All other configuration (LLM providers, API keys) is done via the web UI after first run. Voice auto-detects native dependencies (no Docker needed).

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
