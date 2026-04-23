# Quick Start

## Install

```bash
curl -fsSL https://chalie.ai/install | bash
```

Checks for Python 3.9+, downloads the latest release, and opens Chalie at **http://localhost:8081**. No root access required. Everything lives in `~/.chalie/`.

---

## CLI

```bash
chalie               # Start → http://localhost:8081
chalie --port=9000   # Custom port
chalie stop
chalie restart
chalie update
chalie status
chalie logs
```

---

## First Run

1. Open **http://localhost:8081/on-boarding/** and create an account.
2. Log in — you land in the chat interface.
3. Open **http://localhost:8081/brain/** → Settings → Providers → Add Provider and configure an LLM.

See [02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md) for provider details.

---

## Recommended Providers

| Option | How |
|--------|-----|
| **Local / free** | Install [Ollama](https://ollama.ai), run `ollama pull gemma3:4b`, point Chalie at `http://localhost:11434` |
| **Cloud (quickest)** | Paste an OpenAI or Anthropic API key in Brain → Settings → Providers |

---

## Voice

Voice features auto-detect on startup. When native dependencies are installed by the installer, voice appears automatically. When they are not, it is silently hidden. To skip voice during install: `--disable-voice`.

---

## Update

```bash
chalie update
```

Stops the process, pulls the latest release, reinstalls dependencies. Data in `~/.chalie/` is never touched.

---

## Uninstall

```bash
chalie stop
rm -rf ~/.chalie ~/.local/bin/chalie
```

Remove the `export PATH="$HOME/.local/bin:$PATH"` line from `~/.bashrc` or `~/.zshrc` if it was added by the installer.

---

## Running from Source

```bash
git clone https://github.com/chalie-ai/chalie.git && cd chalie
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/run.py          # http://localhost:8081
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
