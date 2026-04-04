# Chalie

**Your AI that actually knows you.**

Other AI tools forget you the moment you close the tab. Chalie remembers. It learns what you care about, figures out what you need, and handles things before you ask — all running locally on your machine.

> Alpha software. Rough edges ahead. [Feedback welcome.](https://github.com/chalie-ai/chalie/issues)

```bash
curl -fsSL https://chalie.ai/install | bash
chalie
# → http://localhost:8081
```

---

## What makes it different

**It remembers you.** Not just your last message — your preferences, your projects, your patterns. Mention something in passing today, and it connects the dots three months from now.

**It figures out your goals.** Talk about wanting to learn piano across a few conversations. Chalie notices, forms the goal on its own, and starts helping — no explicit instructions needed.

**It handles your life.** Email, calendar, contacts, web research, reminders, notes, code execution — Chalie picks the right tool and uses it. You say what you want done; it handles the how.

**It thinks when you're not looking.** Background reasoning runs continuously — surfacing insights, tracking your goals, noticing when something needs your attention.

**Nothing leaves your machine.** One SQLite file. Zero telemetry. Zero analytics. Your data stays yours. Period.

---

## Try it now

**With a cloud provider** (fastest start):
```bash
curl -fsSL https://chalie.ai/install | bash
chalie
# Pick OpenAI, Anthropic, or Gemini during onboarding
```

**Fully local and private** (no data leaves your network):
```bash
ollama pull qwen3:8b
curl -fsSL https://chalie.ai/install | bash
chalie
# Select Ollama during onboarding → http://localhost:11434
```

**From source** (for tinkerers):
```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/run.py
# → http://localhost:8081/on-boarding/
```

---

## What it can do

| | |
|---|---|
| **Memory** | Learns and remembers across every conversation — with built-in forgetting so only what matters sticks |
| **Goals** | Detects what you're trying to achieve and tracks progress across sessions |
| **Email & Calendar** | Reads, drafts, schedules — connects to your existing accounts (IMAP, CalDAV, CardDAV) |
| **Web search** | Finds answers when it needs them (DuckDuckGo, no tracking) |
| **Background tasks** | Long-running work that continues across sessions |
| **Notes & lists** | Persistent, searchable, always available in conversation |
| **Code execution** | Runs Python in a sandbox when you need computation |
| **Voice** | Talk to it — local STT and TTS, no cloud transcription |
| **Browser** | Navigates the web when search isn't enough |
| **App integrations** | External apps can pair and expose their own capabilities |

---

## How it's built

Single Python process. No Redis, no Postgres, no message queue, no microservices. One SQLite database with encrypted API keys (AES-256-GCM). Works with Ollama, OpenAI, Anthropic, and Gemini — or all of them at once.

Curious about the internals? → [Architecture](docs/04-ARCHITECTURE.md) · [Schema](backend/schema.sql)

---

## CLI

```bash
chalie                  # start
chalie --port=9000      # custom port
chalie stop             # stop
chalie update           # update to latest
chalie logs             # tail logs
```

---

## Docs

[Quick Start](docs/01-QUICK-START.md) · [Providers](docs/02-PROVIDERS-SETUP.md) · [Architecture](docs/04-ARCHITECTURE.md) · [Tools](docs/09-TOOLS.md) · [Interfaces](docs/15-INTERFACES.md) · [FAQ](docs/FAQ.md)

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [Open an issue](https://github.com/chalie-ai/chalie/issues)

## License

Apache 2.0 — [LICENSE](LICENSE)
