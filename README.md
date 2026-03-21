# Chalie

> **⚠️ ALPHA SOFTWARE — expect bugs, breaking changes, and rough edges.**
> Your feedback is genuinely valuable — please [open an issue](https://github.com/chalie-ai/chalie/issues) with anything you find.

**A personal intelligence layer that remembers, reasons, and acts on your behalf — so you can think less about doing.**

```bash
curl -fsSL https://chalie.ai/install | bash
chalie
```

That's it. Open **http://localhost:8081**, create an account, pick an LLM provider, and start talking.

<img src="docs/images/cognition.png" width="700" alt="Cognition" />

---

## What Is Chalie?

Most AI tools are fast but forgetful. Conversations reset. Notes pile up without meaning. Automation acts without awareness.

Chalie is different — it's a **persistent cognitive agent** that runs on your machine. It remembers what matters, forgets what doesn't, and builds understanding over time. Every interaction makes it smarter about *you*.

- **Persistent memory** — conversations carry forward across sessions. Context surfaces when it's relevant, not just when you ask.
- **Acts on your behalf** — manages lists, sets reminders, searches the web, runs code, and connects to external apps.
- **Thinks on its own** — generates spontaneous insights during idle time, not just when prompted.
- **Fully private** — everything stays on your machine in a single local database. Zero telemetry, zero analytics.

---

## Quick Start

**One-line install:**
```bash
curl -fsSL https://chalie.ai/install | bash
```

**CLI basics:**
```bash
chalie                 # Start → http://localhost:8081
chalie --port=9000     # Custom port
chalie stop            # Stop
chalie update          # Update to latest
chalie logs            # Follow the log
```

**Recommended: run a local model (free, fully private):**
```bash
ollama pull qwen:8b
# During onboarding, select Ollama → http://localhost:11434
```

Also works with **OpenAI**, **Anthropic**, and **Google Gemini** — configure one or several, assign different providers to different jobs.

<img src="docs/images/providers.png" width="680" alt="Providers" />

For full setup details, see [Quick Start Guide](docs/01-QUICK-START.md).

---

## What Can It Do?

### Remember

Chalie maintains layered memory that decays naturally over time — like human memory. Important things stick around. Noise fades. Retrieval gets smarter with use.

<img src="docs/images/memory-frontend.png" width="680" alt="Memory" />

### Manage Lists & Reminders

Tell Chalie to add items, set reminders, or schedule tasks — all in natural language. It handles the bookkeeping.

<img src="docs/images/lists-frontend.png" width="480" alt="Lists" /> <img src="docs/images/scheduler-frontend.png" width="480" alt="Scheduler" />

### Use Tools

Built-in tools that Chalie can use on its own when needed:

| Tool | What it does |
|---|---|
| Weather | Current conditions and forecasts (no API key needed) |
| Web Search | Privacy-focused search via DuckDuckGo |
| Code Eval | Run Python snippets in a sandbox |
| Docs Search | Official docs for 12 languages and 11 frameworks |

### Connect External Apps

Chalie pairs with external applications via a bluetooth-style protocol. Connected apps can expose their own capabilities that Chalie uses as tools. See [Interfaces](docs/15-INTERFACES.md) for the protocol spec.

---

## Build from Source

```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/run.py
```

Open **http://localhost:8081/on-boarding/** to get started. Run tests with `cd backend && pytest`.

---

## Privacy

All memory, conversation history, and learned context stays on your machine in a single SQLite database. Chalie makes zero external calls unless you configure an external LLM provider — and even then, only the current message is transmitted. API keys are encrypted at rest.

No telemetry. No analytics. No background sync. You own your data.

---

## Documentation

| Document | Contents |
|---|---|
| [Vision](docs/00-VISION.md) | Product vision and design philosophy |
| [Quick Start](docs/01-QUICK-START.md) | Full setup guide and deployment |
| [Providers](docs/02-PROVIDERS-SETUP.md) | Configuring LLM providers |
| [Architecture](docs/04-ARCHITECTURE.md) | System architecture and services |
| [Tools](docs/09-TOOLS.md) | Tools system and how to add new ones |
| [Interfaces](docs/15-INTERFACES.md) | External app integration protocol |
| [FAQ](docs/FAQ.md) | Common questions |

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, or [open an issue](https://github.com/chalie-ai/chalie/issues).

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
