<p align="center">
  <img src="logo.png" alt="Chalie" width="180">
</p>

<h1 align="center">Chalie</h1>

<p align="center">
  <strong>Life, handled.</strong><br>
  One persistent intelligence that sees your entire digital life, understands what matters, and handles it — so you stop managing and start living.
</p>

<p align="center">
  <a href="https://github.com/chalie-ai/chalie/releases"><img src="https://img.shields.io/github/v/release/chalie-ai/chalie?include_prereleases&style=for-the-badge&color=7c3aed" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-06b6d4?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/chalie-ai/chalie/issues"><img src="https://img.shields.io/github/issues/chalie-ai/chalie?style=for-the-badge&color=ec4899" alt="Issues"></a>
  <a href="docs/04-ARCHITECTURE.md"><img src="https://img.shields.io/badge/Docs-architecture-7c3aed?style=for-the-badge" alt="Docs"></a>
</p>

```bash
curl -fsSL https://chalie.ai/install | bash
chalie
# → http://localhost:31025
```

> Alpha. Sharp edges. If you hit one, [open an issue](https://github.com/chalie-ai/chalie/issues) — we respond fast.

---

## Why Chalie is different

Every other AI tool is a fresh conversation. Chalie is a **continuous reasoning engine** that keeps running in the background, remembers everything that mattered, forgets what didn't, and acts when it's confident.

<table>
<tr><td><b>It remembers you — then forgets the noise</b></td><td>A four-stage memory pipeline compresses raw conversation into episodes, concepts, and abstractions. Lossy on purpose — forgetting is what makes recall useful three months later.</td></tr>
<tr><td><b>It figures out your goals on its own</b></td><td>Mention "I want to learn piano" three times across two weeks. Chalie detects the pattern, forms the goal, and starts helping — no explicit instructions, no reminders you have to set.</td></tr>
<tr><td><b>It thinks while you're not looking</b></td><td>Background workers run continuously — consolidating memory, pursuing goals, checking on things you asked about. When you come back, progress is already made.</td></tr>
<tr><td><b>It handles the doing, not just the talking</b></td><td>Email, calendar, contacts, web research, code execution, voice, notes, browser, reminders — all first-class. You say what you want done; Chalie picks the tool.</td></tr>
<tr><td><b>Your machine, your data</b></td><td>One SQLite file. API keys encrypted with AES-256-GCM. Zero telemetry, zero analytics, zero phoning home. Runs fully local with Ollama, or mix in OpenAI / Anthropic / Gemini if you want.</td></tr>
<tr><td><b>One process, no infrastructure</b></td><td>No Redis, no Postgres, no message queue, no microservices. A single Python process with threads. Starts in seconds, works behind any reverse proxy, installs with one line.</td></tr>
<tr><td><b>Model-agnostic by design</b></td><td>Different cognitive functions can use different models. Local classifier + cloud reasoner + local voice — mix and match. No vendor lock-in.</td></tr>
</table>

---

## Install

**Fastest start — pick any provider on first boot:**
```bash
curl -fsSL https://chalie.ai/install | bash
chalie
```
Then pick OpenAI, Anthropic, Gemini, or Ollama in the onboarding wizard.

**Fully local, nothing leaves your network:**
```bash
ollama pull gemma4:31b
curl -fsSL https://chalie.ai/install | bash
chalie                 # choose Ollama → http://localhost:11434
```

**From source, for tinkerers:**
```bash
git clone https://github.com/chalie-ai/chalie.git && cd chalie
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/run.py  # → http://localhost:31025/on-boarding/
```

---

## What it can do today

| | |
|---|---|
| 🧠 **Persistent memory** | Four-stage pipeline: Topic Context → Episode → Concept → Abstraction. Built-in decay means only what matters sticks. |
| 🎯 **Autonomous goals** | Detects goals from casual mentions, tracks progress, nudges when progress stalls. |
| 📬 **Email & calendar** | IMAP, CalDAV, CardDAV — connects to accounts you already have, no vendor lock. |
| 🔎 **Web search** | DuckDuckGo by default. No tracking, no search history sold. |
| 🗓 **Scheduler** | Natural-language reminders and recurring jobs. "Remind me every Monday at 9" just works. |
| 📝 **Notes & lists** | Persistent, searchable, always in context. |
| 🐍 **Code execution** | Sandboxed Python when you need real computation, not a vibe. |
| 🎙 **Voice, fully local** | Moonshine STT + Kokoro TTS, both ONNX. No cloud transcription, ever. |
| 🌐 **Browser** | Navigates the web when search isn't enough. |
| 🔌 **Open tool protocol** | External apps can pair in and expose their own capabilities. |

---

## How it's built

One Python process. Flask + `flask-sock` WebSocket. SQLite with `sqlite-vec` and FTS5 for semantic + lexical recall. Vanilla ES6 modules on the frontend — no build step, no npm lockfile drama. The entire cognitive runtime — memory, goals, scheduler, voice — runs as daemon threads inside that one process.

Works with **Ollama, OpenAI, Anthropic, Gemini** — or any combination. Different models for different cognitive functions, by design.

Curious? → [Architecture](docs/04-ARCHITECTURE.md) · [Schema](backend/schema.sql) · [Message flow](docs/13-MESSAGE-FLOW.md) · [Tools](docs/09-TOOLS.md)

---

## CLI

```bash
chalie                  # start
chalie --port=9000      # custom port
chalie stop             # stop the daemon
chalie update           # update to latest
chalie logs             # tail the log
```

---

## Docs

[Quick Start](docs/01-QUICK-START.md) · [Providers](docs/02-PROVIDERS-SETUP.md) · [Architecture](docs/04-ARCHITECTURE.md) · [Web Interface](docs/03-WEB-INTERFACE.md) · [Tools](docs/09-TOOLS.md) · [Interfaces](docs/15-INTERFACES.md) · [Testing](docs/12-TESTING.md) · [FAQ](docs/FAQ.md)

---

## Philosophy

- **Judgment over activity.** Fewer high-confidence actions beat many low-confidence ones.
- **Restraint builds trust.** Every token, notification, and action earns its place.
- **Continuity is intelligence.** The persistent runtime — not any single response — is the product.
- **Constraints are features.** Decay, capacity limits, and token budgets are what make wisdom emerge.
- **See everything. Show only what matters.**

Full product compass: [docs/00-VISION.md](docs/00-VISION.md).

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) or [open an issue](https://github.com/chalie-ai/chalie/issues).

## License

[Apache 2.0](LICENSE) — use it, fork it, ship it.
