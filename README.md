# Chalie

**A local-first cognitive runtime that remembers, reasons, and acts — running entirely on your machine.**

> Alpha software. Expect rough edges. [Issues welcome.](https://github.com/chalie-ai/chalie/issues)

```bash
curl -fsSL https://chalie.ai/install | bash
chalie
# → http://localhost:8081
```

Pick an LLM provider during onboarding (Ollama, OpenAI, Anthropic, Gemini) and start talking. That's it.

---

## What this actually is

Chalie is not a chatbot wrapper. It's a **persistent reasoning engine** that runs as a single Python process on your hardware.

**Memory that compounds.** Every conversation feeds a multi-layer memory pipeline — raw transcripts compress into episodes, episodes distill into concepts, concepts decay based on relevance. Context from three months ago surfaces when it matters today.

**Autonomous goal formation.** Mention something casually across a few conversations — Chalie notices the pattern, forms a goal, builds a plan, and starts executing. No explicit instruction needed.

**Background cognition.** A continuous reasoning loop runs during idle time — decaying stale goals, clustering signals, generating insights, and triggering proactive actions. It thinks when you're not talking to it.

**Tool use with judgment.** Web search, code execution, documentation lookup, email, calendar, contacts — Chalie picks and uses tools on its own. External apps can pair via a bluetooth-style interface protocol and expose their own capabilities.

**Single SQLite database.** No Redis, no Postgres, no message queue. One file. Encrypted API keys (AES-256-GCM). Zero telemetry. Zero analytics. Nothing leaves your machine unless you configure an external LLM.

---

## For the fully private setup

```bash
ollama pull qwen3:8b
# During onboarding → select Ollama → http://localhost:11434
```

Everything runs local. No API keys, no cloud, no data leaving your network.

---

## Build from source

```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/run.py
# → http://localhost:8081/on-boarding/
```

```bash
cd backend && pytest -m unit    # run tests
```

### CLI

```bash
chalie                  # start (default :8081)
chalie --port=9000      # custom port
chalie stop             # stop
chalie update           # pull latest
chalie logs             # tail logs
```

---

## Under the hood

Single process, multi-threaded. No microservices, no containers required (Docker available for deployment).

- **Memory hierarchy** — transcript → compaction → episodes → knowledge (hybrid search: exact + FTS5 + vector KNN via sqlite-vec)
- **Cognitive drift engine** — idle-loop goal ecology: decay, clustering, proactive triggers, attention gating
- **Unified LLM path** — one call handles response generation + skill invocation + tool dispatch. No routing gate.
- **Model-agnostic** — different cognitive functions can use different providers/models simultaneously
- **ONNX classifiers** — lightweight specialized models (mode tiebreaker, contradiction detection, trait extraction) run locally
- **Voice** — Moonshine STT + Kokoro TTS, both ONNX, both local
- **Block protocol** — all content is structured JSON blocks, never raw HTML

Architecture docs: [`docs/04-ARCHITECTURE.md`](docs/04-ARCHITECTURE.md) · Schema: [`backend/schema.sql`](backend/schema.sql)

---

## Built-in capabilities

| Capability | Details |
|---|---|
| Persistent memory | Multi-layer pipeline with decay, semantic search, knowledge extraction |
| Goal tracking | Emergent goals from conversation signals, autonomous execution |
| Scheduling | Reminders, recurring tasks, calendar-aware timing |
| Email & contacts | IMAP, CalDAV, CardDAV — multiple provider support |
| Web search | Privacy-focused via DuckDuckGo |
| Code execution | Sandboxed Python eval |
| Doc search | 12 languages, 11 frameworks |
| Notes & lists | Persistent, searchable, structured |
| Browser | Headless Playwright for web interaction |
| Background tasks | Multi-session execution with plan-aware step DAGs |

Tools are self-declaring via manifests. No tool-specific logic in core infrastructure.

---

## Documentation

- [Quick Start](docs/01-QUICK-START.md) — install, configure, deploy
- [Providers](docs/02-PROVIDERS-SETUP.md) — LLM provider setup
- [Architecture](docs/04-ARCHITECTURE.md) — full system design
- [Tools](docs/09-TOOLS.md) — tool system and authoring
- [Interfaces](docs/15-INTERFACES.md) — external app integration
- [FAQ](docs/FAQ.md)

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [Open an issue](https://github.com/chalie-ai/chalie/issues)

## License

Apache 2.0 — [LICENSE](LICENSE)
