# Frequently Asked Questions

## What is Chalie?

Chalie is a **persistent cognitive agent** — a continuously running runtime that forms memories, decays irrelevant information, exercises judgment, and evolves through interaction. It is not a chatbot, not an assistant wrapper, and not a request-response service.

The key distinction: most AI tools respond to what you ask. Chalie runs continuously, accumulates context over time, generates spontaneous thoughts during idle periods, and acts autonomously on background tasks — all while protecting your attention by only involving you when your judgment, identity, or values are required.

---

## How is Chalie different from ChatGPT / Claude / other AI assistants?

| | Chalie | Typical AI assistant |
|---|---|---|
| **Memory** | Persistent, decaying, cross-session | None (or per-session only) |
| **Identity** | Evolves through experience | Stateless |
| **Background activity** | Yes — cognitive drift, curiosity threads, persistent tasks | No |
| **Attention protection** | Core design principle | Not a concern |
| **Runs on your machine** | Yes — local-first, no cloud required | Cloud-dependent |
| **Multiple LLM providers** | Yes — Ollama, Anthropic, OpenAI, Gemini | Single provider |

---

## Does Chalie send my data anywhere?

No. Chalie is local-first by design. All data — conversations, memories, traits, documents — is stored in a SQLite database on your own machine. The only external network calls are to whichever LLM provider you configure (Ollama runs entirely locally; cloud providers like Anthropic/OpenAI receive only the text you send in a message, not your stored memories).

There is no telemetry, no analytics, no cloud sync.

---

## What does "memory decays" mean?

Chalie does not store everything forever. Episodic memories (specific conversation events) decay faster; semantic concepts (distilled knowledge) decay slower. Memories that are reinforced through repeated relevance survive longer. Memories that are never accessed fade and are eventually deleted.

This mirrors how human memory works — and it serves a practical purpose: it prevents Chalie from accumulating an ever-growing pile of outdated, contradictory noise. What persists is what matters.

You can inspect Chalie's memory at any time via the Brain dashboard or the `/system/observability/memory` endpoint.

---

## What LLM providers does Chalie support?

- **Ollama** (local, recommended for privacy) — runs models like `qwen3:8b` entirely on your machine
- **Anthropic** — Claude models via API key
- **OpenAI** — GPT models via API key
- **Google Gemini** — Gemini models via API key

You can assign different providers to different cognitive functions (e.g., use a local model for memory tasks and a cloud model for complex reasoning). See `docs/02-PROVIDERS-SETUP.md` for configuration.

---

## What does Chalie do when I'm not talking to it?

Several things, depending on configuration and activity level:

- **Cognitive drift** — During idle periods, Chalie generates spontaneous thoughts via its Default Mode Network (DMN). These may surface as proactive messages, curiosity threads, or background plan proposals.
- **Memory consolidation** — Episodes are compressed into semantic concepts; memories are decayed.
- **Curiosity pursuit** — Active curiosity threads are explored via the ACT loop (6h cycle).
- **Persistent tasks** — Background tasks continue executing (30min cycles).
- **Autobiography synthesis** — A running narrative of who you are and what matters to you is updated (6h cycle).

All background activity is attention-gated: if you're in deep focus, Chalie stays silent.

---

## Can Chalie take actions autonomously?

Yes, within hard limits. Chalie can:
- Execute tasks via its ACT loop using sandboxed tools
- Schedule reminders and manage lists
- Research topics via curiosity threads
- Generate proactive suggestions and follow-ups

Chalie will **not** take irreversible or destructive actions autonomously. Consequential actions (anything that affects external systems or requires user identity) are paused for confirmation. Silent autonomous handling is the default only for safe, reversible, or informational actions.

---

## What are "tools" in Chalie?

Tools extend Chalie's ability to take action in the world: web search, weather, code execution, etc. Tools are isolated capsules — they run either in sandboxed Docker containers (no access to your system) or as trusted subprocesses (for first-party tools). Chalie's infrastructure is tool-agnostic: it doesn't know or care what specific tools are installed.

See `docs/09-TOOLS.md` for how tools work and `docs/14-DEFAULT-TOOLS.md` for the tools installed by default.

---

## How do I configure an LLM provider?

1. Start Chalie and open `http://localhost:8081/on-boarding/`
2. Complete onboarding — you'll be asked to configure a provider
3. For Ollama: install from [ollama.ai](https://ollama.ai), pull a model (`ollama pull qwen3:8b`), set endpoint to `http://localhost:11434`
4. For cloud providers: paste your API key — it is encrypted and stored locally

See `docs/02-PROVIDERS-SETUP.md` for full details.

---

## How do I reset or delete Chalie's memory?

Via the REST API or Brain dashboard:
- **Delete a specific trait**: `DELETE /system/observability/traits/<key>`
- **Privacy endpoints**: `DELETE /api/privacy/data` — full data wipe
- **Export your data**: `GET /api/privacy/export`

Memories also decay naturally over time without any intervention.

---

## Where does Chalie store its data?

Everything is in a single SQLite database at `backend/data/chalie.db`. No cloud storage, no external databases. You can back it up by copying that file.

---

## Is Docker required?

No. Docker is only used for **sandboxed tool execution** — tools that need isolation from your system. Trusted first-party tools (like the weather tool) run as subprocesses without Docker. The core runtime, voice features, and all cognitive services run natively without Docker.

---

## Does Chalie support voice?

Yes — native speech-to-text (faster-whisper) and text-to-speech (KittenTTS) are built in and auto-detect their dependencies on startup. No Docker required. The voice service degrades gracefully (returns 503) if dependencies aren't installed.

---

## What is the Brain dashboard?

The Brain dashboard (`http://localhost:8081/brain/`) is the admin and observability interface. It shows:
- Routing decision distribution
- Memory layer health
- Active curiosity threads and persistent tasks
- User traits and autobiography narrative
- Tool performance metrics
- Identity vector states

It is read-only — it does not modify Chalie's state.
