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
| **Background activity** | Yes — cognitive drift, proactive thoughts, persistent tasks | No |
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

You can inspect Chalie's memory at any time via the Brain dashboard Memory tab, which shows paginated records across episodes, user facts, and system knowledge.

---

## What LLM providers does Chalie support?

- **Ollama** (local, recommended for privacy) — runs models like `gemma4:31b` entirely on your machine
- **Anthropic** — Claude models via API key
- **OpenAI** — GPT models via API key
- **Google Gemini** — Gemini models via API key

You can assign different providers to different cognitive functions (e.g., use a local model for memory tasks and a cloud model for complex reasoning). See `docs/02-PROVIDERS-SETUP.md` for configuration.

---

## What does Chalie do when I'm not talking to it?

Several things, depending on configuration and activity level:

- **Cognitive drift** — During idle periods, Chalie generates spontaneous thoughts via its Default Mode Network (DMN). These may surface as proactive messages or background plan proposals.
- **Memory consolidation** — Episodes are compressed into semantic concepts; memories are decayed.
- **User summary** — A running synthesis of who you are and what matters to you is updated periodically.
- **Persistent tasks** — Background tasks continue executing.
- **World awareness** — Weather, news, and other ambient signals are refreshed in the background.

All background activity is attention-gated: if you're in deep focus, Chalie stays silent.

---

## Can Chalie take actions autonomously?

Yes, within hard limits. Chalie can:
- Execute tasks via its ACT loop using tools
- Schedule reminders and manage lists
- Research topics autonomously via the goal pursuit system
- Generate proactive suggestions and follow-ups

Chalie will **not** take irreversible or destructive actions autonomously. Consequential actions (anything that affects external systems or requires user identity) are paused for confirmation. Silent autonomous handling is the default only for safe, reversible, or informational actions.

---

## What are "tools" in Chalie?

Tools extend Chalie's ability to take action in the world: search, news, weather, code execution, and more. First-party tools are simple Python modules invoked directly in-process. External apps can also expose tools via the interface protocol. Chalie's infrastructure is tool-agnostic: it doesn't know or care what specific tools are installed.

See `docs/09-TOOLS.md` for how tools work and `docs/14-DEFAULT-TOOLS.md` for the tools installed by default.

---

## How do I configure an LLM provider?

1. Start Chalie, create your account at `http://localhost:8081/on-boarding/`, and log in
2. Open Brain at `http://localhost:8081/brain/` → **Settings** → **Providers** → **Add Provider**
3. For Ollama: install from [ollama.ai](https://ollama.ai), pull a model (`ollama pull gemma4:31b`), set endpoint to `http://localhost:11434`
4. For cloud providers: paste your API key — it is encrypted and stored locally

See `docs/02-PROVIDERS-SETUP.md` for full details.

---

## How do I reset or delete Chalie's memory?

Via the REST API or Brain dashboard:
- **Privacy endpoints**: `DELETE /api/privacy/data` — full data wipe
- **Export your data**: `GET /api/privacy/export`

Memories also decay naturally over time without any intervention.

---

## Where does Chalie store its data?

Everything is in a single SQLite database at `backend/data/chalie.db`. No cloud storage, no external databases. You can back it up by copying that file.

---

## Is Docker required?

No. Docker is optional — it's only used for deploying Chalie itself (via the provided Dockerfile and docker-compose.yml). All tools run in-process. The core runtime, voice features, and all cognitive services run natively without Docker.

---

## Can Chalie use my GPU?

Yes. The embedding model runs on ONNX Runtime, which auto-selects the best available execution provider at startup: **CUDA** on NVIDIA GPUs, **CoreML** on Apple Silicon, **CPU** as the fallback. No configuration needed — whatever hardware ORT detects, it uses.

**Running Chalie in Docker with an NVIDIA GPU?** You must pass the GPU through to the container. Two prerequisites on the host:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
   ```bash
   sudo apt install nvidia-container-toolkit
   sudo systemctl restart docker
   ```
2. Add a GPU reservation to your service in `docker-compose.yml`:
   ```yaml
   services:
     chalie:
       image: chalieai/chalie:latest
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 device_ids: ["0"]    # use GPU 0; change index to pick another
                 capabilities: [gpu]
   ```

Verify with `nvidia-smi` inside the container, or check the Chalie log on startup — look for `[EMBEDDING] Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`. If it says CPU only, the passthrough isn't wired up.

Without passthrough the container falls back to CPU inference silently — Chalie still works, just slower.

---

## Does Chalie support voice?

Yes — native speech-to-text (Moonshine Voice, ONNX) and text-to-speech (Kokoro 82M, ONNX) are built in and auto-detect their dependencies on startup. No Docker required. The voice service degrades gracefully (returns 503) if dependencies aren't installed.

---

## What is the Brain dashboard?

The Brain dashboard (`http://localhost:8081/brain/`) is the admin and observability interface. It shows:
- Routing decision distribution
- Memory layer health
- User traits and data graph
- Tool performance metrics
- Personality controls and provider settings

It is read-only for observability panels — settings panels write to Chalie's configuration.
