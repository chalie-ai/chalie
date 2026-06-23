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

Chalie does not store everything forever. Episodic memories (specific conversation events) decay faster; semantic concepts (distilled knowledge) decay slower. Memories that are reinforced through repeated relevance survive longer. Memories that never become relevant again fade and are eventually deleted — merely reading a memory does not refresh it.

This mirrors how human memory works — and it serves a practical purpose: it prevents Chalie from accumulating an ever-growing pile of outdated, contradictory noise. What persists is what matters.

You can inspect Chalie's memory at any time via the Brain dashboard Memory tab, which shows paginated records across episodes, user facts, and system knowledge.

---

## What LLM providers does Chalie support?

- **Ollama** (local, recommended for privacy) — runs models like `gemma3:4b` entirely on your machine
- **Anthropic** — Claude models via API key
- **OpenAI** — GPT models via API key
- **OpenAI-compatible** — any endpoint speaking the Chat Completions format (Groq, OpenRouter, LM Studio, vLLM, …)
- **Google Gemini** — Gemini models via API key

Three provider roles can be configured independently:

- **Chat (main)** — handles all chat and reasoning turns.
- **Vision** — optional dedicated provider for image understanding; falls back to the main provider when it supports vision.
- **Delegate** — optional dedicated provider for subagent turns (`web_search`, `web_browse`, and other delegated tool work); falls back to the main provider when not set.

Both the Vision and Delegate selectors live inside **Brain → Settings → Providers**. See `docs/02-PROVIDERS-SETUP.md` for configuration.

---

## What does Chalie do when I'm not talking to it?

Several things, depending on configuration and activity level:

- **Cognitive drift** — As part of the subconscious worker tick, Chalie's Default Mode Network (DMN) runs a reflective pass over the user picture and recent episodes. Findings are persisted to the data graph; the DMN itself does not push messages to chat.
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
- Research topics autonomously via the `web_search` / `web_browse` delegate agents
- Generate proactive suggestions and follow-ups

Chalie will **not** take irreversible or destructive actions autonomously. Consequential actions (anything that affects external systems or requires user identity) are paused for confirmation. Silent autonomous handling is the default only for safe, reversible, or informational actions.

---

## What are "tools" in Chalie?

Tools extend Chalie's ability to take action in the world: search, news, weather, code execution, and more. First-party tools are simple Python modules invoked directly in-process. External apps can also expose tools via the interface protocol. Chalie's infrastructure is tool-agnostic: it doesn't know or care what specific tools are installed.

See `docs/09-TOOLS.md` for how tools work and `docs/14-DEFAULT-TOOLS.md` for the tools installed by default.

---

## How do I configure an LLM provider?

1. Start Chalie, create your account at `http://localhost:31025/on-boarding/`, and log in
2. Open Brain at `http://localhost:31025/brain/` → **Settings** → **Providers** → **Add Provider**
3. For Ollama: install from [ollama.ai](https://ollama.ai), pull a model (`ollama pull gemma3:4b`), set endpoint to `http://localhost:11434`
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

Everything is in a single SQLite database at `data/chalie.db` (sibling of `backend/` at the repo root). No cloud storage, no external databases. You can back it up by copying that file.

---

## Is Docker required?

No. Docker is optional — it's only used for deploying Chalie itself (via the provided Dockerfile and docker-compose.yml). All tools run in-process. The core runtime, voice features, and all cognitive services run natively without Docker.

---

## Can Chalie use my GPU?

Yes, at two levels: install time and runtime.

**Install time** — `install.sh` detects the host GPU and installs the matching `onnxruntime` wheel automatically:
- NVIDIA GPU (`nvidia-smi` found): installs `onnxruntime-gpu`
- AMD GPU (`/dev/kfd` + `amdgpu` module): installs `onnxruntime-rocm`
- Everything else: installs `onnxruntime` (CPU)

The GPU wheel is only swapped in after a dry-run confirms it's reachable — machines without access to the GPU package index stay on CPU rather than failing. ORT version is pinned at `1.20.1`. For air-gapped AMD installs, set `ROCM_PIP_INDEX` to a local mirror before running the installer.

Even once a GPU wheel is installed, Chalie verifies it can actually load at boot: if its native libraries are missing — for example the host's CUDA toolkit is absent or a different major version, so `libcudart.so` can't be found — it reinstalls the CPU `onnxruntime` wheel and starts on CPU instead of failing, logging a hint that names the exact version mismatch. The diagnosis appears in the **Cognition → Errors** panel.

**Runtime** — all ONNX sessions are constructed through `backend/services/onnx_session.py`, which auto-selects the best available execution provider: **CUDA** on NVIDIA GPUs, **CoreML** on Apple Silicon, **CPU** as the fallback. No configuration needed.

> **Mac caveat:** if any weight tensor in the model has a dimension larger than **16384** (the Metal 2D-texture ceiling — applies to every Mac, Intel through M4), `onnx_session.py` drops `CoreMLExecutionProvider` automatically and falls back to CPU. CoreML would otherwise partition the graph across ~177 sub-graphs and balloon virtual memory by ~21 GB. The default `gte-modernbert-base` trips this limit because its vocab embedding is `{50368, 768}`. You'll see `[EMBEDDING] Dropped CoreMLExecutionProvider: model has dim > 16384` in the log when this fires. To force CoreML anyway (not recommended — will almost certainly OOM on <64 GB Macs), pass `providers=["CoreMLExecutionProvider", ...]` explicitly.

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

Yes — native speech-to-text (Moonshine, ONNX) and text-to-speech (Kokoro 82M, ONNX). Voice is **off by default**: turning it on in Brain → Settings downloads the dependencies and models on demand. The voice service degrades gracefully (returns 503) when they aren't installed, and the UI hides voice controls automatically.

---

## What is the Brain dashboard?

The Brain dashboard (`http://localhost:31025/brain/`) is the admin and observability interface. It shows:
- Routing decision distribution
- Memory layer health
- User traits and data graph
- Personality controls and provider settings

It is read-only for observability panels — settings panels write to Chalie's configuration.
