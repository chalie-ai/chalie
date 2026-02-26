# Chalie

**What is Chalie?**

Chalie is a human-in-the-loop cognitive assistant that helps you think, remember, and act across your digital life. It combines memory consolidation, semantic reasoning, and proactive assistance into a unified system designed to augment human cognition rather than replace it.

## Features

- **Memory Hierarchy**: Working memory, gists, facts, episodes, and concepts — each with decay and reinforcement
- **Context Assembly**: Intelligent retrieval of relevant memories based on semantic similarity and activation
- **Cognitive Modes**: RESPOND, ACT, CLARIFY, ACKNOWLEDGE — adaptive routing based on conversation context
- **Experience Consolidation**: Tool results are automatically integrated into episodic memory via novelty gates
- **Proactive Assistance**: Spontaneous thoughts and outreach during idle time (DMN-inspired)
- **Voice I/O**: Optional Text-to-Speech and Speech-to-Text via OpenAI-compatible endpoints

## Architecture

```
User Input
  ├─ Immediate Commit (working memory, Redis)
  ├─ Retrieval (context assembly from all memory layers)
  ├─ Classification & Routing (topic + mode selection)
  ├─ LLM Generation (mode-specific prompt, context injection)
  ├─ Post-Response Commit (working memory, PostgreSQL logging)
  └─ Async Background (memory chunking, episodic consolidation, semantic extraction)
```

**Memory Layers**:
- **Working Memory** (Redis, 4 turns, 24h TTL)
- **Gists** (Redis, 30min TTL) — compressed exchange summaries
- **Facts** (Redis, 24h TTL) — atomic key-value assertions
- **Episodes** (PostgreSQL + pgvector) — narrative units with decay
- **Concepts** (PostgreSQL + pgvector) — knowledge nodes and relationships
- **User Traits** (PostgreSQL) — personal facts, category-specific decay
- **Lists** (PostgreSQL) — deterministic list management (shopping, to-do, chores)

## Prerequisites

- Docker & Docker Compose
- An LLM provider:
  - **Local**: [Ollama](https://ollama.ai) (recommended for development)
  - **API**: OpenAI, Anthropic, or Google Gemini

## Quick Start

### 1. Clone & Start

```bash
git clone https://github.com/chalie-ai/chalie.git
cd chalie
docker-compose build && docker-compose up -d
```

Check status:
```bash
docker-compose logs -f backend
docker-compose ps
```

### 2. Onboard

Open http://localhost:8081/on-boarding/ in your browser.

- **Create Account**: Set a password
- **Configure Provider**: Choose your LLM provider:
  - **For Local**: Ollama (install from ollama.ai, select any available model, point to `http://localhost:11434`)
  - **For Cloud**: OpenAI, Anthropic, or Google Gemini (requires API key)
- **Redirect**: After setup, you'll be redirected to the chat interface

### 3. Chat

Start chatting! Chalie will:
- Remember context across exchanges
- Consolidate memories in the background
- Suggest actions when relevant
- Execute tools if configured

## LLM Provider Options

### For Local Runtime

**Ollama** — Run models locally on your machine (no API costs, privacy-first)

1. Install from [ollama.ai](https://ollama.ai)
2. Pull any available model: `ollama pull <model-name>`
   - Popular options: `qwen:8b`, `mistral:latest`, `llama2:latest`, etc.
3. Ensure Ollama is running (`ollama serve`)
4. In onboarding, select **Ollama** and set endpoint to `http://localhost:11434`

### For Cloud Runtime

Choose based on your preference:

**OpenAI**
- Get API key from [platform.openai.com](https://platform.openai.com)
- Models available: GPT-4, GPT-4o, etc.
- In onboarding, select **OpenAI** and paste your key

**Anthropic**
- Get API key from [console.anthropic.com](https://console.anthropic.com)
- Models available: Claude Haiku, Claude Sonnet, Claude Opus
- In onboarding, select **Anthropic** and paste your key

**Google Gemini**
- Get API key from [ai.google.dev](https://ai.google.dev)
- Models available: Gemini Pro, Gemini Flash, etc.
- In onboarding, select **Gemini** and paste your key

## Voice (Built-in)

Voice is built into Chalie as a local Docker service — no configuration needed. When the `voice` container is running, the mic button and speaker buttons appear automatically in the chat interface. When it's not running, they stay hidden. Zero setup, zero settings.

- **STT**: faster-whisper (`small` model) — records from mic, transcribes to text
- **TTS**: KittenTTS Mini (80M params) — speaks Chalie's responses aloud
- **Verify**: `curl localhost:8081/voice/health` should return `{"status": "ok"}`

## Security

- **Local by Default**: Runs entirely on your machine. No cloud uploads unless you configure external providers.
- **API Keys**: Managed via master account auth. Stored securely in PostgreSQL with encryption.
- **CORS**: Defaults to `localhost`. Restrict before exposing publicly.
- **Default Credentials**: Postgres password is `chalie` — **change before production use**.
- **No Telemetry**: Zero tracking or external calls (except to your configured LLM/voice providers).

## Architecture Deep Dive

See the source code for implementation details:

- `backend/consumer.py` — Worker supervisor
- `backend/services/frontal_cortex_service.py` — LLM response generation
- `backend/services/episodic_retrieval_service.py` — Memory search with hybrid ranking
- `backend/services/semantic_consolidation_service.py` — Concept extraction
- `backend/workers/` — Background processing pipeline

## Deployment

### Docker Compose (Single Machine)

The included `docker-compose.yml` runs everything:

```bash
docker-compose up -d
```

Services:
- **postgres**: Vector database (pgvector extension)
- **redis**: Session & queue storage
- **backend**: Python Flask + workers
- **frontend**: Vanilla JS UI (nginx)

### Production

Before exposing to a network:

1. Change `POSTGRES_PASSWORD` in `.env`
2. Set HTTPS in nginx or reverse proxy
3. Restrict CORS origins in `frontend/interface/api.js`
4. Use strong API keys
5. Enable firewall rules

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

**Support & Questions**

- Issues: [GitHub Issues](https://github.com/chalie-ai/chalie/issues)
- Discussions: [GitHub Discussions](https://github.com/chalie-ai/chalie/discussions)
