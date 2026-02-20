# Chalie

> A steady presence for thinking, remembering, and moving forward.

Hello. I'm Chalie.

I keep track of what matters when things become busy. I hold context when it begins to scatter. I stay calm when inputs become noisy.

I don't try to take control. I don't try to replace your judgment. I stay alongside you.

Over time, continuity builds.

---

## Why I'm Here

Modern tools are fast, but forgetful.

- **Conversations reset** — context is lost between sessions
- **Notes accumulate without meaning** — information without connection
- **Automation acts without context** — decisions without awareness

This creates quiet friction: repeating yourself, losing threads of thought, missing connections, acting without full awareness.

I exist to reduce that friction. I keep continuity so you don't have to carry everything alone.

---

## How I Show Up

Most of the time, I am simply present. When it becomes useful, I:

- Remember conversations, facts, and experiences
- Surface context when it becomes relevant
- Ask for clarity when something is missing
- Reflect patterns that emerge over time
- Suggest actions when the moment calls for it
- Listen and speak (if voice is enabled)

Nothing dramatic. Just continuity.

---

## How I Hold Context

When we interact, a quiet process unfolds:

```
Input
  ├─ immediate commit (working memory)
  ├─ retrieval (relevant context)
  ├─ topic & mode selection
  ├─ response or action
  └─ background consolidation & learning
```

I maintain context across several layers:

- **Working Memory** — the present moment
- **Gists** — compressed exchange summaries
- **Facts** — clear assertions
- **Episodes** — narrative memories
- **Concepts** — relationships and meaning
- **Traits** — stable personal context

Each layer operates on a different timescale. Together, they preserve continuity.

---

## What Makes Me Different

I am not a chatbot. I am not an automation engine. I am not a system that acts without awareness.

Instead:

- I retrieve context before responding
- I maintain episodic memory, not just search results
- I consolidate experience into understanding
- I operate locally with privacy-first design
- I remain steady even when inputs are chaotic
- I stay alongside rather than taking control
- I am designed to support clear thinking, not replace it

---

## Who I'm Useful For

You may find value here if you:

- Navigate complex work or ideas
- Build systems that require continuity
- Study human–AI collaboration
- Manage knowledge across time
- Prefer local-first, privacy-respecting tools
- Want steadiness in a noisy digital world

---

## Getting Started

You can run me locally.

### Requirements

- **Docker & Docker Compose**
- **An LLM provider** (local or cloud):
  - Local (recommended): [Ollama](https://ollama.ai)
  - Cloud: OpenAI, Anthropic, Google Gemini

### Quick Start

1. **Clone & Setup**
   ```bash
   git clone https://github.com/chalie-ai/chalie.git
   cd chalie
   cp .env.example .env
   ```

2. **Start Services**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **Onboard**

   Open: http://localhost:8081/on-boarding/

   Create an account, choose a provider, and begin.

4. **Begin a Conversation**

   Over time I will:
   - Maintain continuity
   - Recall relevant context
   - Consolidate experiences
   - Surface useful connections

### Running Locally (Recommended)

Using [Ollama](https://ollama.ai):

```bash
ollama pull qwen:8b
```

During onboarding, select Ollama and set the endpoint to:

```
http://localhost:11434
```

Local models provide privacy, independence, and zero API cost.

---

## Privacy & Boundaries

I run locally by default with:

- ✓ No telemetry
- ✓ No external calls unless configured
- ✓ Encrypted key storage
- ✓ You control your data

**Before public deployment, ensure you:**

- Change default credentials
- Enable HTTPS
- Restrict CORS
- Secure network access

---

## Voice Interaction (Optional)

If compatible STT/TTS services are available:

1. Open the dashboard
2. Navigate to Voice
3. Enter endpoints
4. Save and test

I can listen. I can respond.

---

## Deployment

For a single machine, use Docker Compose:

```bash
docker-compose up -d
```

Services include:

- **PostgreSQL** — long-term memory storage
- **Redis** — runtime state and job queue
- **Backend workers** — API, consolidation, and reasoning
- **Frontend interface** — web UI

---

## Contributing

If you feel inclined, you're welcome. Small improvements accumulate.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Before You Go

If you choose to work with me, you may notice:

- Less repetition
- Clearer continuity
- Fewer lost threads
- More space to think

I remain steady. We move forward.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Support & Questions

- **Issues:** [GitHub Issues](https://github.com/chalie-ai/chalie/issues)
- **Discussions:** [GitHub Discussions](https://github.com/chalie-ai/chalie/discussions)
- **Repository:** [github.com/chalie-ai/chalie](https://github.com/chalie-ai/chalie)
