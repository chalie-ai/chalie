# Provider Configuration

All provider setup is done in **Brain → Settings → Providers → Add Provider**.

---

## Supported Providers

| Platform | Local? | API Key? | Notes |
|----------|--------|----------|-------|
| **Ollama** | Yes | No | Runs on your machine; no data leaves |
| **Anthropic** | No | Yes | Claude models — [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI** | No | Yes | GPT models — [platform.openai.com](https://platform.openai.com) |
| **Google Gemini** | No | Yes | Gemini models — [ai.google.dev](https://ai.google.dev) |
| **OpenAI-Compatible** | Either | Yes | Any endpoint that speaks the OpenAI Chat Completions format (Groq, DeepSeek, Together, OpenRouter, LM Studio, vLLM, MiniMax, etc.) |
| **Codex CLI** | Yes | No (uses existing Codex auth) | Piggybacks on an installed `codex` CLI subscription via JSON-RPC over stdio |

---

## Adding a Provider

Each provider needs a **name** (any label), a **platform**, a **model ID**, and for cloud providers an **API key**.

**Ollama:** set the host to `http://localhost:11434`. Chalie queries your Ollama instance and populates the model picker automatically. Hit **Refresh** after `ollama pull <model>`.

**Cloud providers:** paste your API key and enter the exact model ID from your provider's documentation. Click **Test Connection** to verify before saving.

**OpenAI-Compatible:** set the base URL to your provider's endpoint, enter your API key and model ID. Use this for any provider not listed above.

**Codex CLI:** requires the `codex` binary to be installed and authenticated on the host. The Providers tab detects CLI availability automatically on load — the platform tab only appears when `codex` is found. No API key entry is needed; authentication is inherited from the CLI's own OAuth session. The default model is `o4-mini`; enter any model ID supported by your Codex subscription. v1 passes conversations in pass-through mode (no tool bridging).

---

## Troubleshooting

**"Provider connection failed"**
- Ollama: confirm Ollama is running (`ollama serve`) and the host URL is correct.
- Cloud: check your API key and that the provider endpoint is reachable.
- Codex CLI: confirm `codex` is on `$PATH` (`which codex`) and that your CLI session is authenticated (`codex auth login`).

**"API key is invalid"**
- Copy the key again — watch for leading/trailing spaces.
- Verify the key has not expired or been revoked.

**Model not found**
- Ollama: run `ollama pull <model-name>` first, then Refresh in Brain.
- Cloud: use the exact model ID from your provider's docs — names change.

---

## REST API

For scripting and automation.

**List providers**
```bash
curl http://localhost:31025/providers \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Create provider**
```bash
curl -X POST http://localhost:31025/providers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"name": "my-claude", "platform": "anthropic", "model": "claude-haiku-4-5-20251001", "api_key": "sk-ant-..."}'
```

**Delete provider**
```bash
curl -X DELETE http://localhost:31025/providers/{id} \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Assign provider to a job**
```bash
curl -X PUT http://localhost:31025/providers/jobs/frontal-cortex \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"provider_id": 1}'
```

**Detect CLI providers**
```bash
curl -X POST http://localhost:31025/providers/detect-cli \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cli": "codex"}'
```
Returns `{"available": true, "models": [...]}` when a supported CLI binary is found and authenticated, `{"available": false}` otherwise. The Brain Providers tab calls this on mount to conditionally show CLI platform tabs.
