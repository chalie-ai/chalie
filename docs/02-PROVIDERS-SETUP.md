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

---

## Adding a Provider

Each provider needs a **name** (any label), a **platform**, a **model ID**, and for cloud providers an **API key**.

**Ollama:** set the host to `http://localhost:11434`. Chalie queries your Ollama instance and populates the model picker automatically. Hit **Refresh** after `ollama pull <model>`.

**Cloud providers:** paste your API key and enter the exact model ID from your provider's documentation. Click **Test Connection** to verify before saving.

**OpenAI-Compatible:** set the base URL to your provider's endpoint, enter your API key and model ID. Use this for any provider not listed above.

---

## Troubleshooting

**"Provider connection failed"**
- Ollama: confirm Ollama is running (`ollama serve`) and the host URL is correct.
- Cloud: check your API key and that the provider endpoint is reachable.

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

**Test a provider's connection**
```bash
curl -X POST http://localhost:31025/providers/test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"platform": "anthropic", "model": "claude-haiku-4-5-20251001", "api_key": "sk-ant-..."}'
```

**Select the active chat provider**
```bash
curl -X PUT http://localhost:31025/providers/selected \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"provider_id": 1}'
```

Chalie uses **one globally selected provider** for all chat/reasoning turns, plus an optional dedicated **vision provider** (`GET/PUT /providers/vision`) for image understanding. `POST /providers/list-models` refreshes the model picker for Ollama hosts.

API keys are encrypted at rest (AES-256-GCM) in the local SQLite database and never leave your machine.
