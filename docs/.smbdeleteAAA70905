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

**Add Provider** opens a short wizard that only ever asks for what the chosen provider actually needs:

1. **Pick a provider.** A grid of popular providers appears — Ollama, Anthropic, OpenAI, Google Gemini, DeepSeek, MiniMax, NVIDIA, Groq, Mistral, OpenRouter, and more. Pick the closest match, or choose **Custom (OpenAI-compatible)** for any endpoint not in the list.
2. **Host auto-fills.** Selecting a provider pre-fills its base URL where one applies (e.g. MiniMax → `https://api.minimax.io/v1`, Ollama → `http://localhost:11434`). Native APIs (Anthropic, OpenAI, Gemini) need no host, so that field is skipped.
3. **Enter your API key.** The key field appears only for providers that require one. Ollama runs locally and unauthenticated, so the key step is skipped entirely.
4. **Pick a model.** As soon as the host and key (where required) are present, Chalie fetches the provider's live model list and fills the picker — for every platform, not just Ollama. Choose your model, give the provider a **name** (any label), and **Save**. Use **Test Connection** first if you want to verify the credentials.

Each saved provider maps to one of the platforms above and stores the model you selected.

---

## Troubleshooting

**"Provider connection failed"**
- Ollama: confirm Ollama is running (`ollama serve`) and the host URL is correct.
- Cloud: check your API key and that the provider endpoint is reachable.

**"API key is invalid"**
- Copy the key again — watch for leading/trailing spaces.
- Verify the key has not expired or been revoked.

**Model not found**
- Ollama: run `ollama pull <model-name>` first, then reopen the provider so the model list refetches.
- Cloud: the picker is populated live from the provider — if a model is missing, confirm your key has access to it.

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

Chalie resolves three provider roles at runtime:

| Role | Settings key | Fallback | REST endpoint |
|------|-------------|---------|---------------|
| **Chat (main)** | `selected_provider_id` | — | `GET/PUT /providers/selected` |
| **Vision** | `vision_provider_id` | Falls back to the main provider when it supports vision; shows "Disabled" only when no vision-capable provider exists | `GET/PUT /providers/vision` |
| **Delegate** | `delegate_provider_id` | Falls back to the main provider (no "Disabled" state) | `GET/PUT /providers/delegate` |

The Vision and Delegate selectors appear inside **Brain → Settings → Providers → LLM Providers** — Vision pre-filtered to vision-capable providers, Delegate showing the full provider list plus "Use main provider".

**Deletion guard.** A provider that is currently assigned as the main, vision, or delegate provider cannot be deleted — the API returns **HTTP 409**. Clear or reassign the role first, then delete.

`POST /providers/list-models` fetches the live model list for a given platform and credentials, populating the wizard's model picker for every provider type. `GET /providers/catalog` returns the curated preset list the wizard's provider grid is built from.

API keys are encrypted at rest (AES-256-GCM) in the local SQLite database and never leave your machine.
