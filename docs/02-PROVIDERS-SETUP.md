# Provider Configuration Setup

After deploying the application, you can configure LLM providers through the web interface.

## Overview

Providers define which LLM backends (Ollama, Anthropic, OpenAI, Gemini, OpenAI-compatible third parties, etc.) are available to the system. All provider configuration is stored in the SQLite database.

## Quick Start

### 1. Start the Application

```bash
python backend/run.py
```

The web interface will be available at `http://localhost:8081`.

### 2. Configure in Brain

After creating your account at `http://localhost:8081/on-boarding/`, configure providers in Brain:

1. Open `http://localhost:8081/brain/`
2. Go to **Settings** → **Providers**
3. Click **Add Provider**

### 4. Provider Options

Each provider needs a **name** (any label you want), the **platform**, a **model ID**, and for cloud providers, an **API key**.

#### Local (free, private)

**Ollama** — Runs entirely on your machine. No API key needed.
- **Platform**: Ollama
- **Host**: `http://localhost:11434`
- **Model**: Any model you've pulled (e.g., `qwen:8b`). Run `ollama list` to see available models.

#### Cloud

| Platform | API Key Source | Example Model |
|----------|---------------|---------------|
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com) | Any Claude model ID |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) | Any GPT model ID |
| **Google Gemini** | [ai.google.dev](https://ai.google.dev) | Any Gemini model ID |
| **OpenAI-Compatible** | Your provider's dashboard (MiniMax, Groq, DeepSeek, Together, OpenRouter, LM Studio, vLLM, etc.) | Any model ID supported by that endpoint |

Use the exact model ID from your provider's documentation — model names change frequently.

### 5. Save and Test

After entering provider details, click **Save** or **Test Connection** to verify the configuration works.

## Supported Platforms

| Platform | Local? | Requires API Key? | Notes |
|---|---|---|---|
| **Ollama** | Yes | No | Local inference, requires Ollama running on machine |
| **Anthropic** | No | Yes | Claude API from Anthropic |
| **OpenAI** | No | Yes | GPT models from OpenAI |
| **Google Gemini** | No | Yes | Gemini models from Google |
| **OpenAI-Compatible** | Either | Yes | Generic endpoint — bring your own base URL + API key. Use for MiniMax, Groq, DeepSeek, Together, OpenRouter, LM Studio, vLLM, or any other provider that speaks the OpenAI Chat Completions wire format. |

## Troubleshooting

### "Provider connection failed"
- **For Ollama**: Ensure Ollama is running (`ollama serve`) and the host URL is correct (usually `http://localhost:11434`)
- **For cloud providers**: Double-check that your API key is correct and has the necessary permissions
- **Network issues**: Verify your internet connection and that the provider endpoint is accessible

### "API key is invalid"
- Check that you copied the API key correctly (no extra spaces)
- Verify the key hasn't expired or been revoked
- For Anthropic: Get key from https://console.anthropic.com
- For OpenAI: Get key from https://platform.openai.com/api-keys
- For Gemini: Get key from https://ai.google.dev

### Model not found
- For Ollama: Run `ollama pull <model-name>` to download the model first
- For cloud providers: Verify the exact model name matches what the provider offers

---

## REST API Reference

For programmatic provider configuration (scripts, automation, CI).

### Create Provider
```bash
curl -X POST http://localhost:8081/providers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "name": "claude-haiku",
    "platform": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "api_key": "sk-ant-...",
    "timeout": 120
  }'
```

### List All Providers
```bash
curl http://localhost:8081/providers \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Update Provider
```bash
curl -X PUT http://localhost:8081/providers/{id} \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"api_key": "sk-..."}'
```

### Delete Provider
```bash
curl -X DELETE http://localhost:8081/providers/{id} \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Assign Provider to Job
```bash
curl -X PUT http://localhost:8081/providers/jobs/frontal-cortex \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"provider_id": 1}'
```

## Security Best Practices

### Protecting API Keys
- **Never** commit API keys to version control
- **Never** share API keys in logs or error messages
- Use environment variables or secure secret management for production
- Rotate API keys regularly
- Use database encryption for sensitive columns in production

### Network Security
- Keep Ollama instances local or behind a firewall
- Use HTTPS/TLS for remote API connections
- Restrict file access to the SQLite database
- Enable CORS appropriately for your deployment

## Embedding Models

Chalie runs embeddings locally via ONNX (`gte-modernbert-base`, 768 dimensions). The model downloads automatically on first run (~300MB) and requires no configuration. No LLM provider is used for embeddings.
