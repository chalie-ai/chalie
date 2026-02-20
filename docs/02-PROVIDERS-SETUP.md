# Provider Configuration Setup

After deploying the application, you can configure LLM providers through the web interface.

## Overview

Providers define which LLM backends (Ollama, Anthropic, OpenAI, Gemini, etc.) are available to the system. All provider configuration is stored in the PostgreSQL database.

## Quick Start

### 1. Start the Application

```bash
docker-compose up -d
```

The web interface will be available at `http://localhost:8081`.

### 2. Access Provider Settings

1. Open http://localhost:8081 in your browser
2. Navigate to the **Settings** or **Providers** section (usually in the dashboard)
3. Click **Add Provider** or **Configure Provider**

### 3. Add LLM Providers

Fill in the provider form with your chosen provider's information:

#### For Local Runtime

**Ollama**
- **Name**: Any identifier (e.g., `ollama-local`, `local-model`, etc.)
- **Platform**: Ollama
- **Model**: Your chosen model (e.g., `qwen:8b`, `mistral:latest`, `llama2:latest`, etc.)
- **Host**: `http://localhost:11434`

#### For Cloud Runtime

Choose one or more of these options:

**Anthropic Claude**
- **Name**: Any identifier (e.g., `claude-haiku`, `claude-sonnet`, etc.)
- **Platform**: Anthropic
- **Model**: Claude model name (e.g., `claude-haiku-4-5-20251001`, `claude-sonnet-4-20250514`, etc.)
- **API Key**: Your Anthropic API key (from console.anthropic.com)

**OpenAI**
- **Name**: Any identifier (e.g., `gpt-4o`, `gpt-4-turbo`, etc.)
- **Platform**: OpenAI
- **Model**: OpenAI model name (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`, etc.)
- **API Key**: Your OpenAI API key (from platform.openai.com)

**Google Gemini**
- **Name**: Any identifier (e.g., `gemini-flash`, `gemini-pro`, etc.)
- **Platform**: Gemini
- **Model**: Gemini model name (e.g., `gemini-2.0-flash`, `gemini-1.5-pro`, etc.)
- **API Key**: Your Google Gemini API key (from ai.google.dev)

### 4. Save and Test

After entering provider details, click **Save** or **Test Connection** to verify the configuration works.

## Supported Platforms

| Platform | Local? | Requires API Key? | Notes |
|---|---|---|---|
| **Ollama** | Yes | No | Local inference, requires Ollama running on machine |
| **Anthropic** | No | Yes | Claude API from Anthropic |
| **OpenAI** | No | Yes | GPT models from OpenAI |
| **Google Gemini** | No | Yes | Gemini models from Google |

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

## Advanced: REST API (for Developers)

For programmatic provider configuration, you can use the REST API directly.

### Create Provider via API
```bash
curl -X POST http://localhost:8080/providers \
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
curl http://localhost:8080/providers \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Update Provider
```bash
curl -X PUT http://localhost:8080/providers/{id} \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"api_key": "sk-..."}'
```

### Delete Provider
```bash
curl -X DELETE http://localhost:8080/providers/{id} \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Assign Provider to Job
```bash
curl -X PUT http://localhost:8080/providers/jobs/frontal-cortex \
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
- Restrict database access with network policies
- Enable CORS appropriately for your deployment

## Embedding Models

If you need to use embedding models (e.g., `embeddinggemma` from Ollama):

1. Configure the embedding provider in the UI or via API
2. Set the model name (e.g., `embeddinggemma:latest`)
3. Set dimensions to match the model's output (usually 768)
4. Ensure the system is configured to use this provider for embeddings
