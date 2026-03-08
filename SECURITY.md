# Security Policy

## Local-First Security Model

Chalie is designed local-first. By default, no user data, conversation history, memories, or configuration leaves your machine. The security model is straightforward: **if data never leaves your system, it cannot be leaked externally**.

The only external network calls Chalie makes are to whichever LLM provider you configure:
- **Ollama** (local) — zero external calls
- **Anthropic / OpenAI / Gemini** — the text of your messages is sent to the provider's API; no memory data, no stored traits, no session history is transmitted

---

## Credential Storage

All LLM provider API keys are:
- Stored in the local SQLite database (`backend/data/chalie.db`)
- Encrypted at rest using AES-256-GCM (via the `DB_ENCRYPTION_KEY` environment variable, auto-generated on first run if not set)
- Never written to plain-text config files or logs
- Never transmitted to Chalie infrastructure (there is no Chalie cloud)

---

## Tool Execution Security

Tools extend Chalie's capabilities but run in isolation:

| Trust Level | Execution | Access |
|-------------|-----------|--------|
| **Sandboxed** (default) | Docker container | No access to host filesystem, network, or Chalie internals |
| **Trusted** (first-party) | Subprocess | Access to its own working directory only; no access to SQLite, MemoryStore, or backend APIs |

Tools have **zero access** to Chalie's internal state. A tool cannot read your conversation history, memory, or traits. It receives only the structured input the dispatcher provides and returns structured output. This is enforced architecturally, not by policy — there is no Chalie API exposed to tool containers.

Webhook endpoints (`/api/tools/webhook/<name>`) use HMAC-SHA256 signatures with replay protection (timestamp window + nonce).

---

## Authentication

- Session cookie-based authentication for the web interface
- API key authentication for programmatic access
- All authenticated endpoints use the `@require_session` decorator
- No default or hardcoded credentials — account password is set during onboarding

---

## Data Scope

- Conversation data is scoped by thread (no cross-thread leakage in context assembly)
- User traits are scoped to the authenticated account
- No multi-tenancy in the default single-user deployment

---

## No Telemetry

Chalie contains no analytics, no error reporting, no usage tracking, and no phone-home behavior of any kind. The codebase contains no calls to external analytics endpoints.

---

## CORS

The Flask app defaults to allowing `localhost` origins only. Before exposing Chalie on a network or behind a reverse proxy, restrict CORS to your expected origin in the configuration.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please open a GitHub issue with the `security` label or contact the maintainers directly. Do not disclose vulnerabilities publicly until they have been addressed.
