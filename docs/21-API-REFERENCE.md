# Chalie API Reference — REST Endpoints, Request/Response Formats, and Authentication

This comprehensive API reference documents all REST endpoints exposed by the Chalie backend. Each endpoint includes HTTP methods, request/response schemas, authentication requirements, and example usage to help you integrate with or extend the Chalie platform programmatically.

---

## Table of Contents

1. [Authentication](#authentication)
2. [User Management](#user-management)
3. [Conversations & Messages](#conversations--messages)
4. [Memories](#memories)
5. [Providers](#providers)
6. [Tools](#tools)
7. [System & Health](#system--health)

---

## Authentication

All authenticated endpoints require a valid session cookie (`session`) or Bearer token in the `Authorization` header.

### Login

**Endpoint:** `POST /api/auth/login`

Authenticate with username and password to receive a session token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "id": "uuid-string",
    "username": "string"
  }
}
```

**Response (401 Unauthorized):**
```json
{
  "success": false,
  "error": "Invalid credentials"
}
```

---

### Logout

**Endpoint:** `POST /api/auth/logout`

Invalidate the current session.

**Headers:**
- `Cookie: session=<session_token>`

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

---

### Register

**Endpoint:** `POST /api/auth/register`

Create a new user account.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Registration successful",
  "user": {
    "id": "uuid-string",
    "username": "string"
  }
}
```

---

## User Management

### Get Current User

**Endpoint:** `GET /api/user`

Retrieve the authenticated user's profile.

**Response (200 OK):**
```json
{
  "id": "uuid-string",
  "username": "string",
  "created_at": "ISO-8601 timestamp"
}
```

---

### Update User Settings

**Endpoint:** `PUT /api/user/settings`

Update user preferences and settings.

**Request Body:**
```json
{
  "theme": "dark|light",
  "default_provider_id": "uuid-string",
  "notifications_enabled": true
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "settings": { ... }
}
```

---

## Conversations & Messages

### List Conversations

**Endpoint:** `GET /api/conversations`

Retrieve all conversations for the authenticated user.

**Query Parameters:**
- `limit` (optional): Maximum number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)
- `sort_by` (optional): Sort field (`created_at`, `updated_at`)
- `order` (optional): Sort order (`asc`, `desc`)

**Response (200 OK):**
```json
{
  "conversations": [
    {
      "id": "uuid-string",
      "title": "string",
      "created_at": "ISO-8601 timestamp",
      "updated_at": "ISO-8601 timestamp",
      "message_count": 42
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

---

### Create Conversation

**Endpoint:** `POST /api/conversations`

Create a new conversation.

**Request Body:**
```json
{
  "title": "string (optional)"
}
```

**Response (201 Created):**
```json
{
  "id": "uuid-string",
  "title": "New Conversation",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp"
}
```

---

### Get Conversation Details

**Endpoint:** `GET /api/conversations/{conversation_id}`

Retrieve a specific conversation with its messages.

**Response (200 OK):**
```json
{
  "id": "uuid-string",
  "title": "string",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "messages": [
    {
      "id": "uuid-string",
      "role": "user|assistant",
      "content": "string",
      "created_at": "ISO-8601 timestamp"
    }
  ]
}
```

---

### Delete Conversation

**Endpoint:** `DELETE /api/conversations/{conversation_id}`

Permanently delete a conversation.

**Response (204 No Content):**
*(Empty response body)*

---

### Send Message

**Endpoint:** `POST /api/conversations/{conversation_id}/messages`

Send a message to an existing conversation and receive the assistant's response.

**Request Body:**
```json
{
  "content": "string",
  "provider_id": "uuid-string (optional, uses default if not specified)",
  "use_tools": true,
  "stream": false
}
```

**Response (201 Created):**
```json
{
  "message": {
    "id": "uuid-string",
    "role": "user",
    "content": "string",
    "created_at": "ISO-8601 timestamp"
  },
  "response": {
    "id": "uuid-string",
    "role": "assistant",
    "content": "Assistant response text...",
    "tool_calls": [
      {
        "name": "tool_name",
        "input": {},
        "output": "result"
      }
    ],
    "created_at": "ISO-8601 timestamp"
  }
}
```

---

### Stream Message (SSE)

**Endpoint:** `POST /api/conversations/{conversation_id}/messages/stream`

Send a message and receive streaming responses via Server-Sent Events.

**Request Body:** Same as Send Message with `"stream": true`

**Response Content-Type:** `text/event-stream`

**Event Format:**
```
event: token
data: {"content": "Hello"}

event: tool_call
data: {"name": "calculator", "input": {"expression": "2+2"}}

event: complete
data: {"message_id": "uuid-string"}
```

---

## Memories

### List Memories

**Endpoint:** `GET /api/memories`

Retrieve all memories for the authenticated user.

**Query Parameters:**
- `type` (optional): Filter by type (`fact`, `preference`, `skill`)
- `limit`: Maximum results (default: 50)
- `offset`: Pagination offset (default: 0)

**Response (200 OK):**
```json
{
  "memories": [
    {
      "id": "uuid-string",
      "content": "string",
      "type": "fact|preference|skill",
      "tags": ["tag1", "tag2"],
      "created_at": "ISO-8601 timestamp",
      "updated_at": "ISO-8601 timestamp"
    }
  ],
  "total": 25,
  "limit": 50,
  "offset": 0
}
```

---

### Create Memory

**Endpoint:** `POST /api/memories`

Create a new memory entry.

**Request Body:**
```json
{
  "content": "string",
  "type": "fact|preference|skill",
  "tags": ["array of strings"]
}
```

**Response (201 Created):**
```json
{
  "id": "uuid-string",
  "content": "string",
  "type": "fact",
  "tags": [],
  "created_at": "ISO-8601 timestamp"
}
```

---

### Get Memory Details

**Endpoint:** `GET /api/memories/{memory_id}`

Retrieve a specific memory.

**Response (200 OK):** Same as Create Memory response

---

### Update Memory

**Endpoint:** `PUT /api/memories/{memory_id}`

Update an existing memory.

**Request Body:**
```json
{
  "content": "string",
  "type": "fact|preference|skill",
  "tags": ["array of strings"]
}
```

**Response (200 OK):** Updated memory object

---

### Delete Memory

**Endpoint:** `DELETE /api/memories/{memory_id}`

Permanently delete a memory.

**Response (204 No Content):** *(Empty response body)*

---

## Providers

### List Providers

**Endpoint:** `GET /api/providers`

Retrieve all configured LLM providers.

**Response (200 OK):**
```json
{
  "providers": [
    {
      "id": "uuid-string",
      "name": "string",
      "type": "ollama|anthropic|openai|gemini",
      "model": "string",
      "is_default": false,
      "created_at": "ISO-8601 timestamp"
    }
  ]
}
```

---

### Create Provider

**Endpoint:** `POST /api/providers`

Configure a new LLM provider.

**Request Body (Ollama example):**
```json
{
  "name": "My Ollama",
  "type": "ollama",
  "model": "qwen:8b",
  "config": {
    "host": "http://localhost:11434"
  },
  "is_default": true
}
```

**Request Body (Anthropic example):**
```json
{
  "name": "Claude Pro",
  "type": "anthropic",
  "model": "claude-3-opus-20240229",
  "config": {
    "api_key": "sk-ant-..."
  },
  "is_default": true
}
```

**Response (201 Created):** Provider object with `id` field added

---

### Get Provider Details

**Endpoint:** `GET /api/providers/{provider_id}`

Retrieve a specific provider configuration.

**Response (200 OK):** Full provider object including config (sensitive fields may be masked)

---

### Update Provider

**Endpoint:** `PUT /api/providers/{provider_id}`

Update an existing provider configuration.

**Request Body:** Same as Create Provider (all fields optional except `id`)

**Response (200 OK):** Updated provider object

---

### Delete Provider

**Endpoint:** `DELETE /api/providers/{provider_id}`

Remove a provider configuration.

**Response (204 No Content):** *(Empty response body)*

---

### Test Provider Connection

**Endpoint:** `POST /api/providers/{provider_id}/test`

Test the connection and model availability for a provider.

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Connection successful",
  "latency_ms": 145,
  "model_available": true
}
```

**Response (400 Bad Request - on failure):**
```json
{
  "success": false,
  "error": "Connection refused: Ollama service not running"
}
```

---

## Tools

### List Tools

**Endpoint:** `GET /api/tools`

Retrieve all available tools.

**Query Parameters:**
- `status` (optional): Filter by status (`installed`, `available`, `failed`)

**Response (200 OK):**
```json
{
  "tools": [
    {
      "id": "uuid-string",
      "name": "communicate-tool",
      "description": "string",
      "version": "1.0.0",
      "status": "installed|available|failed",
      "schema": { ... },
      "created_at": "ISO-8601 timestamp"
    }
  ]
}
```

---

### Get Tool Details

**Endpoint:** `GET /api/tools/{tool_id}`

Retrieve detailed information about a specific tool including its schema.

**Response (200 OK):** Full tool object with complete schema definition

---

### Install Tool

**Endpoint:** `POST /api/tools/install`

Install a new tool from the marketplace or URL.

**Request Body:**
```json
{
  "name": "tool-name",
  "source_url": "https://github.com/owner/repo/releases/download/v1/tool.json"
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "message": "Tool installation started",
  "tool_id": "uuid-string"
}
```

---

### Uninstall Tool

**Endpoint:** `DELETE /api/tools/{tool_id}`

Remove an installed tool.

**Response (204 No Content):** *(Empty response body)*

---

## System & Health

### Health Check

**Endpoint:** `GET /api/health`

Check the health status of the Chalie server.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "database_connected": true,
  "providers_count": 3,
  "tools_installed": 5
}
```

---

### System Info

**Endpoint:** `GET /api/system/info`

Retrieve system information and configuration.

**Response (200 OK):**
```json
{
  "version": "1.0.0",
  "python_version": "3.12.0",
  "platform": "linux",
  "data_directory": "/home/user/.chalie/data",
  "database_path": "/home/user/.chalie/data/chalie.db",
  "docker_available": true,
  "voice_enabled": false
}
```

---

### Database Backup

**Endpoint:** `POST /api/system/backup`

Create a backup of the SQLite database.

**Response (200 OK):**
```json
{
  "success": true,
  "backup_path": "/home/user/.chalie/data/backups/chalie_2026-03-07_143022.db",
  "size_bytes": 524288
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

**Response (4xx/5xx):**
```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Authentication is required for this endpoint |
| `INVALID_CREDENTIALS` | Username or password is incorrect |
| `NOT_FOUND` | The requested resource does not exist |
| `VALIDATION_ERROR` | Request body failed validation |
| `PROVIDER_UNAVAILABLE` | LLM provider is not responding |
| `TOOL_EXECUTION_FAILED` | Tool execution encountered an error |
| `RATE_LIMITED` | Too many requests, please wait |

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- **Authentication endpoints:** 10 requests per minute
- **Message streaming:** 5 concurrent streams per user
- **Tool execution:** 20 tool calls per minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1709834400
```

---

## Related Documentation

- **[20-DEPLOYMENT.md](20-DEPLOYMENT.md)** — Deployment options for running the API server
- **[19-TROUBLESHOOTING.md](19-TROUBLESHOOTING.md)** — Troubleshooting common API issues
- **[04-ARCHITECTURE.md](04-ARCHITECTURE.md)** — Backend architecture and component overview

---

*Last updated: 2026-03-07 | Version: Phase 3 Documentation Overhaul*
