---
title: "API Reference"
description: "Complete REST API documentation for Chalie backend services."
tags: ["api", "reference", "endpoints"]
created_at: 2025-01-13T00:00:00Z
updated_at: 2025-01-13T00:00:00Z
---

# Chalie API Reference

Complete documentation for all REST endpoints and WebSocket protocol.

## Table of Contents

- [Authentication](#authentication)
- [WebSocket Protocol](#websocket-protocol)
- [Auth Endpoints](#auth-endpoints)
- [User Auth (Master Account)](#user-auth-master-account)
- [Conversation](#conversation)
- [Memory](#memory)
- [Documents](#documents)
- [Moments](#moments)
- [Lists](#lists)
- [Proactive Notifications](#proactive-notifications)
- [Push Notifications](#push-notifications)
- [Privacy & Data Management](#privacy--data-management)
- [System & Health](#system--health)
- [Tools](#tools)
- [Voice (STT/TTS)](#voice-stttts)

---

## Authentication

All protected endpoints require a valid `chalie_session` cookie. This session is obtained by logging in via `/auth/login`.

### Session Cookie

| Attribute | Value |
|-----------|-------|
| Name | `chalie_session` |
| Path | `/` |
| Secure | Yes (HTTPS only) |
| HttpOnly | Yes |
| SameSite | Lax |

---

## WebSocket Protocol

**Endpoint:** `WS /ws`  
**Authentication:** Requires valid `chalie_session` cookie

The WebSocket provides a single bidirectional channel for chat, voice streaming, and real-time notifications.

### Client → Server Messages

#### Chat Message

```json
{
  "type": "chat",
  "text": "string (required)",
  "source": "text|voice"
}
```

- `text`: The user's message or voice transcript
- `source`: Origin of the input (`text` for keyboard, `voice` for STT)

#### Voice Start

```json
{
  "type": "voice_start",
  "format": "pcm|opus"
}
```

Initiates real-time speech-to-text streaming.

#### Voice Data (PCM Audio)

```json
{
  "type": "voice_data",
  "data": "<base64-encoded-audio-chunk>"
}
```

Send audio chunks during active voice session.

#### Voice End

```json
{
  "type": "voice_end"
}
```

Stops the current STT session and returns final transcript.

### Server → Client Messages

#### Chat Response (SSE-style)

```json
{
  "event": "response",
  "data": {"text": "..."}
}
```

Streamed token-by-token during LLM generation.

#### Voice Transcript Update

```json
{
  "event": "voice_transcript",
  "data": {"transcript": "..."}
}
```

Real-time partial transcripts during STT.

#### Notification Push

```json
{
  "event": "notification",
  "data": {
    "id": "uuid",
    "title": "...",
    "message": "...",
    "priority": "normal|high",
    "timestamp": "ISO8601"
  }
}
```

Proactive notifications from scheduled tasks or external triggers.

#### Error

```json
{
  "event": "error",
  "data": {"message": "..."}
}
```

---

## Auth Endpoints

### POST /auth/login

Authenticate with email and password. Returns session cookie on success.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "string"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "message": "Login successful"
}
```

**Response (401):**
```json
{
  "status": "error",
  "message": "Invalid credentials"
}
```

**cURL Example:**
```bash
curl -X POST https://api.chalie.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"secret"}' \
  --cookie-jar cookies.txt
```

---

### POST /auth/logout

Invalidate the current session.

**Authentication:** Required (`chalie_session` cookie)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Logged out successfully"
}
```

**cURL Example:**
```bash
curl -X POST https://api.chalie.com/auth/logout \
  --cookie cookies.txt
```

---

## User Auth (Master Account)

### POST /auth/master/register

Register a new master account.

**Request Body:**
```json
{
  "email": "master@example.com",
  "password": "string"
}
```

**Response (201):**
```json
{
  "status": "ok",
  "message": "Master account created"
}
```

---

### POST /auth/master/login

Authenticate master account. Returns session cookie.

**Request Body:**
```json
{
  "email": "master@example.com",
  "password": "string"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "message": "Login successful"
}
```

---

### POST /auth/master/logout

Invalidate master session.

**Authentication:** Required (`chalie_session` cookie)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Logged out successfully"
}
```

---

## Conversation

### GET /conversation/recent

Fetch recent conversation history.

**Authentication:** Required

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| limit | int | 20 | Number of messages to return |

**Response (200):**
```json
{
  "status": "ok",
  "messages": [
    {
      "id": "uuid",
      "role": "user|assistant",
      "content": "...",
      "timestamp": "ISO8601"
    }
  ]
}
```

---

### GET /conversation/summary

Get a summarized view of conversation history.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "summary": "...",
  "message_count": 42,
  "first_message_at": "ISO8601",
  "last_message_at": "ISO8601"
}
```

---

### GET /conversation/spark-status

Check if Spark (AI engine) is available.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "spark_available": true,
  "last_check_at": "ISO8601"
}
```

---

## Memory

### GET /memory/context

Retrieve long-term memory context for the current session.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "context": {
    "user_preferences": {...},
    "conversation_history_summary": "...",
    "active_topics": [...]
  }
}
```

---

### POST /memory/forget

Remove specific memories or clear all.

**Authentication:** Required

**Request Body:**
```json
{
  "query": "string (optional)",
  "clear_all": false
}
```

**Response (200):**
```json
{
  "status": "ok",
  "message": "Memory cleared"
}
```

---

### POST /memory/search

Search long-term memory.

**Authentication:** Required

**Request Body:**
```json
{
  "query": "string (required)"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "results": [
    {
      "id": "uuid",
      "content": "...",
      "relevance_score": 0.95,
      "created_at": "ISO8601"
    }
  ]
}
```

---

## Documents

### POST /documents/upload

Upload a document for processing and indexing.

**Authentication:** Required  
**Content-Type:** `multipart/form-data`

**Form Data:**
| Field | Type | Description |
|-------|------|-------------|
| file | binary | Document to upload (PDF, DOCX, TXT, etc.) |

**Response (201):**
```json
{
  "status": "ok",
  "document_id": "uuid",
  "filename": "example.pdf",
  "size_bytes": 123456,
  "processed_at": "ISO8601"
}
```

---

### GET /documents

List all uploaded documents.

**Authentication:** Required

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page |

**Response (200):**
```json
{
  "status": "ok",
  "documents": [
    {
      "id": "uuid",
      "filename": "...",
      "size_bytes": 123456,
      "uploaded_at": "ISO8601"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 42
  }
}
```

---

### DELETE /documents/<id>

Delete a specific document.

**Authentication:** Required  
**Path Parameter:** `id` (uuid)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Document deleted"
}
```

---

## Moments

### POST /moments

Pin a moment (important snippet or insight).

**Authentication:** Required

**Request Body:**
```json
{
  "title": "string (required)",
  "content": "string",
  "tags": ["tag1", "tag2"]
}
```

**Response (201):**
```json
{
  "status": "ok",
  "moment_id": "uuid"
}
```

---

### GET /moments

List all active moments.

**Authentication:** Required

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page |

**Response (200):**
```json
{
  "status": "ok",
  "moments": [
    {
      "id": "uuid",
      "title": "...",
      "content": "...",
      "tags": [...],
      "created_at": "ISO8601"
    }
  ],
  "pagination": {...}
}
```

---

### DELETE /moments/<id>

Unpin (delete) a moment.

**Authentication:** Required  
**Path Parameter:** `id` (uuid)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Moment deleted"
}
```

---

### POST /moments/search

Search moments by query.

**Authentication:** Required

**Request Body:**
```json
{
  "query": "string (required)"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "results": [...]
}
```

---

## Lists

### GET /lists

List all active lists with summary counts.

**Authentication:** Required

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page |

**Response (200):**
```json
{
  "status": "ok",
  "lists": [
    {
      "id": "uuid",
      "name": "...",
      "item_count": 5,
      "created_at": "ISO8601"
    }
  ],
  "pagination": {...}
}
```

---

### POST /lists

Create a new list.

**Authentication:** Required

**Request Body:**
```json
{
  "name": "string (required)"
}
```

**Response (201):**
```json
{
  "status": "ok",
  "list_id": "uuid"
}
```

---

### GET /lists/<id>

Get a specific list with all items.

**Authentication:** Required  
**Path Parameter:** `id` (uuid)

**Response (200):**
```json
{
  "status": "ok",
  "list": {
    "id": "uuid",
    "name": "...",
    "items": [...],
    "created_at": "ISO8601"
  }
}
```

---

### DELETE /lists/<id>

Delete a list and all its items.

**Authentication:** Required  
**Path Parameter:** `id` (uuid)

**Response (200):**
```json
{
  "status": "ok",
  "message": "List deleted"
}
```

---

## Proactive Notifications

### GET /proactive/notifications

Fetch buffered notifications (REST fallback for WebSocket).

**Authentication:** Required

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| since | ISO8601 | - | Fetch notifications after this timestamp |

**Response (200):**
```json
{
  "status": "ok",
  "notifications": [
    {
      "id": "uuid",
      "title": "...",
      "message": "...",
      "priority": "normal|high",
      "timestamp": "ISO8601"
    }
  ]
}
```

---

## Push Notifications

### GET /push/vapid-key

Get VAPID public key for Web Push subscription.

**Authentication:** Not required

**Response (200):**
```json
{
  "status": "ok",
  "vapid_public_key": "base64-encoded-key"
}
```

---

### POST /push/subscribe

Register a new push subscription.

**Authentication:** Required

**Request Body:**
```json
{
  "endpoint": "...",
  "keys": {
    "p256dh": "...",
    "auth": "..."
  }
}
```

**Response (201):**
```json
{
  "status": "ok",
  "subscription_id": "uuid"
}
```

---

### DELETE /push/unsubscribe/<id>

Remove a push subscription.

**Authentication:** Required  
**Path Parameter:** `id` (uuid)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Subscription removed"
}
```

---

## Privacy & Data Management

### GET /privacy/data-summary

Get a summary of all user data stored.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "summary": {
    "conversations_count": 42,
    "documents_count": 15,
    "moments_count": 8,
    "lists_count": 3,
    "total_storage_bytes": 12345678
  }
}
```

---

### POST /privacy/export

Export all user data as JSON.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "export_url": "/downloads/user-export-uuid.json",
  "expires_at": "ISO8601"
}
```

---

### POST /privacy/delete-all

Permanently delete all user data. **Irreversible.**

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "message": "All data deleted"
}
```

---

## System & Health

### GET /health

Basic health check endpoint.

**Authentication:** Not required

**Response (200):**
```json
{
  "status": "ok"
}
```

---

### GET /metrics

Prometheus-style metrics endpoint.

**Authentication:** Not required

**Response (200):**
```text
# HELP chalie_requests_total Total requests
# TYPE chalie_requests_total counter
chalie_requests_total{endpoint="/health"} 150
...
```

---

### GET /system/status

Get system status and component health.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "components": {
    "database": "healthy",
    "spark_engine": "healthy",
    "stt_service": "healthy",
    "tts_service": "degraded"
  },
  "uptime_seconds": 123456,
  "timestamp": "ISO8601"
}
```

---

### GET /system/observability/logs

Fetch recent application logs.

**Authentication:** Required (master account)

**Query Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| level | string | info | Log level filter (debug, info, warning, error) |
| limit | int | 100 | Number of log entries |

**Response (200):**
```json
{
  "status": "ok",
  "logs": [
    {
      "timestamp": "ISO8601",
      "level": "INFO",
      "message": "...",
      "source": "module.name"
    }
  ]
}
```

---

## Tools

### GET /tools

List all available tools.

**Authentication:** Required

**Response (200):**
```json
{
  "status": "ok",
  "tools": [
    {
      "id": "tool_id",
      "name": "...",
      "description": "...",
      "enabled": true,
      "config_schema": {...}
    }
  ]
}
```

---

### GET /tools/<id>

Get details for a specific tool.

**Authentication:** Required  
**Path Parameter:** `id` (string)

**Response (200):**
```json
{
  "status": "ok",
  "tool": {
    "id": "...",
    "name": "...",
    "description": "...",
    "enabled": true,
    "config_schema": {...},
    "current_config": {...}
  }
}
```

---

### POST /tools/<id>/enable

Enable a tool.

**Authentication:** Required  
**Path Parameter:** `id` (string)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Tool enabled"
}
```

---

### POST /tools/<id>/disable

Disable a tool.

**Authentication:** Required  
**Path Parameter:** `id` (string)

**Response (200):**
```json
{
  "status": "ok",
  "message": "Tool disabled"
}
```

---

### POST /tools/<id>/config

Update tool configuration.

**Authentication:** Required  
**Path Parameter:** `id` (string)

**Request Body:**
```json
{
  "key1": "value1",
  "key2": 123
}
```

**Response (200):**
```json
{
  "status": "ok",
  "message": "Configuration updated"
}
```

---

## Voice (STT/TTS)

### POST /voice/stt

Convert audio to text using faster-whisper.

**Authentication:** Required  
**Content-Type:** `multipart/form-data` or `application/json`

**Request Body (JSON):**
```json
{
  "audio": "<base64-encoded-audio>",
  "format": "wav|mp3",
  "language": "en"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "transcript": "..."
}
```

---

### POST /voice/tts

Convert text to speech using KittenTTS.

**Authentication:** Required

**Request Body:**
```json
{
  "text": "string (required)",
  "voice_id": "default",
  "format": "wav|mp3"
}
```

**Response (200):**
```json
{
  "status": "ok",
  "audio_url": "/downloads/tts-uuid.wav",
  "expires_at": "ISO8601"
}
```

---

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request

```json
{
  "status": "error",
  "message": "Invalid request parameters"
}
```

### 401 Unauthorized

```json
{
  "status": "error",
  "message": "Authentication required"
}
```

### 403 Forbidden

```json
{
  "status": "error",
  "message": "Access denied"
}
```

### 500 Internal Server Error

```json
{
  "status": "error",
  "message": "Internal server error"
}
```

---

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

| Endpoint Pattern | Limit | Window |
|------------------|-------|--------|
| `/auth/*` | 10 requests | 1 minute |
| `/voice/*` | 30 requests | 1 minute |
| All others | 100 requests | 1 minute |

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when window resets

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-13 | Initial API documentation |
