# Interfaces — Communication Contract

Chalie is a cognitive router, not a message router. It receives signals and messages from the world, interprets them through the lens of its human, and decides what matters. Interfaces are deaf to each other — they interact only with Chalie.

---

## 1. Two Channels

| | Signal | Message |
|---|---|---|
| **Analogy** | Overhearing a conversation | Someone talking directly to you |
| **LLM cost** | Zero | Reasoning cost |
| **Destination** | World state (deterministic, in-memory) | Reasoning loop (cognitive pipeline) |
| **Urgency** | Passive — surfaces when relevant | Active — reasoned about now |
| **Example** | "AAPL up 10%" | "Your appointment moved to 3pm" |
| **Broadcast?** | Yes — one source, many Chalies | No — targeted to one Chalie |

Signals update world state and decay naturally (6-hour half-life). The reasoning loop discovers them during idle cycles and surfaces them when they intersect with something the human cares about. Messages enter the cognitive pipeline immediately — Chalie reasons about them, may act, and responds.

---

## 2. Communication Topology

```
                    ┌─────────────────────────────────────────┐
                    │              CHALIE INSTANCE             │
                    │                                         │
  INBOUND           │   ┌───────────┐    ┌────────────────┐  │           OUTBOUND
  ──────────────────┼──►│   World   │    │   Reasoning    │  │  ──────────────────
                    │   │   State   │    │     Loop       │  │
  Signals (passive) │   │           │    │                │  │  Tool execution
  POST /api/signals─┼──►│  zero LLM │    │  LLM pipeline  │──┼──► POST interface/execute
                    │   │           │───►│  (idle cycle)  │  │
  WS subscription ──┼──►│           │    │                │  │  Client push
                    │   └───────────┘    │                │──┼──► WebSocket /ws
                    │                    │                │  │
  Messages (direct) │                    │                │  │
  POST /api/messages┼───────────────────►│                │  │
  WebSocket /ws ────┼───────────────────►│                │  │
                    │                    └────────────────┘  │
                    └─────────────────────────────────────────┘
```

### Inbound

| Transport | Endpoint | Channel | Use Case |
|-----------|----------|---------|----------|
| HTTP POST | `/api/signals` | Signal → world state | Point-to-point signal |
| HTTP POST | `/api/signals/batch` | Signal → world state | Batch signals (max 50) |
| HTTP POST | `/api/messages` | Message → reasoning loop | Interface-targeted message |
| WebSocket | Chalie subscribes outbound | Signal → world state | Broadcast streams (feeds, IoT) |
| WebSocket | `/ws` | Message → reasoning loop | Human chat |

### Outbound

| Transport | Target | Purpose |
|-----------|--------|---------|
| WebSocket `/ws` | Human clients | Push responses, cards, notifications |
| HTTP POST | Interface `/execute` | Invoke interface tools |

Interfaces never connect to `/ws`. The only data flowing from Chalie to an interface is a specific tool invocation — scoped, intentional, gated by the reasoning loop.

---

## 3. Signal Contract

### POST /api/signals

**Auth:** Bearer token (issued during pairing) or session cookie.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `signal_type` | string | yes | — | Signal category (e.g. `price_alert`, `weather_update`) |
| `content` | string | yes | — | Human-readable description |
| `source` | string | no | wrapper_id | Originating system identifier |
| `topic` | string \| null | no | null | Topic hint for salience scoring |
| `activation_energy` | float 0–1 | no | 0.5 | Salience weight — higher values persist longer |
| `metadata` | object \| null | no | null | Arbitrary key-value context |

**Response:** `202 {"ok": true, "signal_id": "<uuid>"}`

| Code | Condition |
|------|-----------|
| 400 | Validation failure |
| 401 | No valid session or bearer token |
| 403 | Wrapper not permitted to emit this signal_type |
| 429 | Rate limit exceeded (100 signals/min per wrapper) |

**Example:**

```json
{
  "signal_type": "weather_forecast",
  "content": "Heavy rain expected this evening, 80% chance",
  "source": "weather-service",
  "topic": "weather",
  "activation_energy": 0.4,
  "metadata": { "precipitation_chance": 0.8, "temperature_high": 18 }
}
```

### POST /api/signals/batch

Same signal schema, as a JSON array. Max 50 per request. Each signal is validated and rate-limited independently — valid signals persist even if others in the batch fail.

**Response:** `200 {"accepted": N, "rejected": M, "errors": [{"index": I, "error": "..."}]}`

### Broadcast Signals (WebSocket Subscription)

For high-volume streams, Chalie subscribes outbound to the interface's stream. Same fields as the point-to-point schema, wrapped in a type envelope (`"type": "signal"`). The interface broadcasts to all subscribers; each Chalie updates its world state independently.

> Broadcast subscription is planned. Point-to-point REST signals are implemented.

### Signal Lifecycle

Signals are written to an in-memory capped list (max 100 entries). On each world state assembly they are scored by temporal decay (6-hour half-life), activation energy, and a 0.15 minimum salience threshold. The top 5 most salient items enter the reasoning context. Signals below threshold disappear naturally.

---

## 4. Message Contract

### WebSocket /ws — Human Messages

**Auth:** Session cookie validated at WebSocket upgrade.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | yes | — | Must be `"chat"` |
| `text` | string | yes | — | Message text |
| `source` | string | no | `"text"` | `"text"` or `"voice"` |
| `image_ids` | string[] | no | — | Pre-uploaded image IDs (max 3) |

### POST /api/messages — Interface Messages

**Auth:** Bearer token (issued during pairing). Planned — schema below is the contract.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Message content directed at Chalie |
| `source` | string | no | wrapper_id | Originating interface identifier |
| `topic` | string \| null | no | null | Topic hint |
| `metadata` | object \| null | no | null | Structured context (IDs, timestamps, etc.) |

**Response:** `202 {"ok": true, "message_id": "<uuid>"}`

**Example:**

```json
{
  "text": "Your appointment has been moved from 2:00 PM to 3:00 PM tomorrow",
  "source": "hospital-portal",
  "topic": "health",
  "metadata": {
    "appointment_id": "apt_12345",
    "original_time": "2026-03-18T14:00:00Z",
    "new_time": "2026-03-18T15:00:00Z"
  }
}
```

### Signal vs Message Decision Table

| Scenario | Channel | Why |
|----------|---------|-----|
| Stock price changed | Signal | Passive — relevant only if human cares |
| Weather forecast updated | Signal | Background knowledge |
| Hospital emergency broadcast | Signal | Each Chalie interprets independently |
| Your appointment moved | Message | Directed — requires reasoning |
| Lab results ready | Message | Actionable — human needs to know |
| New email summary | Signal | Passive awareness |
| Calendar conflict detected | Message | Requires resolution |
| Restaurant closed tonight | Signal | Only relevant if human planned to go |

---

## 5. WebSocket Outbound Contract (/ws)

Human clients only. No interface connects here.

**Auth:** Session cookie validated at WebSocket upgrade.

### Client → Server

| Type | Required Fields | Description |
|------|----------------|-------------|
| `chat` | `type`, `text` | User message. Optional: `source` (`"text"`/`"voice"`), `image_ids` |
| `action` | `type`, `payload` | Deterministic skill invocation — `payload` is skill name + parameters |
| `act_steer` | `type`, `text` | Mid-execution redirect instruction |
| `resume` | `type`, `last_seq` | Reconnect — server replays events with `seq > last_seq` |
| `pong` | `type` | Keepalive response |

### Server → Client

All server messages include a `seq` (monotonic sequence number) for deduplication on reconnect.

| Type | Key Fields | Description |
|------|-----------|-------------|
| `status` | `stage`, `seq` | Processing stage (`"processing"`, `"thinking"`, `"retrieving_memory"`, `"responding"`) |
| `message` | `blocks`, `topic`, `mode`, `confidence`, `exchange_id`, `seq` | Final response. `blocks` is a block array — no raw text/HTML over the wire |
| `act_narration` | `text`, `step`, `seq` | Live ACT loop progress (tool invocation or reasoning step) |
| `card` | `html`, `css`, `scope_id`, `tool_name`, `title`, `accent_color`, `output_id`, `topic`, `seq` | Tool result card |
| `done` | `duration_ms`, `seq` | Response complete |
| `error` | `message`, `recoverable`, `seq` | Error. `recoverable` signals whether the client should retry |
| `notification` | `content`, `topic`, `seq` | Background notification |
| `escalation` | `content`, `topic`, `seq` | Critic-flagged action requiring attention |
| `task` | `seq` | Persistent task update |
| `image_ready` | `image_id`, `status`, `seq` | Image analysis complete (`status`: `"success"` or `"failed"`) |
| `ping` | — | Keepalive probe (every 15s). Client responds with `pong` |

### Reconnection

Send `{"type": "resume", "last_seq": N}`. Server replays all events with `seq > N` from a 200-event catch-up buffer.

---

## 6. Tool Execution Contract (Chalie → Interface)

Any application implementing these three endpoints can pair with Chalie. No interface-specific code is required in infrastructure.

### GET /health

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` |
| `name` | string | Interface display name |
| `version` | string | Interface version |

Health checks run every 30 seconds. Three consecutive failures mark the interface offline and its tools invisible. Recovery restores them automatically.

### GET /capabilities

Returns an array of tool definitions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Tool identifier (unique within interface) |
| `description` | string | yes | What the tool does |
| `documentation` | string | no | Detailed usage documentation |
| `parameters` | array | yes | Parameter definitions |
| `parameters[].name` | string | yes | Parameter name |
| `parameters[].type` | string | yes | `string`, `number`, `boolean`, or `object` |
| `parameters[].required` | boolean | yes | Whether the parameter is required |
| `parameters[].description` | string | yes | Parameter description |
| `returns` | object | no | Return type description |

### POST /execute

Chalie invokes a tool. Request:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `capability` | string | yes | Tool name to invoke |
| `params` | object | yes | Tool parameters |

Response:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string \| null | Human-readable result |
| `data` | object \| null | Structured result data |
| `error` | string \| null | Error description (null on success) |
| `blocks` | array \| null | Block array for UI rendering (overlay updates) |
| `openUrl` | string \| null | URL to open in a new browser tab |

### GET /

Returns the daemon's UI for the app overlay as a `Block[]`. The gateway proxies `/gw/<interface_id>/render` → daemon's `/`, adds a `gateway` field, and forwards to the frontend. No HTML, JS, or CSS — structure only. See `plans/block-protocol.md` for the full block schema.

---

## 7. Interface Lifecycle

### Pairing Flow

1. Human opens brain dashboard → "Generate Pairing Key" (one-time key, 10-minute expiry)
2. Human pastes key + Chalie host:port into the interface's setup
3. Interface calls `POST /api/interfaces/pair`
4. Chalie validates key → health-checks interface → fetches capabilities → registers tools
5. Chalie returns `interface_id` + `signal_token` to the interface

**POST /api/interfaces/pair request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pairing_key` | string | yes | One-time key from brain dashboard |
| `name` | string | yes | Interface display name |
| `host` | string | yes | Interface hostname/IP |
| `port` | integer | yes | Interface port (1–65535) |

**Response:** `201 {"interface_id": "<uuid>", "signal_token": "<token>"}`

After pairing, the interface's capabilities register as normal tools. The LLM does not distinguish them from innate skills or local tools.

### Management Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/interfaces/pairing-key` | Cookie | Generate one-time key (10min TTL) |
| POST | `/api/interfaces/pair` | Pairing key | Complete pairing |
| GET | `/api/interfaces` | Cookie | List all paired interfaces |
| GET | `/api/interfaces/<id>` | Cookie | Get interface details + tools |
| POST | `/api/interfaces/<id>/refresh` | Cookie | Re-fetch capabilities |
| DELETE | `/api/interfaces/<id>` | Cookie | Unpair and remove all tools |

---

## 8. Authentication

| Client Type | Auth Method | Issued When | Scope |
|-------------|-------------|-------------|-------|
| Chat UI / Brain dashboard | Session cookie | `POST /auth/login` | Full API access |
| Paired interface | Bearer token | Pairing | Declared signal types + tool execution |
| Mobile app | Session cookie | `POST /auth/login` | Full API access |

Interfaces authenticate every request with `Authorization: Bearer <signal_token>`. The token is issued once during pairing (shown once, hash stored). Capability scoping ensures an interface can only emit signal types it declared during pairing.

---

## 9. Real-World Examples

### Restaurant (Broadcast Signal)

Restaurant broadcasts "Closed tonight" to all subscribers. Each Chalie receives the signal and updates world state (zero LLM). During the idle cycle, one Chalie's world state holds both "Luigi's closed tonight" (salience 0.82) and "Dinner reservation at 8pm" (salience 0.91). The reasoning loop connects the dots and messages the user: "Luigi's is closed tonight. You have a reservation — want me to find alternatives?"

### Hospital (Signals + Messages + Tools)

**Signal (broadcast):** Hospital broadcasts "Emergency in Wing B — visitor restrictions active." All subscribed Chalies update world state passively. The one Chalie whose human has an appointment in Wing B surfaces the information.

**Message (targeted):** Hospital sends `POST /api/messages` to that Chalie: "Your appointment moved to 3pm." Chalie reasons, surfaces to the user, checks for calendar conflicts.

**Tool execution:** User says "Cancel my appointment." ACT loop selects `cancel_appointment` → `POST hospital/execute` → hospital confirms → Chalie tells the user.

---

## 10. Design Principles

**Signals bypass the reasoning loop.** Written directly to world state. Zero LLM cost regardless of volume.

**Interfaces are tool providers, not data consumers.** They expose tools. They push signals and messages. They never receive broadcast output.

**WebSocket is for humans, REST is for machines.** Broadcast streams use outbound WebSocket subscription from Chalie.

**Interface tools are normal tools.** The LLM prompt does not distinguish source. Offline interfaces have their tools silently removed; recovery restores them.

**No interface-specific code in infrastructure.** Any application implementing `/health`, `/capabilities`, and `/execute` can pair with Chalie.
