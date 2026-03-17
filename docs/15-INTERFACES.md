# Interfaces — Communication Contract

This document defines how external systems connect to Chalie, what protocols they use, and how data flows in and out. It is the governing specification for all interface integration work.

**Core insight:** Chalie is a cognitive router, not a message router. It does not forward data between interfaces. It receives signals and messages from the world, interprets them through the lens of its human, and decides what matters. Interfaces are deaf to each other — they only interact with Chalie.

---

## 1. The Two Channels

Every piece of data flowing into Chalie is one of two things:

| | Signal | Message |
|---|---|---|
| **Analogy** | Overhearing a conversation | Someone talking directly to you |
| **Cost** | Zero LLM tokens | LLM reasoning cost |
| **Destination** | World state (deterministic, in-memory) | Reasoning loop (cognitive pipeline) |
| **Urgency** | Passive — Chalie notices when relevant | Active — Chalie reasons about it now |
| **Example** | "AAPL up 10%" | "Your appointment moved to 3pm" |
| **Broadcast?** | Yes — one source, many Chalies | No — targeted to one Chalie |

**Signals** update Chalie's world state. They decay naturally (6-hour half-life). The reasoning loop sees them during idle cycles and surfaces them if they intersect with something the human cares about. A restaurant broadcasting "closed tonight" becomes relevant only if the human was planning to go there.

**Messages** enter the cognitive pipeline. Chalie reasons about them, may act on them, and responds. A hospital sending "your appointment moved" triggers reasoning, which may reschedule things, notify the user, or invoke tools.

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

### 2.1 Inbound (Data Flowing INTO Chalie)

| Transport | Endpoint | Channel | Use Case |
|-----------|----------|---------|----------|
| HTTP POST | `/api/signals` | Signal → world state | Point-to-point signal from interface |
| HTTP POST | `/api/signals/batch` | Signal → world state | Batch signals (up to 50 per request) |
| HTTP POST | `/api/messages` | Message → reasoning loop | Direct communication from interface |
| WebSocket | Chalie subscribes outbound to interface stream | Signal → world state | Broadcast streams (stock feeds, IoT, news) |
| WebSocket | `/ws` (chat type) | Message → reasoning loop | Human chat messages |

### 2.2 Outbound (Data Flowing OUT of Chalie)

| Transport | Target | Purpose |
|-----------|--------|---------|
| WebSocket `/ws` | Human clients (chat UI, mobile) | Push responses, cards, notifications |
| HTTP POST | Interface `/execute` endpoint | Invoke interface tools |

### 2.3 Interfaces Are Deaf to Each Other

The stock exchange never sees medical data. The hospital never sees stock data. Neither connects to `/ws`. They only receive data from Chalie when Chalie explicitly invokes one of their tools — scoped, intentional, and gated by the reasoning loop.

---

## 3. Signal Contract

### 3.1 Point-to-Point Signal — `POST /api/signals`

**Auth:** Bearer token (issued during pairing) or session cookie.

**Request schema:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `signal_type` | string | yes | — | Signal category (e.g. `price_alert`, `weather_update`, `emergency`) |
| `content` | string | yes | — | Human-readable signal description |
| `source` | string | no | wrapper_id | Originating system identifier |
| `topic` | string \| null | no | null | Topic hint for salience scoring |
| `activation_energy` | float 0–1 | no | 0.5 | Salience weight — higher values persist longer in world state |
| `metadata` | object \| null | no | null | Arbitrary key-value context |

**Response:** `202 {"ok": true, "signal_id": "<uuid>"}`

**Error responses:**

| Code | Condition |
|------|-----------|
| 400 | Validation failure (missing fields, invalid types) |
| 401 | No valid session or bearer token |
| 403 | Wrapper not permitted to emit this signal_type |
| 429 | Rate limit exceeded (100 signals/min per wrapper) |

**Example — weather update:**

```json
{
  "signal_type": "weather_forecast",
  "content": "Heavy rain expected this evening, 80% chance",
  "source": "weather-service",
  "topic": "weather",
  "activation_energy": 0.4,
  "metadata": {
    "precipitation_chance": 0.8,
    "temperature_high": 18,
    "temperature_low": 12
  }
}
```

**Example — stock price alert:**

```json
{
  "signal_type": "price_alert",
  "content": "AAPL at $185.50, up 10.2% today",
  "source": "stock-exchange",
  "activation_energy": 0.7,
  "metadata": {
    "ticker": "AAPL",
    "price": 185.50,
    "change_pct": 10.2,
    "volume": 89420000
  }
}
```

### 3.2 Batch Signals — `POST /api/signals/batch`

**Auth:** Bearer token or session cookie.

**Request body:** JSON array of signal objects (same schema as single signal). Maximum 50 per request.

**Response:** `200 {"accepted": N, "rejected": M, "errors": [{"index": I, "error": "..."}]}`

Each signal is validated, capability-checked, and rate-limited independently. Valid signals are stored even when others in the batch fail.

### 3.3 Broadcast Signals (WebSocket Subscription)

For high-volume broadcast streams (stock tickers, IoT sensors, news feeds), Chalie subscribes outbound to the interface's stream rather than receiving individual HTTP calls.

**Stream message schema** (same fields as `POST /api/signals`, wrapped in a type envelope):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | Must be `"signal"` |
| `signal_type` | string | yes | Signal category |
| `content` | string | yes | Human-readable description |
| `source` | string | no | Originating system identifier |
| `topic` | string \| null | no | Topic hint |
| `activation_energy` | float 0–1 | no | Salience weight (default 0.5) |
| `metadata` | object \| null | no | Arbitrary context |

**Example — broadcast stream event:**

```json
{
  "type": "signal",
  "signal_type": "price_update",
  "content": "MSFT at $420.10, down 0.5%",
  "source": "nasdaq",
  "activation_energy": 0.3,
  "metadata": {"ticker": "MSFT", "price": 420.10}
}
```

The interface broadcasts to all subscribers. It does not know how many Chalies are listening. Each Chalie independently updates its world state.

> **Status:** The subscription worker is planned infrastructure. Point-to-point signals via REST are implemented. Broadcast subscription will be added when the first broadcast interface is built.

### 3.4 Signal Lifecycle

1. Signal arrives (REST or stream subscription)
2. Written to an in-memory capped list (max 100 entries, oldest dropped)
3. Scored on each world state assembly:
   - **Temporal decay:** 6-hour half-life (24h old ≈ 6% salience)
   - **Activation energy amplifier:** higher energy = stays visible longer
   - **Threshold:** 0.15 minimum salience to appear
4. Competes with other world state items (scheduled reminders, tasks, ambient context) — only top 5 most salient items enter the reasoning context
5. Reasoning loop idle cycle discovers relevant signals and may surface them proactively
6. Signals that decay below threshold naturally disappear — no cleanup needed

---

## 4. Message Contract

### 4.1 Human Messages — WebSocket `/ws`

**Auth:** Session cookie validated at WebSocket upgrade.

**Request schema (client → server):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | yes | — | Must be `"chat"` |
| `text` | string | yes | — | Message text (can be empty if image_ids present) |
| `source` | string | no | `"text"` | Input method: `"text"` or `"voice"` |
| `image_ids` | string[] | no | — | Pre-uploaded image IDs (max 3) |

### 4.2 Interface Messages — `POST /api/messages`

**Auth:** Bearer token (issued during pairing).

> **Status:** This endpoint is planned. The schema below defines the contract.

**Request schema:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Message content directed at Chalie |
| `source` | string | no | wrapper_id | Originating interface identifier |
| `topic` | string \| null | no | null | Topic hint |
| `metadata` | object \| null | no | null | Structured context (appointment IDs, timestamps, etc.) |

**Response:** `202 {"ok": true, "message_id": "<uuid>"}`

**Example — appointment change:**

```json
{
  "text": "Your appointment has been moved from 2:00 PM to 3:00 PM tomorrow",
  "source": "hospital-portal",
  "topic": "health",
  "metadata": {
    "appointment_id": "apt_12345",
    "original_time": "2026-03-18T14:00:00Z",
    "new_time": "2026-03-18T15:00:00Z",
    "clinic": "Cardiology",
    "doctor": "Dr. Smith"
  }
}
```

**Example — urgent server alert:**

```json
{
  "text": "Production server api-03 is unresponsive. Last healthy check was 5 minutes ago.",
  "source": "monitoring-system",
  "topic": "infrastructure",
  "metadata": {
    "server": "api-03",
    "last_healthy": "2026-03-17T14:25:00Z",
    "severity": "critical"
  }
}
```

### 4.3 When to Use Signals vs Messages

| Scenario | Channel | Why |
|----------|---------|-----|
| Stock price changed | Signal | Passive information — relevant only if human cares |
| Weather forecast updated | Signal | Background knowledge |
| Hospital emergency broadcast | Signal | Broadcast — each Chalie interprets independently |
| Your appointment moved | Message | Directed at this human — requires reasoning |
| Lab results ready for pickup | Message | Actionable — human needs to know |
| New email received (summary) | Signal | Passive awareness |
| Calendar conflict detected | Message | Requires resolution |
| Restaurant closed tonight | Signal | Only relevant if human planned to go |
| Urgent: server is down | Message | Requires immediate attention |

---

## 5. WebSocket Outbound Contract (`/ws`)

This is how Chalie pushes data to human clients (chat UI, mobile app). No interface connects here.

**Auth:** Session cookie validated at WebSocket upgrade.

### 5.1 Client → Server Messages

**chat** — User message:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"chat"` |
| `text` | string | yes | Message text |
| `source` | string | no | `"text"` or `"voice"` |
| `image_ids` | string[] | no | Pre-uploaded image IDs (max 3) |

**action** — Deterministic skill invocation:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"action"` |
| `payload` | object | yes | Skill name + parameters |

**act_steer** — Mid-execution redirect:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"act_steer"` |
| `text` | string | yes | Redirect instruction |

**resume** — Reconnect with missed event replay:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"resume"` |
| `last_seq` | integer | yes | Last sequence number received |

**pong** — Keepalive response:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | yes | `"pong"` |

### 5.2 Server → Client Messages

All server messages include a `seq` (monotonic sequence number) for deduplication on reconnect.

**status** — Processing stage:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"status"` |
| `stage` | string | Processing stage (e.g. `"processing"`, `"thinking"`, `"retrieving_memory"`, `"responding"`) |
| `seq` | integer | Sequence number |

**message** — Final text response:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"message"` |
| `text` | string | Response text (markdown) |
| `topic` | string | Conversation topic |
| `mode` | string | `"RESPOND"`, `"ACT"`, or `"IGNORE"` |
| `confidence` | float | Response confidence (0–1) |
| `exchange_id` | string | Unique exchange identifier |
| `actions` | array \| null | Optional reply action buttons |
| `seq` | integer | Sequence number |

**act_narration** — Live ACT loop progress:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"act_narration"` |
| `text` | string | Narration line (tool invocation or reasoning step) |
| `step` | integer | Iteration count |
| `seq` | integer | Sequence number |

**card** — Tool result card:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"card"` |
| `html` | string | Sanitized HTML content |
| `css` | string | Scoped CSS styles |
| `scope_id` | string | CSS deduplication key |
| `tool_name` | string | Originating tool |
| `title` | string \| null | Card title |
| `accent_color` | string \| null | Hex color for card accent |
| `output_id` | string | Unique card identifier |
| `topic` | string \| null | Related topic |
| `seq` | integer | Sequence number |

**done** — Response complete:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"done"` |
| `duration_ms` | integer | Wall-clock processing time |
| `seq` | integer | Sequence number |

**error** — Error:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"error"` |
| `message` | string | Error description |
| `recoverable` | boolean | Whether the client should retry |
| `seq` | integer | Sequence number |

**notification** — Background notification:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"notification"` |
| `content` | string | Notification text |
| `topic` | string \| null | Related topic |
| `seq` | integer | Sequence number |

**escalation** — Critic flagged action:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"escalation"` |
| `content` | string | Escalation description |
| `topic` | string \| null | Related topic |
| `seq` | integer | Sequence number |

**task** — Persistent task update:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"task"` |
| `seq` | integer | Sequence number |

**image_ready** — Image analysis complete:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"image_ready"` |
| `image_id` | string | Uploaded image identifier |
| `status` | string | `"success"` or `"failed"` |
| `seq` | integer | Sequence number |

**ping** — Keepalive probe (every 15s):

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"ping"` |

### 5.3 Reconnection

On reconnect, the client sends `{"type": "resume", "last_seq": N}`. The server replays all events with `seq > N` from a 200-event catch-up buffer. Client deduplicates by tracking seen sequence numbers.

---

## 6. Tool Execution Contract (Chalie → Interface)

When Chalie's reasoning loop decides to act, it calls the interface's execute endpoint.

### 6.1 Interface Must Expose

**GET /health**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` |
| `name` | string | Interface display name |
| `version` | string | Interface version |

**GET /capabilities** — Returns array of tool definitions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Tool identifier (unique within interface) |
| `description` | string | yes | What the tool does |
| `documentation` | string | no | Detailed usage documentation |
| `parameters` | array | yes | Parameter definitions |
| `parameters[].name` | string | yes | Parameter name |
| `parameters[].type` | string | yes | Parameter type (`string`, `number`, `boolean`, `object`) |
| `parameters[].required` | boolean | yes | Whether the parameter is required |
| `parameters[].description` | string | yes | Parameter description |
| `returns` | object | no | Return type description |

**POST /execute** — Chalie invokes a tool:

Request:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `capability` | string | yes | Tool name to invoke |
| `params` | object | yes | Tool parameters |

Response:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string \| null | Human-readable result |
| `html` | string \| null | Rich HTML output (rendered as card) |
| `data` | object \| null | Structured result data |
| `error` | string \| null | Error description (null on success) |

### 6.2 Health Monitoring

- Health checks run every **30 seconds**
- **3 consecutive failures** → interface marked offline → its tools become invisible
- On recovery → tools automatically re-appear
- No user notification for transient state changes

---

## 7. Interface Lifecycle

### 7.1 Pairing

1. Human opens brain dashboard → "Generate Pairing Key"
2. Chalie generates a one-time key (10-minute expiry) and displays its host:port
3. Human enters host:port + key into the interface's setup
4. Interface calls `POST /api/interfaces/pair`
5. Chalie validates key → health-checks interface → fetches capabilities → registers tools
6. Chalie returns `interface_id` + `signal_token` to the interface

**`POST /api/interfaces/pair` request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pairing_key` | string | yes | One-time key from brain dashboard |
| `name` | string | yes | Interface display name |
| `host` | string | yes | Interface hostname/IP |
| `port` | integer | yes | Interface port (1–65535) |

**Response:** `200 {"interface_id": "<uuid>", "signal_token": "<token>"}`

After pairing, the interface's capabilities are registered as tools. The LLM sees them alongside innate skills and local tools — it does not know the difference.

### 7.2 Management Endpoints

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
|-------------|------------|-------------|-------|
| Chat UI / Brain dashboard | Session cookie | Login (`POST /auth/login`) | Full API access |
| Paired interface | Bearer token | Pairing | Declared signal types + tool execution responses |
| Mobile app | Session cookie | Login | Full API access |

Interfaces authenticate every request with `Authorization: Bearer <signal_token>`. The token is issued once during pairing (shown once, hash stored). Capability scoping ensures an interface can only emit signal types it declared during pairing.

---

## 9. Real-World Examples

### 9.1 Restaurant (Broadcast Signal)

1. Restaurant broadcasts "Closed tonight" to all subscribers
2. Each subscribed Chalie receives the signal → world state updated (zero LLM)
3. User hasn't chatted today — reasoning loop enters idle cycle
4. World state shows: "Luigi's closed tonight" (salience 0.82) + "Dinner reservation at 8pm" (salience 0.91)
5. Reasoning loop connects the dots → proactive message to user: "Luigi's is closed tonight. You have a reservation — want me to find alternatives?"

### 9.2 Hospital (Signals + Messages + Tools)

**Signal (broadcast):** Hospital broadcasts "Emergency in Wing B — visitor restrictions active." All subscribed Chalies update world state. Most ignore it. One Chalie's human has an appointment in Wing B — it surfaces the information.

**Message (targeted):** Hospital sends `POST /api/messages` to one Chalie: "Your appointment moved to 3pm." Chalie reasons, surfaces to user, checks for calendar conflicts.

**Tool execution (Chalie → Hospital):** User says "Cancel my appointment." ACT loop selects `cancel_appointment` tool → `POST hospital/execute` → hospital confirms → Chalie tells user.

### 9.3 Stock Exchange (High-Volume Broadcast)

1. Chalie subscribes to the exchange's broadcast stream
2. Exchange sends thousands of price updates per minute — each one updates world state (zero cost)
3. User asks "How's my portfolio?" — reasoning loop has fresh world state with current prices
4. Chalie responds with a portfolio summary using the latest signal data

### 9.4 Chat Interface (Human Client)

1. Chat app connects to Chalie's `/ws` (session cookie auth)
2. User speaks → chat transcribes → sends `chat` message via WebSocket
3. Chalie reasons (world state includes signals from all paired interfaces)
4. Responds via `/ws` with integrated context: "Your appointment moved and Luigi's is closed tonight"
5. User says "Cancel both" → ACT loop invokes `cancel_appointment` on hospital + `cancel_reservation` on restaurant
6. Both interfaces confirm via `/execute` response → Chalie tells user "Done"

---

## 10. Design Principles

**Signals bypass the reasoning loop.** Written directly to world state. Zero LLM cost regardless of volume. A stock exchange sending 10,000 updates costs nothing.

**Interfaces are tool providers, not data consumers.** They expose tools. They push signals and messages. They never receive broadcast output. The only data from Chalie to an interface is a specific tool invocation.

**WebSocket is for humans, REST is for machines.** Human clients use WebSocket for real-time push. Interfaces use REST for signals and messages. Broadcast streams use outbound WebSocket subscription.

**Signal tokens reuse wrapper auth.** No separate auth system. The signal_token is a standard wrapper bearer token with capability scoping and rate limiting.

**Interface tools are normal tools.** The LLM prompt does not distinguish source types. Offline interfaces have their tools silently removed; recovery restores them.

**No interface-specific code in infrastructure.** The protocol is generic. Any application implementing `/health`, `/capabilities`, and `/execute` can pair with Chalie.

---

## 11. Key Files

| File | Purpose |
|------|---------|
| `backend/api/signals.py` | Signal ingestion endpoints |
| `backend/api/interfaces.py` | Interface pairing and management |
| `backend/api/wrappers.py` | Wrapper token management |
| `backend/api/websocket.py` | WebSocket endpoint for human clients |
| `backend/services/world_state_service.py` | World state aggregator (includes external signals) |
| `backend/services/interface_registry_service.py` | Interface lifecycle (pairing, health, capabilities) |
| `backend/services/wrapper_auth_service.py` | Bearer token authentication |
| `backend/services/tool_registry_service.py` | Tool discovery and dispatch |
| `backend/workers/interface_health_worker.py` | Health check daemon thread |
| `backend/services/reasoning_loop_service.py` | Signal consumer and continuous reasoning |
| `docs/18-SIGNAL-CONTRACT.md` | Internal signal contract (cognitive services only) |
