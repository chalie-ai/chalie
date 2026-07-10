# Chalie Vocabulary

Domain-specific terminology used throughout the Chalie system.

| Term | Description | Example |
|---|---|---|
| `turn` | The per-channel conversation boundary; many transcript rows share one `turn_id`. The unit of interruption. | `turn_id=42` |
| `transcript` | The persistent, channel-scoped conversation log table (`transcript` rows: role, content, turn_id, settled). | A row with `role='assistant'`, `turn_id=42`, `settled=1` |
| `channel` | The transcript/telemetry namespace string that scopes a conversation. | `'user'`, `'dmn'`, `'delegate:web_search'` |
| `thread` | A turn grown past its settle0 by one or more user replies. | `turn_id=42` once a user replies into it |
| `turn_execution` | The DB-backed lifecycle record for one turn's run: `state` (`working`/`completed`/`cancelled`/`crashed`), `cancel_requested`, `started_at`/`ended_at`. `cancel()` is the sole authority (P10) for a turn's terminal state — it stamps `cancel_requested=1` and `state='cancelled'` synchronously in one call, without waiting for the still-running step loop; `finish()` becomes a no-op once it observes the row already closed. | `state='cancelled'`, `cancel_requested=1` |
| `settle0` | The id of the first settled assistant row in a turn — the boundary between the main spine and fork views. | `settle0=15` |
| `settled` | Per-row flag marking the assistant row that closes a turn. | `settled=1` on the closing reply |
| `model` | The specific LLM model id associated with one provider row. | `claude-opus-4-7`, `gpt-4o`, `gemma4:31b` |
| `provider` | The API provider for a family of LLM models; one row per platform+model. | `ollama`, `anthropic`, `openai`, `gemini` |
| `ProviderType` | Enum selecting which provider slot a send resolves to. | `CHAT`, `VISION`, `DELEGATE`, `VISUAL_OUTPUT` |
| `ThinkingLevel` | Request-level reasoning-effort knob. | `LOW`, `MEDIUM`, `HIGH`, `MAX` |
| `ability` | A built-in tool the agent can dispatch (subclasses `Ability`). | `weather`, `email`, `document` |
| `ability action` | A tool subcommand every dispatchable ability accepts. | `memory(action='recall')`, `document(action='search')` |
| `act_summary` | The 3–10 word "what I'm doing" string required on every ability call. | `"Checking the weather in Valletta"` |
| `act_trail` | The cumulative record of tool_calls rows for one ACT loop. | `[weather] Valletta, MT → 22°C, sunny` |
| `ToolResult` | The frozen dataclass every `Ability.run` returns. | `ToolResult.ok(body=...)`, `ToolResult.err(code='not-connected')` |
| `skill` | A step-by-step YAML playbook discoverable via `find_skills`. | `a3-problem-analysis.yaml` |
| `delegate` | A tool call opted into async background execution. | `delegate_id="web_search_a3b2c1d4"` |
| `subagent` | User-facing label for a backgrounded delegate. | `GET /api/subagents/all` |
| `capability` | An external system adapter an ability wraps. | `mail_capability` (IMAP), `home_capability` (Home Assistant) |
| `MCP` | Model Context Protocol connection to a remote MCP server. | `_mcp_notes_create_document` |
| `always_available` | Tools pinned in every LLM call on a channel. | `["find_skills", "find_tools", "memory"]` |
| `DISCOVERABLE` | Ability trait; when False the tool only reaches the model by being pinned. | `thinking`, `browser` are non-discoverable |
| `counts_as_settle` | Ability trait; when True a tool_calls row demotes its row's `settled=1` to 0. | `bash` (True), `thinking` (False) |
| `episode` | A narrative memory unit (a transcript-window gist) with salience 1–10 and decay. | `gist='Discussed travel to Japan'`, `salience=7` |
| `salience` | Per-episode importance score (1–10). | `salience=8` |
| `super-episode` | A consolidated level-1 episode built from leaf episodes via clustering. | `level=1` covering 50 leaves |
| `era digest` | A level-2+ consolidated episode covering many super-episodes. | `level=2` covering 25 super-episodes |
| `data_graph` | Bi-temporal key-value graph of typed facts (the concepts layer). | kind `user_specific`, `place`, `contact` |
| `flashback` | The curated memory bundle injected before iteration 0 on session start. | "5 facts + 3 dated episode gists" |
| `compaction` | Off-spine durable checkpoint of past transcript rows. | MAIN watermark `turn_id=15` |
| `documents` | Uploaded file metadata + chunks. | `mime_type='application/pdf'`, `status='ready'` |
| `policies` | The Allow / Ask / Deny gate per tool action. | `'allow'`, `'ask'`, `'deny'` |
| `vault` | Envelope-encrypted credential store (AES-256-GCM, DEK wrapped by password-derived KEK). | `kdf_iterations=600000` |
| `MessageProcessor` | The single flat orchestrator for every LLM turn (one per turn, per channel). | lifecycle signals: `working`, `done`, `tool_called` |
| `ExecutionTracker` | Per-turn object the `MessageProcessor` builds after turn resolution; owns the `turn_execution` row, is the sole emitter of lifecycle WS frames, and answers `should_stop()`. | `ExecutionTracker(config, turn_id)` |
| `should_stop` | Cooperative-stop predicate checked at each turn checkpoint; True once a cancel has been requested. Replaces the old in-memory cancel `Event`. | `if self.should_stop(): raise _TurnCancelled` |
| `ProcessorConfig` | Frozen dataclass parameterising one channel's `MessageProcessor`. | `UserConfig`, `DmnConfig`, `DiscoveryConfig` |
| `policy_channel` | Enum picking which policy rows apply. | `CHAT`, `SUBCONSCIOUS`, `EXTERNAL_AGENT` |
| `post_turn_hooks` | Tuple of independent after-turn work units a config owns. | `ProactiveSuggestionHook`, `PersistUserSummaryHook` |
| `memory_seed` | Flag that fires a `memory.recall` at turn 0. | `UserConfig.memory_seed=True` |
| `broadcast_to` | The live WebSocket output target for a channel. | `"user"` (UserConfig only) |
| `SubconsciousWorker` | Idle-gated 5-minute cognition tick (consolidate, decay, patterns, synthesis…). | runs `_step_consolidate`, `_step_decay` |
| `SchedulerService` | Background poller that wakes every wall-clock minute and fires any enabled schedule whose day/hour/minute cron fields match — no separate "due" state to track. | `day=None, hour=3, minute=0` → every day at 03:00 |
| `WorldAwarenessService` | Hourly interest-driven news scan that pushes signals. | interest `{'term':'machine learning'}` |
| `WorldState` | Singleton holding the agent's "what's going on" cache. | `last_heartbeat_at`, `current_device_class` |
| `signal` | Typed world-state update absorbed from heartbeats / user messages. | `push_signal('news', 'X released Y')` |
| `heartbeat` | FE POST `/health` payload persisted to telemetry. | `{device.name, location.lat, locale.timezone}` |
| `usage_class` | Per-call LLM categorization. | `'chat'`, `'subagent'`, `'subconscious'` |
| `snapshot` | Full-instance backup (db, mcp_tools, pre-trained, vault key material). | a `.chalie-snapshot` file |
| `lane` | Per-conversation-surface FE state key. | `'main'` (spine), `'t42'` (thread 42) |
| `chip` | FE rendering of a tool-call under an assistant message. | `{tool_name, summary}` |
| `card` | Rich-media payload rendered as a structured card. | `WeatherCard`, `SchedulerCard` |
| `segment` | FE parse of assistant content into text / rich blocks. | `text`, `rich` |
| `Endpoint` | ABC every migrated REST CRUD group subclasses (`api/endpoints/`); the base generates the routes, auth, and envelopes from `slug()` alone. | `GET /api/{slug}/all`, `POST /api/{slug}/-1` (create) |
| `Action` | `Endpoint` subclass for verb-shaped operations (`api/actions/<slug>/<verb>.py`); `all` is a reserved verb. | `POST /api/lists/items/<id>` |
| `envelope` | The uniform response shape built only by the `Response` DTO base. | `{success, result}`, error: `{success: false, result: [], error}` |
| `find_tools` | Discovery ability that surfaces tools the model can use. | `find_tools(query=['weather','valletta'])` |
| `internal_dev` | Env-var flag gating in-development features. | `CHALIE_INTERNAL_DEV='1'` |