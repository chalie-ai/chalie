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
| `provider` | A configured way to reach one model: one row per platform+model, carrying the host and credential. | `ollama`, `anthropic`, `openai`, `gemini` |
| `platform` | The vendor a provider row talks to, stored as a stable string in `providers.platform`. Every platform has exactly one client class declaring it, so the string is the dispatch key for sending, for listing models, and for the setup catalog — all three read the same registry, and a new vendor is a new module rather than an edit to any of them. Vendors speaking the OpenAI wire protocol subclass the shared base and state only what differs, chiefly where they publish their context window. | `vllm`, `llama_cpp`, `xai`, `openai_compatible` (the escape hatch for a host with no dedicated class) |
| `ProviderType` | Enum selecting which provider slot a send resolves to. | `CHAT`, `VISION`, `DELEGATE`, `VISUAL_OUTPUT` |
| `ThinkingLevel` | Request-level reasoning-effort knob. | `LOW`, `MEDIUM`, `HIGH`, `MAX` |
| `ability` | A built-in tool the agent can dispatch (subclasses `Ability`). | `weather`, `email`, `vision` |
| `ability action` | A tool subcommand every dispatchable ability accepts. | `memory(action='recall')`, `list(action='create')` |
| `act_summary` | The 3–10 word "what I'm doing" string required on every ability call. | `"Checking the weather in Valletta"` |
| `act_trail` | The cumulative record of tool_calls rows for one ACT loop. | `[weather] Valletta, MT → 22°C, sunny` |
| `ToolResult` | The frozen dataclass every `Ability.run` returns. | `ToolResult.ok(body=...)`, `ToolResult.err(code='not-connected')` |
| `image_search` | Built-in ability that finds images on the web for a text query, then vision-verifies each candidate when a vision provider is configured. | `verified=true` per image, or `degraded=true` with no vision provider configured |
| `web_fetch` | Built-in ability and the single URL-owning tool: fetches a URL and persists it under the data directory — HTML becomes markdown (or verbatim HTML with `convert_to_markdown=false`) in the web-pages store, overwritten in place on re-fetch; every other content type streams to the downloads folder. Small pages return their content directly; large ones return only the saved path for `read` to load in line-range chunks. `read` itself is file-only and redirects any URL here. | `web_fetch(url='https://example.com/post')` → `data/web/pages/example.com--post.md` |
| `skill` | A step-by-step YAML playbook discoverable via `find_skills`. | `a3-problem-analysis.yaml` |
| `delegate` | A tool call opted into async background execution. | `delegate_id="web_search_a3b2c1d4"` |
| `delegate ability` | An `Ability` that builds its own `ProcessorConfig` and drives a focused `MessageProcessor` loop instead of acting directly, scoped to a pinned tool set. | `web_search`, `web_browse`, `pim`, `code_agent` |
| `pim` | The delegate ability owning the user's personal information — email, calendar, contacts, and reminders — behind one tool; the sole route to those abilities now that they're non-discoverable elsewhere. | `pim(instructions="what do I have tomorrow")` |
| `code_agent` | The delegate ability that writes and runs TypeScript in one persistent, sandboxed file workspace shared across every invocation — a script written in chat is immediately available to any schedule. | `code_agent(instructions="write a script that...")` |
| `subagent` | User-facing label for a backgrounded delegate. | `GET /api/subagents/all` |
| `capability` | An external system adapter an ability wraps. | `mail_capability` (IMAP), `home_capability` (Home Assistant) |
| `MCP` | Model Context Protocol connection to a remote MCP server. | `_mcp_notes_create_document` |
| `always_available` | Tools pinned in every LLM call on a channel. | `["find_skills", "find_tools", "mcp_manager", "memory"]` |
| `DISCOVERABLE` | Ability trait; when False the tool only reaches the model by being pinned. | `mcp_manager`, `browser` are non-discoverable |
| `CATEGORY` | Ability trait; the heading the tool is listed under in the `find_tools` menu. Required on every discoverable ability — the registry refuses to load one without it. The enum's declaration order is the menu's render order. | `read` → `File Operations`, `pim` → `Delegate (subagent)` |
| `ALLOW_EMPTY` | Ability trait; the required params that may legitimately arrive as `""`. The dispatcher's pre-gate rejects an empty required param by default — a global rule this opts individual params out of, without making them optional (an absent one is still `missing-params`). | `edit_file` → `("replace",)`, because an empty replacement is how you delete |
| `VERBATIM` | Ability trait; the params whose values reach `run()` exactly as the model sent them. The dispatch seam scrubs every argument value by default (leaked provider sentinel tokens out, surrounding whitespace trimmed) — right for a value the tool interprets, wrong for one it stores, where a leading tab or a trailing newline is data. | `edit_file` → `("search", "replace")`, so a whole-line edit keeps its `\n` |
| `UNTRUSTED_CONTENT` | Ability trait; maps an action to the steer appended to that action's successful result, for actions whose payload can carry prose written outside the conversation (`docs/SECURITY.md`). Keyed like `ACTION_REQUIRED` — on the resolved action, or `""` for a tool with none. Per action, because a tool's actions differ: name what the content is, who controls it, and how an attack arrives through that channel. An action left out gets nothing, which is the common case and deliberate — a warning on Chalie's own output trains the model past the one that matters. | `email` → `search`, `read` steered, `send` not; `browser` → `open`, `click` steered, `fill` not |
| `counts_as_settle` | Ability trait; when True a tool_calls row demotes its row's `settled=1` to 0. | `bash` (True), `thinking` (False) |
| `episode` | A narrative memory unit (a transcript-window gist) with salience 1–10 and decay. | `gist='Discussed travel to Japan'`, `salience=7` |
| `salience` | Per-episode importance score (1–10). | `salience=8` |
| `super-episode` | A consolidated level-1 episode built from leaf episodes via clustering. | `level=1` covering 50 leaves |
| `era digest` | A level-2+ consolidated episode covering many super-episodes. | `level=2` covering 25 super-episodes |
| `data_graph` | Bi-temporal key-value graph of typed facts (the concepts layer). | kind `user_specific`, `place`, `contact` |
| `flashback` | The curated memory bundle injected before iteration 0 on session start. | "5 facts + 3 dated episode gists" |
| `compaction` | Off-spine durable checkpoint of past transcript rows. | MAIN watermark `turn_id=15` |
| `TextReader` | The single place a URL or filesystem path becomes plain text: a URL branch (fetch + HTML extraction, with plain-text suffixes passing through verbatim) or a file branch (mime detect + extract). Returns the full text verbatim — no truncation, no whitespace munging — and raises rather than returning empty on failure. | `TextReader(file_path_or_url).get_value()` |
| `ImageDescription` | The single place an image becomes text: a two-rung ladder, the configured vision provider first, an OCR fallback second. A description is mandatory — exhausting both rungs raises rather than returning empty. | `ImageDescription(file_path, prompt).get_value()` |
| `documents folder` | The flat on-disk file store at `data/documents/` (`screenshots/`, `uploads/` subdirs). Documents are plain files on disk, managed by file abilities — not database rows. | `data/documents/uploads/report.pdf` |
| `file index` | Filesystem-wide FTS5 full-text index (`FileIndexService`, its own `data/file_index.sqlite`) that backs `search_files`'s content search. Kept current by a filesystem watcher plus an hourly reconcile sweep. | `search_files(action='content', query='quarterly report')` |
| `search_files` | Built-in ability that locates files by name (glob), live content (grep), or the pre-built file index (content). | `search_files(action='content', query='kitchen renovation')` |
| `transcript_files` | Filepath-keyed link table joining a transcript turn to an attached file; composite `(transcript_id, path)` key, no `id` column. `path` is stored relative to the documents root. | `(transcript_id=42, path='uploads/report.pdf')` |
| `voice_transcript` | Transcript-keyed link table holding the synthesized speech for one settled reply; `(transcript_id, file_path)`, no `id` column, `ON DELETE CASCADE` to `transcript(id)`. A NULL `file_path` is a recorded failure — the pipeline exhausted its attempts and the row exists so it is never retried. | `(transcript_id=42, file_path='data/generated/voice/42.wav')` |
| `pre-synthesis` | Speaking a reply before it is asked for: when a turn settles, the settled row alone is synthesized in the background and stored, so pressing the speaker button plays a file rather than starting a synthesis. Only the settled row is spoken — mid-turn rows have no audio and never will. | `voice_transcript` WS frame: `pending` → `ready` \| `failed` |
| `policies` | The Allow / Ask / Deny gate per tool action. | `'allow'`, `'ask'`, `'deny'` |
| `vault` | Envelope-encrypted credential store (AES-256-GCM, DEK wrapped by password-derived KEK). | `kdf_iterations=600000` |
| `MessageProcessor` | The single flat orchestrator for every LLM turn (one per turn, per channel). | lifecycle signals: `working`, `done`, `tool_called` |
| `ExecutionTracker` | Per-turn object the `MessageProcessor` builds after turn resolution; owns the `turn_execution` row, is the sole emitter of lifecycle WS frames, and answers `should_stop()`. | `ExecutionTracker(config, turn_id)` |
| `should_stop` | Cooperative-stop predicate checked at each turn checkpoint; True once a cancel has been requested. Replaces the old in-memory cancel `Event`. | `if self.should_stop(): raise _TurnCancelled` |
| `ProcessorConfig` | Frozen dataclass parameterising one channel's `MessageProcessor`. | `UserConfig`, `DmnConfig`, `DiscoveryConfig` |
| `policy_channel` | Enum picking which policy rows apply. | `CHAT`, `SUBCONSCIOUS`, `EXTERNAL_AGENT` |
| `memory_seed` | Flag that fires a `memory.recall` at turn 0. | `UserConfig.memory_seed=True` |
| `RENDERS_HTML` | The `ProcessorConfig` class constant marking a channel whose output a human reads. One fact with three consequences that move together: `PromptService` appends the response-format contract (the model is told to emit `markup.PROMPT_TAGS`, never markdown), `MessageProcessor._format` converts and sanitizes the reply at persist time, and `DispatchService` assigns rich-media ordinals. Splitting them was the old defect — a channel sanitized but never instructed still leaks markdown, because the `markdown_to_html` fallback is inline-only and cannot rescue headings, lists, or tables. Distinct from `BROADCASTS_STATE`, which streams live turn progress; this governs the markup of the settled reply. | declared `True` by `UserConfig` and `ScheduledConfig` only |
| `SubconsciousWorker` | Idle-gated 5-minute cognition tick (consolidate, decay, patterns, synthesis…). | runs `_step_consolidate`, `_step_decay` |
| `SchedulerService` | Background poller that wakes every wall-clock minute and fires any enabled schedule whose day/hour/minute cron fields match — no separate "due" state to track. | `day=None, hour=3, minute=0` → every day at 03:00 |
| `WorldState` | Singleton holding the agent's "what's going on" cache. | `last_heartbeat_at`, `current_device_class` |
| `signal` | Typed world-state update absorbed from heartbeats / user messages. | `Signal(source='/health', kind='heartbeat')` |
| `heartbeat` | FE POST `/health` payload persisted to telemetry. | `{device.name, location.lat, locale.timezone}` |
| `USAGE_TYPE` | The `ProcessorConfig` class constant naming the spend bucket a channel's provider calls bill to; written verbatim to `llm_call_log.type`. Defaults to `'background'`, so spend reaches the user only by explicit declaration. Distinct from `policy_channel`, which gates tools — `DiscoveryConfig` is policy-gated as `CHAT` but bills as `'background'`. | `'foreground'` (the user's own conversation), `'background'` (everything Chalie runs on its own behalf) |
| `snapshot` | Full-instance backup (db, mcp_tools, pre-trained, vault key material). | a `.chalie-snapshot` file |
| `lane` | Independent conversation-surface identity — the spine, or a thread by its own `turn_id` — that busy/queue state is scoped to, so work in one never blocks a send in another. Keys the send queue as a string (`'main'` / `'t42'`) and, in the DOM busy contract, as a `(type, turn_id)` pair; the spine reserves a fixed pair since it has no `turn_id` of its own until a send resolves one. | queue: `'main'`, `'t42'` (thread 42); DOM: reserved pair (spine), `(type, 42)` (thread 42) |
| `chip` | FE rendering of a tool-call under an assistant message. | `{tool_name, summary}` |
| `card` | Rich-media payload rendered as a structured card. | `WeatherCard`, `SchedulerCard` |
| `segment` | FE parse of assistant content into text / rich blocks. | `text`, `rich` |
| `Endpoint` | ABC every migrated REST CRUD group subclasses (`api/endpoints/`); the base generates the routes, auth, and envelopes from the slug it is constructed with. | `GET /api/{slug}/all`, `POST /api/{slug}/-1` (create) |
| `Action` | `Endpoint` subclass for verb-shaped operations (`api/actions/<slug>/<verb>.py`); `all` is a reserved verb. | `POST /api/lists/items/<id>` |
| route map | `backend/api/routes.py` — the single table mounting every controller at its slug (and verb); controllers declare no path of their own. | `Skills("skills")`, `SkillCopy("skills", "copy")` |
| `envelope` | The uniform response shape built only by the `Response` DTO base. | `{success, result}`, error: `{success: false, result: [], error}` |
| `find_tools` | Discovery ability that surfaces tools the model can use. | `find_tools(query=['weather','valletta'])` |
| `internal_dev` | Env-var flag gating in-development features. | `CHALIE_INTERNAL_DEV='1'` |
| `garbage collector` | Hourly sweep that hard-deletes `transcript` rows once they are at least 90 days old and cited by no live episode, then the `tool_calls` rows left orphaned by that (or by any earlier) deletion. One owner, one age window for both tables. | `GarbageCollectionJob` fires at the top of every hour |