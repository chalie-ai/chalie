# Message Flow / Threading

## Vocabulary
Read the vocabulary to understand the systems' terminology
[VOCABULARY.md](VOCABULARY.md)

## System Context
- `ConfigTypeEnum` (`USER`, `SCHEDULED`, `DISCOVERY`) is the API-facing config identifier. `get_by_type(type)` resolves a type string to a `ProcessorConfig` instance or throws `Invalid type provided`. `channel` is backend-internal storage only.
- `ProcessorConfig.type()` returns its `ConfigTypeEnum` member.
- `TranscriptService(config, turn_id)` is instance-based; every query auto-filters by `config.channel`. `MessageProcessor` holds `self.ts`.
- `MessageProcessor(config, turn_id=-1, raw_input, metadata)` — constructor pre-allocates the input transcript row via `make_row_id()`, exposes `get_meta_data()` synchronously, and `run()` spawns the turn on a daemon thread.
- `turn_id == -1` → new thread; `MAX(turn_id)+1` allocated per channel. A supplied `turn_id` → append to that thread; if it names no existing turn the constructor throws `Invalid turn_id specified`.
- `_forked` is derived from `turn_id` presence — no external flag.

## BE: API Flow (New Turn)
This flow describes a message arriving from the main spine

1. A message is received via `POST /api/thread/-1` with the parameter `type` (`-1` is the unset sentinel — same endpoint as the in-thread flow, one chokepoint).
API endpoint handler resolves the config via `ConfigTypeEnum.get_by_type({type})`; invalid type throws `Invalid type provided`.

2. API endpoint handler initializes a new `MessageProcessor` like so: `new MessageProcessor(conf=request_config, turn_id=-1, raw_input={request_message})`

3. On initialization the `MessageProcessor`;
	1. Must create a new row in `transcript` table and generate a `turn_id` - `{turn_id}` and set `channel=_self.config.channel, role=_self.config.role, content=_self.raw_input`
	2. MP must emit the Websocket event with state `working` and properties: `turn_id` & `config.type`
	3. MP must fire the pre-LLM API call tool calls as needed (based on the `ProcessorConfig` loaded)
   
4. `MessageProcessor` sends LLM API call
	1. MP must create a transcript row where `role=assistant` (could contain an assistant text message or not, doesn't matter)
	2. If the LLM returns tool calls; 
		1. we invoke the tools
		2. We attach the `tool_calls` to the `transcript` row id from step 4.1
		3. Each tool call emits the WS message `tool_called` when it starts and `tool_done` on completion
		4. Loop back to step 4 on the same `MessageProcessor` instance (same config, same `turn_id`) for the next LLM API call
5. Now that LLM stopped emitting tool calls, we close the `MessageProcessor` instance and emit the WS message with state `done`

## BE: API Flow (Message in thread)
This floww describes a message arriving from a thread

1. A message is received via `POST /api/thread/<turn_id>` with the parameter `type`.
API endpoint handler resolves the config via `ConfigTypeEnum.get_by_type({type})`; invalid type throws `Invalid type provided`.

2. API endpoint handler initializes a new `MessageProcessor` like so: `new MessageProcessor(conf=request_config, turn_id={turn_id}, raw_input={request_message})`. A `turn_id` that names no existing turn throws `Invalid turn_id specified` — no phantom turn is minted.

3. On initialization the `MessageProcessor`;
	1. Must create a new row in `transcript` table with the supplied `turn_id` - `{turn_id}` and set `channel=_self.config.channel, role=_self.config.role, content=_self.raw_input`
	2. MP must emit the Websocket event with state `working` and properties: `turn_id` & `config.type`
	3. MP must fire the pre-LLM API call tool calls as needed (based on the `ProcessorConfig` loaded)
   
4. `MessageProcessor` sends LLM API call
	1. MP must create a transcript row where `role=assistant` (could contain an assistant text message or not, doesn't matter)
	2. If the LLM returns tool calls; 
		1. we invoke the tools
		2. We attach the `tool_calls` to the `transcript` row id from step 4.1
		3. Each tool call emits the WS message `tool_called` when it starts and `tool_done` on completion
		4. Loop back to step 4 on the same `MessageProcessor` instance (same config, same `turn_id`) for the next LLM API call
5. Now that LLM stopped emitting tool calls, we close the `MessageProcessor` instance and emit the WS message with state `done`

## FE: Visual Feedback (New Turn in Main Spine)
This flow describes what the interface does when the user sends a message from the main input dock. Doctrine: the feed buffer is filled ONLY by API responses — no optimistic echo, no payload-driven rendering.

1. `InputDock` submits via `session.sendMessage(text, files, threadId=null, type)`; empty text with no files is a no-op; files without text fall back to the `[File attached]` placeholder.
2. If the main lane is already busy (`isLaneBusy`), the message is enqueued per-lane (`useQueueStore`) and auto-sent when the lane frees — flow ends here.
3. `sendMessage` claims the lane: `lanes['main'] = { liveTurnId: null, userText, type }`. Lane presence IS the busy state — `isSending` drives the PresenceBar pulse and the feed's live-turn spinner. The composed text is kept on the lane for restore-on-stop.
4. The message is POSTed as a multipart form (`text`, `type`, repeated `files`) to `POST /api/thread/-1`.
5. On HTTP 200 the body `{turn_id, type}` binds `lane.liveTurnId = turn_id` — the handle every later step keys on (block rendering, stop button), held synchronously from the POST response.
6. On failure (offline / non-200) `_onSendFailure` surfaces the error as the dock toast, clears the turn's working flag and releases the lane — no signal will ever arrive for this turn.
7. While the lane is live, the stop button calls `session.requestStop(turn_id)`: drops the turn's block and all its visual state (`dropLiveTurn`), fires a best-effort `DELETE /api/thread/<turn_id>`, and restores the lane's captured text into the dock.
8. Rendering (`useConversationFeed(type)`): turn blocks enter the reactive buffer only via `GET /api/thread/<turn_id>` / `GET /api/threads/batch` (`fetchTurn` → `upsertTurn`, guarded by a monotonic max-row-id version check so a stale response never overwrites a newer block). Transient visuals — `working` spinner flags, live tool pills with elapsed timers, unseen-`done` markers — live beside the buffer and never mutate turn data; persisted `tool_calls` on a fetched row supersede that row's live pills.

## FE: Visual Feedback (New Turn in Thread View)
This flow describes a reply sent from the slide-over thread panel. It is the main-spine flow with these differences:

1. The panel's `InputDock` carries the open thread's id, so the submit is `session.sendMessage(text, files, threadId={turn_id}, type)`.
2. The lane key is `t{turn_id}` instead of `main` and it is claimed with `liveTurnId` preset to that id — each thread is an independent lane, so a working thread never blocks the spine (or other threads) and its queue drains independently.
3. The busy gate additionally checks the feed's working flag for that turn (`isTurnWorking`), since a thread has a stable id whose in-flight state outlives the lane record.
4. The POST goes to `POST /api/thread/<turn_id>`; the 200 body echoes the same `turn_id` back.
5. Working/stop/failure behavior is identical, keyed on the thread's `turn_id`; the working spinner and live tool pills render inside the thread panel's turn view.
6. On settle, a thread the user is NOT currently viewing keeps a standing `done` marker (the Activity dock's blue state) until the panel is opened (`seenThread` clears it); the open thread just drops its spinner.