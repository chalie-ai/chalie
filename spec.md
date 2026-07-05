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
- `MP.current_transcript_id` is the in-memory binder every `tool_calls` row anchors to. It starts at the turn's input row and moves ONLY when an assistant row is actually written — assistant rows are written only when the LLM emits text (or on the turn's terminal step), never for tool-call-only steps.

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
	1. If the LLM returned text, MP creates a transcript row `role=assistant` and moves `MP.current_transcript_id` onto it. A tool-call-only response writes NO row — the binder stays where it is, so consecutive tool-only steps collapse onto the row that opened them (no empty assistant rows)
	2. If the LLM returns tool calls; 
		1. we invoke the tools
		2. Each `tool_calls` row binds to `MP.current_transcript_id`
		3. Each tool call emits its WS frame (state `started` / `done` / `error` — see TKT-1233)
		4. Loop back to step 4 on the same `MessageProcessor` instance (same config, same `turn_id`) for the next LLM API call
5. The LLM stopped emitting tool calls — the terminal step. Its transcript row is ALWAYS written, even with empty text (it is the turn's settle row — the spine's settle0/working state keys on it). We close the `MessageProcessor` instance and emit the WS message with state `done`

## BE: API Flow (Message in thread)
This floww describes a message arriving from a thread

1. A message is received via `POST /api/thread/<turn_id>` with the parameter `type`.
API endpoint handler resolves the config via `ConfigTypeEnum.get_by_type({type})`; invalid type throws `Invalid type provided`.

2. API endpoint handler initializes a new `MessageProcessor` like so: `new MessageProcessor(conf=request_config, turn_id={turn_id}, raw_input={request_message})`. A `turn_id` that names no existing turn throws `Invalid turn_id specified` — no phantom turn is minted.

3. On initialization the `MessageProcessor`;
	1. Must create a new row in `transcript` table with the supplied `turn_id` - `{turn_id}` and set `channel=_self.config.channel, role=_self.config.role, content=_self.raw_input`
	2. MP must emit the Websocket event with state `working` and properties: `turn_id` & `config.type`
	3. MP must fire the pre-LLM API call tool calls as needed (based on the `ProcessorConfig` loaded)
   
4. `MessageProcessor` sends LLM API call — identical to the new-turn flow: text moves `MP.current_transcript_id` onto a fresh assistant row, tool-call-only steps write nothing and bind their `tool_calls` to the current binder, each call emits its WS frame, loop.
5. Terminal step (no tool calls): its transcript row is always written; we close the `MessageProcessor` instance and emit the WS message with state `done`

## FE: Visual Feedback (New Turn in Main Spine)
This flow describes what the interface does when the user sends a message from the main input dock. The feed is filled only by API responses — no optimistic echo.

1. `InputDock` submits the message via `session.sendMessage(text, files, threadId=null, type)`.

2. If the main lane is already busy, the message is queued per-lane (`useQueueStore`) and auto-sent when the lane frees — flow ends here.

3. `sendMessage` claims the `main` lane in the session store;
	1. Lane presence is the busy state — it drives the PresenceBar pulse and the feed's live-turn spinner
	2. The composed text is kept on the lane so a stop can restore it into the dock

4. The message is POSTed as a multipart form (`text`, `type`, `files`) to `POST /api/thread/-1`
	1. On HTTP 200 the body `{turn_id, type}` binds `lane.liveTurnId = turn_id` — the handle every later step keys on
	2. On failure the error surfaces as the dock toast and the lane is released — no signal will ever arrive for this turn

5. While the lane is live, the stop button calls `session.requestStop(turn_id)`: drops the turn's block (`dropLiveTurn`), fires a best-effort `DELETE /api/thread/<turn_id>`, and restores the lane's text into the dock.

6. Turn blocks enter the feed (`useConversationFeed`) only via `GET /api/thread/<turn_id>` / `GET /api/threads/batch`; transient visuals — `working` spinner, live tool pills, unseen-`done` markers — live beside the buffer, and a fetched row's persisted `tool_calls` supersede its live pills.

## FE: Visual Feedback (New Turn in Thread View)
This flow describes a reply sent from the slide-over thread panel. It is the main-spine flow with these differences:

1. The panel's `InputDock` carries the open thread's id: `session.sendMessage(text, files, threadId={turn_id}, type)`.

2. The lane key is `t{turn_id}` instead of `main` — each thread is an independent lane, so a working thread never blocks the spine or other threads.

3. The busy gate additionally checks the feed's working flag for that turn (`isTurnWorking`), since a thread's in-flight state outlives the lane record.

4. The POST goes to `POST /api/thread/<turn_id>`; the 200 body echoes the same `turn_id` back.

5. Working/stop/failure behavior is identical, keyed on the thread's `turn_id`; the spinner and live tool pills render inside the thread panel.

6. On settle, a thread the user is not viewing keeps a standing `done` marker (the Activity dock's blue state) until the panel is opened; the open thread just drops its spinner.

