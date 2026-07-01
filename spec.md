# Message Flow / Threading

## System Context
- `ConfigTypeEnum` (`USER`, `SCHEDULED`) is the API-facing config identifier. `get_by_type(type)` resolves a type string to a `ProcessorConfig` instance or throws `Invalid type supplied`. `channel` is backend-internal storage only.
- `ProcessorConfig.type()` returns its `ConfigTypeEnum` member.
- `TranscriptService(config, turn_id)` is instance-based; every query auto-filters by `config.channel`. `MessageProcessor` holds `self.ts`.
- `MessageProcessor(config, turn_id=-1, raw_input, metadata)` — constructor pre-allocates the input transcript row via `make_row_id()`, exposes `get_meta_data()` synchronously, and `run()` spawns the turn on a daemon thread.
- `turn_id == -1` → new thread; `MAX(turn_id)+1` allocated per channel. A supplied `turn_id` → append to that thread.
- `_forked` is derived from `turn_id` presence — no external flag.

## API > BE Flow (New Turn)
This flow describes a message arriving from the main spine

1. A message is received via `POST /api/thread` with the parameter `type`.
API endpoint handler resolves the config via `ConfigTypeEnum.get_by_type({type})`; invalid type throws `Invalid type supplied`.

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
		4. Initialize a new `MessageProcessor` with the exact same configs we ran in step 2 with the difference:
           1. `turn_id` => we use from step 3.1
5. Now that LLM stopped emitting tool calls, we close the `MessageProcessor` instance and emit the WS message with state `done`

## API > BE Flow (Message in thread)
