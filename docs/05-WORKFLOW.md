# System Workflow

## Overview
The application processes user prompts through a pipeline of workers, queues, and services. Each component is responsible for a single concern, following the **Single Responsibility Principle**.

1. **Consumer (`consumer.py`)** – Supervises worker processes and tracks shared state. Receives raw prompts via REST API entry point and enqueues them into the `prompt-queue`.
2. **Prompt Queue (`services/prompt_queue.py`)** – A Redis-backed queue that holds prompt jobs. Workers consume jobs from this queue.
3. **Digest Worker (`workers/digest_worker.py`)** – Primary worker that:
   - Loads configuration for the classifier, mode router, and mode-specific generation prompts.
   - Calls the **classifier** (embedding-based, deterministic) to determine the topic and confidence.
   - Persists the prompt to a conversation thread via `ThreadConversationService` (Redis-backed).
   - Tracks session activity through `SessionService` and triggers episodic memory generation when a topic switch is detected.
   - Collects routing signals and runs **deterministic mode routing** (~5ms) via `ModeRouterService`.
   - Generates a response using a **mode-specific prompt** via `FrontalCortexService`:
     - ACT mode → action loop → re-route → terminal response
     - IGNORE → empty response (no LLM call)
     - RESPOND/CLARIFY/ACKNOWLEDGE → single LLM call with mode-specific prompt
   - Enqueues a job to the `memory-chunker-queue`.
4. **Memory Chunker Worker (`workers/memory_chunker_worker.py`)** – Enriches individual exchanges with *memory chunks*:
   - Loads the current world state from `WorldStateService`.
   - Generates a memory chunk via the **memory-chunker** LLM prompt.
   - Stores the enriched exchange back into the conversation thread (Redis).
5. **Episodic Memory Worker (`workers/episodic_memory_worker.py`)** – Builds episodes from a sequence of exchanges:
   - Waits for all memory chunks to be available.
   - Formats a session context for the **episodic-memory** LLM prompt.
   - Stores the resulting episode in PostgreSQL through `EpisodicStorageService`.
6. **Experience Assimilation Service** – Converts tool results into episodic memory. Runs on 60s poll cycle.
7. **Cognitive Drift Engine** (`services/cognitive_drift_engine.py`) – Default Mode Network service. During idle periods (all queues empty), generates spontaneous thoughts via spreading activation and LLM synthesis. Stores as drift gists that surface in frontal cortex context.
8. **Routing Stability Regulator** (`services/routing_stability_regulator_service.py`) – Single authority for mode router weight mutation. Runs on 24h cycle, reads pressure signals, applies bounded corrections.
9. **Routing Reflection Service** (`services/routing_reflection_service.py`) – Idle-time peer review of routing decisions via strong LLM. Feeds dimensional analysis into pressure signals consumed by the regulator.
10. **Scheduler Service** – Fires due reminders and scheduled tasks. Runs on 60s poll cycle.
11. **Thread Expiry Service** – Expires stale conversation threads. Runs on 5min poll cycle.
12. **Autobiography Synthesis Service** – Synthesizes user narrative from interactions. Runs on 6h cycle.
13. **Triage Calibration Service** – Scores triage/routing correctness. Runs on 24h cycle.
14. **Profile Enrichment Service** – Enriches tool capability profiles. Runs on 6h cycle.
15. **Redis / PostgreSQL** – Redis holds queue topics and small runtime state. PostgreSQL stores long-term episodic memories, semantic concepts, and routing decisions.

### Key Decisions
- **Shared State** – `WorkerBase._update_shared_state` merges per-worker updates into a shared dictionary managed by the `WorkerManager`. This avoids global locks and keeps the worker pool lightweight.
- **Deterministic Routing** – Mode selection is decoupled from LLM generation. A mathematical router scores modes using observable signals (~5ms), eliminating the previous approach where the LLM did both mode selection and response generation in a single ~15s call.
- **Mode-Specific Prompts** – Each mode (RESPOND, CLARIFY, ACKNOWLEDGE, ACT, IGNORE) has its own focused prompt template (~30-80 lines each), replacing the old combined 240-line prompt.
- **Single Authority** – Multiple monitors observe routing quality but only the `RoutingStabilityRegulator` (24h cycle) mutates weights, preventing tug-of-war between independent monitors.
- **Confidence Calculation** – `ThreadConversationService._calculate_new_confidence` uses a bounded reinforcement formula `new = current + (new_confidence - current) * 0.5`.
- **Thread Context** – Conversation threads are managed in Redis with expiry, not persistent files. Thread state includes active topic, confidence, and conversation history.
- **Error Handling** – All workers catch JSON decoding errors from LLM responses and log meaningful messages instead of crashing.

### Error Recovery Per Pipeline Stage

| Stage | Failure Mode | Recovery |
|---|---|---|
| Classification | Embedding service down | Falls back to default topic ("general") with low confidence |
| Mode Routing | LLM tiebreaker fails | Uses deterministic scores only (no tiebreak) |
| Generation | LLM timeout / malformed JSON | Retries once; if still fails, returns a generic "I had trouble thinking" message |
| Memory Chunker | LLM returns invalid JSON | Re-raises `JSONDecodeError`; exchange is stored without memory enrichment |
| Episodic Memory | Not all chunks available | Waits with backoff; proceeds with available chunks after deadline |
| Tool Worker | Tool container crash | Critic evaluates partial result; escalates to user if consequential |

### Latency Budget (Typical)

| Component | Target | Notes |
|---|---|---|
| Topic Classification | <5ms | Embedding lookup, deterministic |
| Mode Routing | <5ms | Mathematical scoring, no LLM |
| LLM Tiebreaker (rare) | ~200ms | Only when top-2 modes within margin |
| Response Generation | 2–15s | Depends on model and context length |
| Memory Chunker | 1–5s | Background, non-blocking |
| Episodic Memory | 2–8s | Background, non-blocking |

### ACT Loop Detail

When the mode router selects **ACT**, the digest worker enters an autonomous action loop:

1. **Dispatch** — `ActDispatcherService` routes each action to the registered innate skill handler (recall, memorize, schedule, etc.)
2. **Execute** — Handler runs with a timeout (default 10s); result includes structured output + confidence estimate
3. **Critic** — `CriticService` evaluates the result via a lightweight LLM call. Safe actions (recall, memorize) get silent correction. Consequential actions (schedule, persistent_task) pause for user confirmation if confidence is low
4. **Re-plan** — If the critic suggests a different approach, the loop re-routes. Otherwise, the loop continues with the next action or exits to a terminal mode (RESPOND/CLARIFY)
5. **Budget** — Each action costs fatigue points (recall=1.0, memorize=0.8, introspect=0.5). The loop exits when the budget is exhausted

### SSE Lifecycle

The `/chat` endpoint uses Server-Sent Events for real-time streaming:

1. Client sends `POST /chat` with `{text, source}`
2. Server returns `text/event-stream` with `X-Request-ID` header
3. Background thread runs `digest_worker` with the request UUID
4. Server listens on `sse:{uuid}` Redis pub/sub channel
5. Events flow: `status:processing` → `status:thinking` → `message:{response}` → `done`
6. Keepalive pings every 15s; status updates every 20s; 360s hard timeout
7. If background thread completes without pub/sub (race condition), server polls `output:{request_id}` key as fallback

## Flow Diagram
```
[Listener] → [Consumer] → [Prompt Queue] → [Digest Worker] →
    |--(classification)--> [TopicConversationService] → [Conversation JSON]
    |--(routing)---------> [ModeRouterService] → {mode, confidence, scores} (~5ms)
    |--(generation)------> [FrontalCortexService] → mode-specific prompt → LLM
    |                      (ACT mode → action loop → re-route → terminal)
    |--(memory_chunk_job)→ [Memory Chunker Queue] → [Memory Chunker Worker] →
        [Conversation JSON] (enriched)
    |--(episode_job)----> [Episodic Memory Queue] → [Episodic Memory Worker] →
        PostgreSQL Episodes Table

[Routing Stability Regulator] ← reads routing_decisions (24h cycle)
    → adjusts configs/generated/mode_router_config.json

[Routing Reflection Service] ← reads reflection-queue (idle-time)
    → writes routing_decisions.reflection → feeds pressure to regulator
```
