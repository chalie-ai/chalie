# Workers & Services Overview

## Queue Workers

| Worker | Type | Entry Point | Responsibilities | Notes |
|---|---|---|---|---|
| **Digest Worker** (`workers/digest_worker.py`) | `idle-busy` | `run.py` | Load configs, classify prompt, deterministic mode routing, mode-specific LLM generation, enqueue episodic memory job | Uses `ModeRouterService` for routing (~5ms), `FrontalCortexService` for generation. ACT mode triggers action loop then re-routes. |
| **Episodic Memory Worker** (`workers/episodic_memory_worker.py`) | `idle-busy` | Enqueued by Digest Worker | Reads raw conversation turns; consolidates into episodes when turn count >=5 or idle timeout triggers. Stores in SQLite. Triggers semantic consolidation. | |
| **Semantic Consolidation Worker** (`workers/semantic_consolidation_worker.py`) | `idle-busy` | Enqueued by Episodic Memory Worker | Extracts concepts + relationships from episodes. Matches existing (similarity > 0.85) or creates new. | |
| **Tool Worker** (`workers/tool_worker.py`) | `idle-busy` | Enqueued by ACT Loop | Background ACT loop execution via PromptQueue thread. Manages tool invocation and result handling. | |

## Services

| Service | Entry Point | Responsibilities | Notes |
|---|---|---|---|
| **REST API** (`workers/rest_api_worker.py`) | `run.py` | Flask REST API on port 8080. | |
| **Cognitive Drift Engine** (`services/cognitive_drift_engine.py`) | `run.py` | Default Mode Network — generates spontaneous thoughts during idle. Selects seed concepts, runs spreading activation, synthesizes via LLM, stores as drift gists. | Requires all queues idle + recent episodes. Fatigue budget prevents runaway drift. |
| **Idle Consolidation** (`services/idle_consolidation_service.py`) | `run.py` | Triggers semantic consolidation during idle periods. | |
| **Decay Engine** (`services/decay_engine_service.py`) | `run.py` | Periodic decay: episodic (0.05/hr), semantic (0.03/hr). Runs every 30min. | High-salience decays slower. |
| **Growth Pattern Service** (`services/growth_pattern_service.py`) | `run.py` | Tracks longitudinal communication style shifts. Compares current style against a slowly-updated EMA baseline to detect persistent changes in certainty, depth, challenge appetite, verbosity, and formality. Stores `growth_signal:{dim}` traits (category=core) when a shift persists 3+ consecutive cycles (90min+). 30min cycle. | Growth signals surface via `AdaptiveLayerService` as optional growth reflections in the response prompt (24h cooldown). Uses `StyleMetricsService` for deterministic 5-dimension measurements (~1ms, zero LLM). |
| **Routing Reflection** (`services/routing_reflection_service.py`) | `run.py` | Idle-time peer review of routing decisions via strong LLM (qwen3:14b). Stratified sampling, dimensional ambiguity analysis, anti-authority safeguards. | Consultant, not authority. Feeds pressure signals. |
| **Experience Assimilation** (`services/experience_assimilation_service.py`) | `run.py` | Converts tool results into episodic memory. 60s poll cycle. | |
| **Thread Expiry Service** (`services/thread_expiry_service.py`) | `run.py` | Expires stale conversation threads. 5min poll cycle. | |
| **Scheduler Service** (`services/scheduler_service.py`) | `run.py` | Fires due reminders and scheduled tasks. 60s poll cycle. | |
| **Autobiography Synthesis Service** (`services/autobiography_synthesis_service.py`) | `run.py` | Synthesizes user narrative from interactions. 6h cycle. | |
| **Profile Enrichment Service** (`services/profile_enrichment_service.py`) | `run.py` | Enriches tool capability profiles from execution data. 6h cycle. | |
| **Temporal Pattern Service** (`services/temporal_pattern_service.py`) | `run.py` | Mines behavioral patterns from interaction timestamps — hour-of-day peaks, day-of-week peaks, topic-time clusters. Stores as `behavioral_pattern` user traits. 24h cycle, 5min startup warmup. | Uses generalized time labels ("evenings" not "10-11pm") for privacy. |

## Worker Base

**Worker Base** (`services/worker_base.py`) — Base class for all workers. Provides `_update_shared_state` helper to merge state into shared dict; manages `state`, `job_count`, `pid`.

## Registration

Each worker is registered in `run.py` via `WorkerManager.register_worker` (queue workers) or `WorkerManager.register_service` (services). Queue names correspond to key prefixes defined in `configs/connections.json`.

## Worker Lifecycle

```
[run.py] → spawn thread → [WorkerBase.__init__]
    → create MemoryStore/DB connections
    → enter main loop
        → poll queue / sleep interval
        → process job
        → _update_shared_state
    → on exception: log + continue (unless fatal)
    → on SIGTERM: set shutdown flag → exit loop → cleanup

[run.py] health check (every 30s):
    → for each registered worker:
        → if process.is_alive() → ok
        → if dead → log warning → respawn
```

### Health Check & Restart

The `WorkerManager` in `run.py` runs a health check loop every 30 seconds:
- Iterates all registered workers and services
- Dead processes are respawned with the same configuration
- Restart count is tracked per worker; excessive restarts trigger a warning log
- On `SIGTERM`/`SIGINT`, the manager sends shutdown signals to all children and waits for graceful exit

### Per-Worker Configuration

| Worker / Service | Interval | Jitter | Notes |
|---|---|---|---|
| Digest Worker | Queue-driven | N/A | Blocks on `BRPOP` from prompt queue |
| Episodic Memory | Queue-driven | N/A | Blocks on `BRPOP` from episodic-memory queue |
| Tool Worker | PromptQueue | N/A | Managed by PromptQueue thread |
| Persistent Task Worker | 30min cycle | ±30% (0.7–1.3x) | Bounded ACT loop per cycle |
| Cognitive Drift | Idle-triggered | N/A | Only runs when all queues are empty |
| Decay Engine | 30min | None | Fixed interval |
| Scheduler Service | 60s poll | None | Checks `due_at <= NOW()` |
| Thread Expiry | 5min poll | None | Expires threads with no activity |
| Autobiography Synthesis | 6h cycle | None | Narrative synthesis |
| Triage Calibration | 24h cycle | None | Scoring/learning signals |
| Profile Enrichment | 6h cycle | None | Tool profile enrichment |
| Curiosity Pursuit | 6h cycle | None | Explores curiosity threads |
| Temporal Pattern Service | 24h cycle | None | 5min startup warmup; mines interaction timestamps |

### Failure Handling

- **Queue workers** (digest, episodic): Catch exceptions per-job, log, and continue polling. A single bad job never takes down the worker.
- **Polling services** (scheduler, thread expiry, decay): Catch exceptions per-cycle, log, and sleep until the next interval.
- **LLM-dependent workers**: If the LLM returns invalid JSON, the worker logs the raw response and moves on. No retry — the next cycle will pick up any missed work.
- **Fatal errors** (SQLite/MemoryStore error): The worker crashes. `run.py` detects the dead thread and respawns it.

### Graceful Shutdown

On `SIGTERM` or `SIGINT`:
1. `run.py` sets a global shutdown flag
2. Each worker checks the flag at the top of its loop
3. Queue workers finish processing their current job, then exit
4. Service workers complete their current cycle, then exit
5. `run.py` waits up to 10s for all children, then sends `SIGKILL` to stragglers
