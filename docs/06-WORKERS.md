# Workers & Services Overview

## Queue Workers

| Worker | Type | Entry Point | Responsibilities | Notes |
|---|---|---|---|---|
| **Digest Worker** (`workers/digest_worker.py`) | `idle-busy` | `consumer.py` | Load configs, classify prompt, deterministic mode routing, mode-specific LLM generation, enqueue memory chunk job | Uses `ModeRouterService` for routing (~5ms), `FrontalCortexService` for generation. ACT mode triggers action loop then re-routes. |
| **Memory Chunker Worker** (`workers/memory_chunker_worker.py`) | `idle-busy` | Enqueued by Digest Worker | Enriches a single exchange with a memory chunk via the **memory-chunker** LLM and stores it back into the conversation file. | Handles JSON decoding errors gracefully. |
| **Episodic Memory Worker** (`workers/episodic_memory_worker.py`) | `idle-busy` | Enqueued by Digest Worker | Builds episodes from a sequence of exchanges, waits for memory chunks, generates via LLM, stores in PostgreSQL. Triggers semantic consolidation. | |
| **Semantic Consolidation Worker** (`workers/semantic_consolidation_worker.py`) | `idle-busy` | Enqueued by Episodic Memory Worker | Extracts concepts + relationships from episodes. Matches existing (similarity > 0.85) or creates new. | |
| **Telegram Worker** (`workers/telegram_worker.py`) | `service` | `consumer.py` | Converts Telegram messages to prompts and enqueues into `prompt-queue`. Sends responses back to Telegram. | |

## Services

| Service | Entry Point | Responsibilities | Notes |
|---|---|---|---|
| **REST API** (`workers/rest_api_worker.py`) | `consumer.py` | Flask REST API on port 8080. | |
| **Cognitive Drift Engine** (`services/cognitive_drift_engine.py`) | `consumer.py` | Default Mode Network — generates spontaneous thoughts during idle. Selects seed concepts, runs spreading activation, synthesizes via LLM, stores as drift gists. | Requires all queues idle + recent episodes. Fatigue budget prevents runaway drift. |
| **Idle Consolidation** (`services/idle_consolidation_service.py`) | `consumer.py` | Triggers semantic consolidation during idle periods. | |
| **Decay Engine** (`services/decay_engine_service.py`) | `consumer.py` | Periodic decay: episodic (0.05/hr), semantic (0.03/hr). Runs every 30min. | High-salience decays slower. |
| **Topic Stability Regulator** (`services/topic_stability_regulator_service.py`) | `consumer.py` | Adaptive tuning of topic switching parameters. 24h cycle. | |
| **Routing Stability Regulator** (`services/routing_stability_regulator_service.py`) | `consumer.py` | Single authority for mode router weight mutation. 24h cycle. Reads pressure signals, applies bounded corrections (max ±0.02/day), 48h cooldown per parameter. Closed-loop control (reverts ineffective adjustments). | Persists to `configs/generated/mode_router_config.json`. |
| **Routing Reflection** (`services/routing_reflection_service.py`) | `consumer.py` | Idle-time peer review of routing decisions via strong LLM (qwen3:14b). Stratified sampling, dimensional ambiguity analysis, anti-authority safeguards. | Consultant, not authority. Feeds pressure signals to regulator. |

## Worker Base

**Worker Base** (`services/worker_base.py`) — Base class for all workers. Provides `_update_shared_state` helper to merge state into shared dict; manages `state`, `job_count`, `pid`.

## Registration

Each worker is registered in `consumer.py` via `WorkerManager.register_worker` (queue workers) or `WorkerManager.register_service` (services). Queue names correspond to Redis topics defined in `configs/connections.json`.
