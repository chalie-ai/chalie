# Kill Topic Classifier — Thread-Scoped Context

**Status:** Planned
**Created:** 2026-03-27
**Motivation:** Cross-session context leakage via stale topic assignment (#1579). The topic classifier routes new messages to old topics, causing context assembly to inject stale data (e.g., week-old weather results presented as current). The memory pipeline (episodes, knowledge, semantic search) already handles cross-session recall — the topic layer is redundant and dangerous.

## Problem

The topic classifier fetches ALL topics from the database with no age filter. Freshness penalty is too weak (0.3 weight, asymptotes at 5 min). When a new message semantically matches an old topic, context assembly loads that topic's compaction and transcript — injecting stale data the LLM treats as current.

This caused #1579 (fabricated weather data) and likely #1577, #1575 (topic misassignment).

## Design

Replace topic-scoped context with thread-scoped context. The thread system already provides temporal segmentation (30min soft expiry, 4hr hard expiry). Cross-session recall is already handled by knowledge/episode semantic search.

**Before:** Message → Topic Classifier → Topic → Context Assembly (compaction + transcript by topic)
**After:** Message → Thread → Context Assembly (compaction + transcript by thread_id)

## What to delete

| File | Lines | What |
|------|-------|------|
| `services/topic_classifier_service.py` | ~400 | Topic classification + embedding similarity |
| `services/two_signal_boundary_service.py` | ~300 | Topic boundary detection |
| `services/recent_topic_service.py` | ~60 | Recent topic MemoryStore cache |
| `tests/test_topic_classifier_service.py` | ~500 | Tests |
| `tests/test_two_signal_boundary_service.py` | ~700 | Tests |
| Digest worker classification block | ~80 | `digest_worker.py` lines ~1880-1920 |

## What to change

### 1. Database: Re-key transcript and compaction to thread_id

```sql
-- topic_transcript: add thread_id column, migrate existing rows
ALTER TABLE topic_transcript ADD COLUMN thread_id TEXT;
-- Backfill: use topic as thread_id for existing rows (lossy but acceptable)
UPDATE topic_transcript SET thread_id = topic WHERE thread_id IS NULL;

-- topic_compactions: same treatment
ALTER TABLE topic_compactions ADD COLUMN thread_id TEXT;
UPDATE topic_compactions SET thread_id = topic WHERE thread_id IS NULL;
```

Consider renaming tables (`topic_transcript` → `transcript`, `topic_compactions` → `compactions`) in a follow-up to avoid confusion.

### 2. Context assembly: Use thread_id instead of topic

`context_assembly_service.py` — `_get_working_memory(topic)` becomes `_get_working_memory(thread_id)`:
- `compaction_service.get_compaction(thread_id)` instead of `get_compaction(topic)`
- `transcript_service.get_recent(thread_id, ...)` instead of `get_recent(topic, ...)`

### 3. Transcript service: Key by thread_id

`transcript_service.py` — All methods that accept `topic` param switch to `thread_id`.

### 4. Compaction service: Key by thread_id

`compaction_service.py` — `get_compaction(thread_id)`, `save_compaction(thread_id, ...)`.

### 5. Digest worker: Skip classification

Remove the classify block (~lines 1880-1920). The thread_id is already resolved earlier in the flow. Pass thread_id directly to context assembly and transcript storage.

### 6. Episodic consolidation: Trigger per-thread

`episodic_memory_worker.py` — Consolidation currently triggers on topic turn count. Switch to thread turn count.

### 7. Conversation phase service: Already thread-scoped

`conversation_phase_service.py` — Already keyed by thread_id. No changes needed.

### 8. Topics table: Keep but stop writing

Keep the `topics` table for historical reference but stop inserting new rows. The topic name can still be derived for display purposes via LLM summary if needed (cosmetic, not structural).

## What stays the same

- **Thread lifecycle** — soft/hard expiry unchanged
- **Memory pipeline** — episodes, knowledge, semantic search all work independently of topics
- **Context assembly semantic retrieval** — episode and knowledge search already uses the message embedding, not the topic
- **Conversation phase service** — already thread-scoped
- **Compaction logic** — same incremental summarization, just keyed by thread_id

## Risks

- **Transcript search across threads:** Currently `transcript_service.search(topic=None)` can search all topics. With thread-scoped data, cross-thread search still works (query all threads). No regression.
- **Long-running threads:** If a user talks for 8 hours straight without a thread expiry, the transcript grows large. Compaction already handles this — it summarizes older turns when token budget is exceeded.
- **Display:** Frontend may show topic names. Replace with thread timestamps or let the LLM generate a summary title per thread.

## Execution order

1. Add `thread_id` column to `topic_transcript` and `topic_compactions`, backfill
2. Update `transcript_service` and `compaction_service` to accept thread_id
3. Update `context_assembly_service` to pass thread_id
4. Update `digest_worker` to skip classification, pass thread_id
5. Update `episodic_memory_worker` to trigger per-thread
6. Delete `topic_classifier_service`, `two_signal_boundary_service`, `recent_topic_service` + tests
7. Run full test suite, fix any remaining topic references

## Issues resolved

- #1579 — Fabricated weather data (cross-session leakage)
- #1577 — Dog messages assigned to stale topic
- #1575 — Car messages assigned to unrelated topic
- #1558 — Phase state machine 'deepening' (test expectation, not this plan — but topic removal simplifies phase tracking)
