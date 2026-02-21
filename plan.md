# Fix: Scheduler Reminders & Cognitive Drift Never Reach User

## The Intended E2E Flow

Both scheduler reminders and cognitive drift follow the same delivery pipeline once they produce output:

```
[Source]  ──enqueue──▶  prompt-queue (Redis/RQ)
                              │
                     digest_worker picks up
                              │
                     mode router ▶ frontal cortex ▶ generate response
                              │
                     orchestrator ▶ OutputService.enqueue_text()
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    Redis pub/sub        Web Push       notifications:recent
   "output:events"    (VAPID/webpush)     (catch-up buffer)
              │                               │
              ▼                               ▼
    /events/stream SSE  ◀─────── drain on reconnect
              │
              ▼
    Frontend EventSource listener
              │
    renderer.appendChalieForm() ──▶ chat UI
```

**Scheduler path:** `scheduler_service` polls `scheduled_items` every 60s → `_fire_item()` → `PromptQueue.enqueue()` → digest_worker → output pipeline

**Drift path:** `cognitive_drift_engine` runs every ~5min → selects seed concept → LLM synthesizes thought → `ActionDecisionRouter` → if COMMUNICATE passes 3 gates (quality, timing, engagement) → `PromptQueue.enqueue(source='proactive_drift')` → digest_worker → output pipeline

---

## Root Causes — Three Independent Breaks

### BUG 1: Scheduler — `PromptQueue` instantiated without `worker_func` (fatal)

**File:** `backend/services/scheduler_service.py:137`

```python
queue = PromptQueue(queue_name="prompt-queue")   # ← BUG: no worker_func
queue.enqueue({...})                              # ← raises ValueError immediately
```

`PromptQueue.enqueue()` at `prompt_queue.py:36` checks:
```python
if not self._worker_func:
    raise ValueError("No worker function configured for this queue")
```

Every `_fire_item()` call throws, gets caught by the except on line 114, and the item is marked `status='failed'`.

**Database proof:** Both scheduled reminders in the DB have `status='failed'` and `last_fired_at=NULL`:
| id | message | due_at | status | last_fired_at |
|---|---|---|---|---|
| 347aab7c | Drink water | 2026-02-21 15:04:00 | **failed** | NULL |
| 2c9fe814 | Drink water | 2026-02-21 15:48:00 | **failed** | NULL |

**Same bug in:** `backend/services/tool_registry_service.py:181` (cron tool results also can't enqueue).

### BUG 2: Cognitive Drift — Zero semantic concepts in DB (starved)

**Database proof:** `SELECT COUNT(*) FROM semantic_concepts WHERE deleted_at IS NULL` → **0 rows**

The drift engine's seed selection has 4 strategies (decaying, recent, salient, random) — **all query `semantic_concepts`** and all return nothing. Logs confirm: `"No viable seed found after 3 attempts"` repeated every cycle.

Without a seed, no thought is ever generated, so the `ActionDecisionRouter` (COMMUNICATE/REFLECT/NOTHING) is never invoked, and no proactive drift messages are ever created.

**Root cause of zero concepts:** The semantic consolidation pipeline that should create concepts from episodes has never produced output. Needs investigation — either it's not running, failing silently, or its gates are blocking all episodes.

### BUG 3: Web Push — VAPID key deserialization fails (broken last mile)

**Log proof:**
```
[Push] Send error: Could not deserialize key data. The data may be in an incorrect
format... ASN.1 parsing error: invalid length
```

Even if bugs 1 and 2 were fixed, `send_push_to_all()` would crash when trying to sign the push payload. The PEM private key stored in Redis is likely corrupted during JSON serialization (newlines in PEM are critical for ASN.1 parsing).

---

## Fix Plan

### Step 1 — Fix scheduler `PromptQueue` instantiation

**File:** `backend/services/scheduler_service.py`

Change `_fire_item()` to pass the worker function:
```python
from workers.digest_worker import digest_worker   # add import
queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
```

This matches how it's done correctly in `communicate_action.py` and `tool_worker.py`.

Apply the same fix in `backend/services/tool_registry_service.py` for cron tool results.

### Step 2 — Re-fire the failed reminders

Reset the two failed items back to pending so they fire on the next poll:
```sql
UPDATE scheduled_items SET status='pending' WHERE status='failed';
```

### Step 3 — Investigate semantic consolidation pipeline

Determine why zero concepts exist despite 12 episodes:
- Check if the semantic consolidation worker is running in consumer.py
- Check if episodes have been marked for consolidation
- Check if the LLM call for concept extraction is failing
- This unblocks cognitive drift seed selection

### Step 4 — Fix VAPID key storage/retrieval

**File:** `backend/api/push.py` — `_get_vapid_keys()`

Investigate how the PEM key is stored in Redis. Likely fix: ensure newlines in the PEM string survive JSON round-trip.

### Step 5 — Add `source` field to SSE event payload

**File:** `backend/services/output_service.py:84-88`

Add `'reminder'` and `'task'` to the source-type map so the frontend can distinguish these from regular responses:
```python
source_type_map = {
    'proactive_drift': 'drift',
    'tool_followup': 'tool_followup',
    'reminder': 'reminder',
    'task': 'scheduled_task',
}
```

Register corresponding event listeners in `frontend/interface/app.js` `_connectDriftStream()`.

### Step 6 — Frontend: don't drop reminder/drift events during active chat

**File:** `frontend/interface/app.js`

The `_handleEvent()` currently ignores all `response`-type events while `_isSending` is true. With step 5 giving reminders their own event type, this is automatically fixed. But also add an explicit guard: never drop `reminder` or `drift` events regardless of `_isSending` state.

---

## Summary

| Bug | Impact | Fix |
|---|---|---|
| PromptQueue missing `worker_func` | All reminders fail silently — **this is the primary blocker** | One-line fix in `scheduler_service.py` + `tool_registry_service.py` |
| Zero semantic concepts | Drift engine starved, never generates thoughts | Investigate semantic consolidation worker |
| VAPID key corrupted | Push notifications fail at delivery | Fix PEM storage format in `push.py` |
| Reminders typed as generic 'response' | Dropped during active chat, indistinguishable in UI | Add event type mapping + frontend listener |
