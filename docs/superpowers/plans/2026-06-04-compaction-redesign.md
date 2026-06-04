# Compaction Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tangled two-threshold compaction control flow with a single pre-flight + 413 retry loop, make `MessageProcessor` own a bound `Providers` instance (param-free methods scaffolded from `mp`), move the history-compaction watermark into the `transcript` table, and retire the `Providers` singleton by converting its three remaining non-`mp` callers.

**Architecture:** `mp` is the orchestrator and owns `self.providers = Providers(self)`. `providers.send()` pre-flights (clamped 90%/8k-headroom check), sends, logs. `ContextOverflowError`/`PayloadTooLargeError` are caught in `_loop`, which compacts and retries (cap 2, reset on success). History compaction writes a `transcript` `role='compaction'` row whose own id is the watermark; trail compaction writes a `tool_calls` `trail_compaction` row. `thinking` becomes an internal (never-discoverable) ability dispatched at turn 0; `skill_association` becomes its own MP loop; `substitute_provider_content_field` reads the label through `mp.providers`.

**Tech Stack:** Python 3, SQLite (`transcript` + `tool_calls`), pytest (feature tests, zero mocks), nightly-test YAML scenarios.

**Design doc:** [docs/superpowers/specs/2026-06-04-compaction-redesign-design.md](../specs/2026-06-04-compaction-redesign-design.md)

---

## Reference: current → target signatures

| Symbol | Current | Target |
|---|---|---|
| `Providers.instance()` | singleton accessor | **deleted** |
| `Providers.__init__` | `()` | `(self, mp)` → `self.mp = mp` |
| `Providers.send_messages(system, messages, job, tools, …, mp)` | param-threaded | `send(self)` — scaffolds from `self.mp` |
| `Providers.calculate(...)` | returns fraction | **deleted** (folded into `pre_flight_check`) |
| `Providers.pre_flight_check` | — | new, `(self)`, raises `ContextOverflowError` |
| `Providers.selected_provider` | — | new, `(self)` → resolved provider instance |
| `mp.providers` | — | `Providers(self)`, set in `__init__` |
| `mp._compaction_retries` | — | new int, 0 in `__init__`, reset on send |
| `mp._previous_rows()` | — | new, the watermark-bounded transcript query |
| `mp._compact()` | — | new, history→transcript + trail→tool_calls |
| `compaction_persistence.get_compaction` | `tool_calls` join | `transcript WHERE role='compaction'` |
| `mp._compact_trail` / `_compact_history` / `_run_full_compaction` | exist | **deleted** |
| `mp._run_thinking_exploration` / `_persist_exploration_to_tool_calls` | exist | **deleted** (→ `ThinkingAbility`) |

---

## Execution ordering & singleton-retirement transition (OVERRIDES phase numbering)

The `Providers` singleton is atomic with `__init__`'s signature: `instance()` does `Providers()` (no args), so requiring `mp` breaks every `instance()` caller in one commit. There are exactly four runtime callers — `skill_association_service.py:121` (old `send`), `message_processor.py:508` (old `send_messages`, thinking), `message_processor.py:674` (`_loop`), `configs/channels/_common.py:22` (`instance()._resolve`). Plus one test reference: `tests/test_pattern_match_processor.py:8` (docstring). To keep **every intermediate commit runnable**, execute in this order and move the deletions to the points marked:

1. **Task 1.1 (transitional):** `__init__(self, mp=None)` — `mp` OPTIONAL so the singleton still constructs. Add `mp.providers = Providers(self)` + `_compaction_retries`. Add NEW param-free `send(self)`, `pre_flight_check`, `selected_provider`, DB-read `get_context_limit`, `ContextOverflowError`. The NEW `send(self)` collides with the OLD positional `send(user_prompt, …)` → rename the OLD one to `send_legacy` and update its single caller (`skill_association_service.py:121` → `instance().send_legacy(...)`). KEEP `send_messages`, `instance()`, `_instance`, `_lock`, `calculate` alive. Commit is runnable (both worlds coexist).
2. **Task 4.2 (skill_association):** convert to MP loop → removes the only `send_legacy` caller → **delete `send_legacy`** in this commit.
3. **Task 4.1 (thinking):** convert → removes the `send_messages` caller at `:508`.
4. **Task 4.3 (_common substitution):** convert → removes the `instance()._resolve` caller at `:22`.
5. **Task 2.1 + 2.2 (_loop + _compact):** rewrite `_loop` to `self.providers.send()` → removes the last `send_messages`/`instance()` callers at `:674`. In this commit, after grep-confirming zero `Providers.instance()` and zero `send_messages`/`send_legacy`/`calculate` callers remain, **delete `instance()`, `_instance`, `_lock`, `send_messages`, `calculate`, and make `__init__(self, mp)` required.**
6. **Task 6.1:** fix `tests/test_pattern_match_processor.py:8` docstring reference.

So the literal dispatch order is: **0.1 → 0.2 → 0.3 (baseline FAIL) → 1.1 → 3.1 → 3.2 → 4.2 → 4.1 → 4.3 → 2.1 → 2.2 → 5.1 → 5.2 → 6.1 → 6.2 → Phase 7 gates.** (3.x watermark before the conversions because `_compact()`/`get_previous_messages` depend on the repointed `get_compaction`; the singleton final-delete is the very last code change, in 2.x.)

Each subagent is told which deletions belong to its task per the list above; no subagent deletes a symbol whose callers a later task still owns.

---

## Phase 0 — Tests first (baseline MUST fail)

### Task 0.1: Repoint nightly scenario 120 to the new watermark model

**Files:**
- Modify: `/Volumes/llm/chalie-nightly-test/scenarios/120-compaction-continuity-long-history.yaml`

The scenario asserts a `tool_calls tool_name='compaction'` row and lowers `compact_at` to trigger. Under the redesign the summary is a `transcript role='compaction'` row, and `compact_at` is unread (the trigger is `req >= 0.90*window`). Repoint the trigger to shrink the active provider's reported window and repoint both assertions to `transcript`.

- [ ] **Step 1: Rewrite the trigger + assertion steps**

> **Window source (§3.3 decision):** after Phase 1, `get_context_limit()` returns the declared `providers.max_tokens` capped at 200k, so `UPDATE providers SET max_tokens=<N>` deterministically sets the window. **The window cannot go below the fixed single-turn overhead** (system prompt + tool schemas + minimal input) — if one turn's `req` exceeds `<N>`, `_compact()` can't shrink the irreducible overhead → 2-retry cap → `ContextOverflowError` and the turn hard-fails. So set `<N>` ABOVE the measured overhead and accumulate history across several turns to cross `0.90*<N>`. The two seeded facts ("boss", "traffic") must be in the FIRST message so they land in the compacted history.

Replace the `db_exec` trigger step, the chat steps, the `poll`, and the `db_query` steps. The trigger lowers the window (placeholder `<N>=8000`; **calibrated at baseline**, Task 0.3):

```yaml
- action: >
    Lower the active provider's declared window so accumulated history crosses
    0.90*window and pre_flight_check fires compaction. <N> is set above the
    single-turn overhead so each individual turn still fits. Calibrated from
    llm_call_log prompt-token counts at the pre-development baseline.
  expect: UPDATE returns ok.
  grading: exclude
  tool:
    name: db_exec
    args:
      sql: "UPDATE providers SET max_tokens = 8000 WHERE is_active = 1"
```

Then a first message carrying BOTH facts, sized to fit one turn (~1500 tokens of deterministic detail), followed by several short follow-ups that accumulate history past the window. The first:

```yaml
- action: >
    Morning recap carrying the two seeded facts the compaction summary must
    preserve: "meeting with boss" and "stuck in traffic for 3 hours". Padded
    with deterministic detail so a few turns of accumulated history crosses
    0.90*window. Must fit a single turn (under the window).
  expect: HTTP 200.
  grading: exclude
  tool:
    name: chat
    args:
      text: "Long deterministic morning recap … meeting with boss … stuck in traffic for 3 hours … <padded to ~1500 tokens, exact text finalized at baseline>"
  check: {json_path: "events[type=done]", not_empty: true}
```

Add **N follow-up chat steps** (count finalized at baseline — start with 4) of short messages, each `check: {json_path: "events[type=done]", not_empty: true}`, so accumulated history grows past `0.90*window` and pre_flight fires before one of them.

Replace the SUCCESS-row poll:

```yaml
- action: Verify a compaction summary row exists on the user transcript.
  expect: At least one role='compaction' transcript row appears within 60s.
  tool:
    name: poll
    args:
      sql: >
        SELECT COUNT(*) AS count FROM transcript
        WHERE channel = 'user' AND role = 'compaction'
      condition: {json_path: "0.count", gte: 1}
      interval_s: 5
      timeout_s: 60
```

Replace the seed-content assertion:

```yaml
- action: >
    Latest compaction summary contains both seeded facts. Case-insensitive.
  expect: Summary mentions "boss" AND "traffic".
  tool:
    name: db_query
    args:
      sql: >
        SELECT LOWER(content) AS result FROM transcript
        WHERE channel = 'user' AND role = 'compaction'
        ORDER BY id DESC LIMIT 1
  check:
    - {json_path: "0.result", contains: "boss"}
    - {json_path: "0.result", contains: "traffic"}
```

Replace the final reset step (restore the real window):

```yaml
- action: Reset max_tokens to the boot-computed value for later runs.
  expect: UPDATE returns ok.
  grading: exclude
  tool:
    name: db_exec
    args:
      sql: "UPDATE providers SET max_tokens = NULL WHERE is_active = 1"
```

Update `description:` and `tags:` to drop the `compact_at` / `_handle_overflow` / `ump` references and name the new mechanism (pre_flight_check, transcript role='compaction').

- [ ] **Step 2: Commit the scenario change**

```bash
git add /Volumes/llm/chalie-nightly-test/scenarios/120-compaction-continuity-long-history.yaml
git commit -m "test(nightly): repoint scenario 120 to transcript role='compaction' watermark"
```

### Task 0.2: Python feature tests for the deterministic surfaces

**Files:**
- Create: `backend/tests/test_compaction_watermark.py`
- Create: `backend/tests/test_thinking_internal_tool.py`

These drive the **real** services against the real shared DB (zero mocks). They cover the deterministic, non-LLM behaviours; the LLM-dependent end-to-end is scenario 120. Follow `writing-feature-tests`.

- [ ] **Step 1: Watermark + previous-messages feature test**

```python
# backend/tests/test_compaction_watermark.py
import pytest
from services import compaction_persistence, transcript_service
from services.database_service import get_shared_db_service

pytestmark = pytest.mark.integration


def _clear(channel):
    db = get_shared_db_service()
    with db.connection() as conn:
        conn.execute("DELETE FROM transcript WHERE channel = ?", (channel,))


def test_get_compaction_reads_transcript_role_compaction():
    ch = "test_wm"
    _clear(ch)
    transcript_service.write_input_row(ch, "user", "hello one")
    transcript_service.write_input_row(ch, "assistant", "reply one")
    cid = transcript_service.write_input_row(ch, "compaction", "SUMMARY: one happened")

    row = compaction_persistence.get_compaction(ch)
    assert row is not None
    assert row["compacted_text"] == "SUMMARY: one happened"
    assert row["compacted_up_to_id"] == cid  # the row's OWN id is the watermark
    _clear(ch)


def test_previous_rows_excludes_through_watermark_and_has_no_limit():
    ch = "test_wm2"
    _clear(ch)
    for i in range(25):  # >20 — proves the old limit=20 bug is gone
        transcript_service.write_input_row(ch, "user", f"msg {i}")
    cid = transcript_service.write_input_row(ch, "compaction", "checkpoint")
    after = transcript_service.write_input_row(ch, "user", "after compaction")

    rows = transcript_service.get_recent(ch, since_id=cid)
    ids = [r["id"] for r in rows]
    assert after in ids
    assert cid not in ids               # watermark row itself excluded
    assert all(i > cid for i in ids)    # nothing at/below the watermark
    _clear(ch)
```

- [ ] **Step 2: Run them — expect FAIL (get_compaction still joins tool_calls)**

Run: `cd backend && pytest tests/test_compaction_watermark.py -v`
Expected: `test_get_compaction_reads_transcript_role_compaction` FAILS (current `get_compaction` reads `tool_calls`, returns None).

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_compaction_watermark.py backend/tests/test_thinking_internal_tool.py
git commit -m "test(compaction): failing feature tests for transcript watermark + previous-rows"
```

### Task 0.3: Pre-development baseline run

- [ ] **Step 1: Push branch**

```bash
git push origin rc-0.9.0
```

- [ ] **Step 2: Run targeted nightly baseline**

```
POST http://localhost:9888/api/pipeline/start
Body: {"scenarios": ["120-compaction-continuity-long-history.yaml"], "branch": "rc-0.9.0"}
```
Poll `/api/pipeline/status` until idle (timeout 600000ms). **Verify scenario 120 FAILS** (no `transcript role='compaction'` row is produced by the current code). Record the run id + verdict in the plan's progress notes. Only proceed once it fails as expected.

---

## Phase 1 — `mp` owns a bound `Providers`

### Task 1.1: `Providers` becomes mp-bound; add `pre_flight_check` + `selected_provider`; delete singleton + `calculate`

**Files:**
- Modify: `backend/services/providers.py`

- [ ] **Step 1: Add `ContextOverflowError` + make `__init__` mp-aware (transitional — KEEP the singleton)**

Add the exception above the class and give `__init__` an OPTIONAL `mp` (so the singleton's `Providers()` still constructs — see the Execution-ordering override; the singleton is deleted later in Task 2.x):

```python
class ContextOverflowError(Exception):
    """Raised by pre_flight_check when the scaffolded request would not fit the
    provider's context window (clamped 90% / 8k-headroom rule)."""


class Providers:
    """Per-mp provider gateway. Owns the bound MessageProcessor and scaffolds
    every send / size-check / resolution from it. One instance per mp."""

    def __init__(self, mp=None):
        self.mp = mp
```

**DO NOT delete `_instance`, `_lock`, or `instance()` in this task** — four callers still use them. They are deleted in Task 2.x after every caller is converted (per the override).

- [ ] **Step 2: Add the new param-free `send()` + `pre_flight_check()` (rename OLD `send`→`send_legacy`)**

The new `send(self)` collides with the OLD positional `send(self, user_prompt, system_prompt, job, tools, …)`. Rename the OLD method to `send_legacy` (keep its body identical) and update its single caller `skill_association_service.py:121` to `instance().send_legacy(...)`. KEEP `send_messages` as-is (thinking + `_loop` still call it). Then add:

```python
    def send(self):
        """Scaffold the request from self.mp, pre-flight it, send, log. Returns LLMResponse."""
        mp = self.mp
        system = mp.config.get_system_prompt(mp)
        from services.message_processor import _wrap_with_checkpoint  # noqa: PLC0415
        user = _wrap_with_checkpoint(mp.config.channel, mp.config.get_user_prompt(mp))
        from abilities._registry import AbilityRegistry  # noqa: PLC0415
        tools = AbilityRegistry.build_tools(mp)
        job = mp.config.job
        thinking_mode = getattr(mp.config, "thinking_mode", None) or mp.thinking_level

        self.pre_flight_check(system, user, tools, job)

        provider = self._resolve(job, mp)
        messages = [{"role": "user", "content": user}]
        import time  # noqa: PLC0415
        t0 = time.monotonic()
        response = provider.send_messages(system, messages, True, tools=tools, thinking_mode=thinking_mode)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(system, messages, tools, job, response, wall_ms, mp)
        return response

    def pre_flight_check(self, system, user, tools, job):
        """Raise ContextOverflowError when the scaffolded request would overflow.

        Clamped rule (design §3.3): fire on req >= 0.90*window OR
        (window - req) <= min(8000, 0.10*window). The clamp keeps the literal 8k
        headroom on large windows and degrades to 10% on small ones, so small
        (e.g. 8k Ollama) windows never trip on every request."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        provider = self._resolve(job, self.mp)
        window = self.get_context_limit()   # declared max_tokens, capped 200k (Step 3b)
        if not window:
            return
        body = provider.build_request_body(system, [{"role": "user", "content": user}], tools)
        req = estimate_tokens(body)
        headroom = min(8000, int(0.10 * window))
        if req >= 0.90 * window or (window - req) <= headroom:
            raise ContextOverflowError(
                f"request {req} tok would overflow window {window} "
                f"(headroom {headroom}); channel={self.mp.config.channel}"
            )

    def selected_provider(self):
        """The resolved provider instance for this mp's job (design §3.1, §6.3)."""
        return self._resolve(self.mp.config.job, self.mp)
```

**Do NOT delete `calculate()` in this task** — `_loop` (line 674) still calls it. It is deleted in Task 2.1 when `_loop` is rewritten.

- [ ] **Step 3: Make the helpers read `self.mp` where they took it as a param**

`_log_after_call`, `_resolve`, `_get_tools`, `count_tokens`, `_record_send_counters` keep working but the public callers no longer pass `mp`/`job` positionally. Leave `_resolve(self, job, mp=None)` and `_log_after_call(...)` signatures intact (still called internally with explicit args by `send()`); only `instance()`, `calculate()`, and the param-threaded `send`/`send_messages` are removed.

- [ ] **Step 3b: Repoint `get_context_limit()` to read the declared `max_tokens` (§3.3 decision)**

Today `Providers.get_context_limit(job='unified')` returns `min(self._resolve(job).get_context_limit(), MAX_CONTEXT_WINDOW)` — it re-calls the live provider method (Anthropic hardcodes `return 200_000`) and ignores the backfilled `providers.max_tokens`. Make it read the declared column so the window is DB-driven, capped 200k, with a first-boot fallback:

```python
    def get_context_limit(self):
        """Declared context window for the active provider/model, hard-capped at
        MAX_CONTEXT_WINDOW (200k). Reads the backfilled providers.max_tokens
        (set by provider_token_limits.backfill_one = min(model window, 200k)).
        Falls back to the live provider method before backfill has run. §3.3."""
        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
        config = ProviderCacheService.get_selected_provider() or {}
        declared = config.get("max_tokens")
        if declared and int(declared) > 0:
            return min(int(declared), MAX_CONTEXT_WINDOW)
        return min(self._resolve(self.mp.config.job, self.mp).get_context_limit(), MAX_CONTEXT_WINDOW)
```

Then expose `max_tokens` on the selected-provider config dict so the read above sees it. In `backend/services/provider_cache_service.py` `get_selected_provider()` (line 128), add `max_tokens` to the returned dict:

```python
            if selected:
                return {
                    'platform': selected['platform'],
                    'model': selected['model'],
                    'host': selected.get('host'),
                    'api_key': selected.get('api_key'),
                    'dimensions': selected.get('dimensions'),
                    'timeout': selected.get('timeout'),
                    'max_tokens': selected.get('max_tokens'),
                }
```

- [ ] **Step 3c: Feature test — declared window is read + capped**

Append to `backend/tests/test_compaction_watermark.py`:

```python
def test_get_context_limit_reads_declared_max_tokens_capped():
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    from services.providers import MAX_CONTEXT_WINDOW
    from services.provider_db_service import ProviderDbService
    from services.provider_cache_service import ProviderCacheService
    db = get_shared_db_service()
    svc = ProviderDbService(db)
    sel = svc.get_selected_provider()
    if not sel:
        pytest.skip("no active provider in this env")
    pid = sel["id"]
    original = sel.get("max_tokens")
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    try:
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = 8000 WHERE id = ?", (pid,))
        ProviderCacheService.invalidate()
        assert mp.providers.get_context_limit() == 8000          # declared value honoured
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = 999999 WHERE id = ?", (pid,))
        ProviderCacheService.invalidate()
        assert mp.providers.get_context_limit() == MAX_CONTEXT_WINDOW   # capped at 200k
    finally:
        with db.connection() as conn:
            conn.execute("UPDATE providers SET max_tokens = ? WHERE id = ?", (original, pid))
        ProviderCacheService.invalidate()
```

`MAX_CONTEXT_WINDOW = 200_000` is defined in `services/providers.py:25` (provider_token_limits imports it from there).

Run: `cd backend && pytest tests/test_compaction_watermark.py -k context_limit -v` — expected PASS after Step 3b.

- [ ] **Step 4: Construct the bound instance in `MessageProcessor.__init__`**

In `backend/services/message_processor.py` `__init__`, after `self._metrics = MetricsAccumulator()` (line 229), add:

```python
        # The mp owns its provider gateway — param-free, scaffolds from self.
        from services.providers import Providers  # noqa: PLC0415
        self.providers = Providers(self)
        # Reactive-compaction retry budget; reset to 0 after any successful send.
        self._compaction_retries: int = 0
```

- [ ] **Step 5: Run the watermark test to confirm nothing imports-broke**

Run: `cd backend && python -c "import services.message_processor, services.providers"`
Expected: no ImportError.

- [ ] **Step 6: Commit**

```bash
git add backend/services/providers.py backend/services/message_processor.py backend/services/provider_cache_service.py backend/services/skill_association_service.py backend/tests/test_compaction_watermark.py
git commit -m "refactor(providers): mp-owned Providers(mp); param-free send/pre_flight_check/get_context_limit (singleton kept transitionally)"
```

---

## Phase 2 — Loop rewrite + `_compact()`

### Task 2.1: Collapse `_loop` to send → catch overflow → compact → retry

**Files:**
- Modify: `backend/services/message_processor.py` (`_loop` 669–708, `_COMPACTION_CHANNELS` 665–667)

- [ ] **Step 1: Replace `_loop` body**

```python
    def _loop(self) -> str:  # noqa: C901
        """ACT game loop — send, dispatch tools, compact-and-retry on overflow."""
        from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
        from services.providers import ContextOverflowError  # noqa: PLC0415
        from services.llm_service import PayloadTooLargeError  # noqa: PLC0415
        while True:
            if self._should_stop():
                return ""
            if len(self._previous_rows()) > 50:   # proactive turn-count compaction
                self._compact()
            try:
                response = self.providers.send()
            except (ContextOverflowError, PayloadTooLargeError):
                if self._compaction_retries >= 2:
                    raise
                self._compaction_retries += 1
                self._compact()
                continue                          # retry re-reads the now-compacted DB
            self._compaction_retries = 0          # reset on every successful send
            if not response.tool_calls:
                return response.text or ""
            dispatcher = ToolDispatcher(self)
            for tc in response.tool_calls:
                if self.cancel_event.is_set():
                    return ""
                dispatcher.dispatch(tc["name"], tc["input"])
            self._record_narration(response)
            self.current_iteration += 1
```

Delete the `_COMPACTION_CHANNELS` class attribute (lines 665–667) and its docstring — the recursion guard is replaced by the universal retry cap (design §3.3).

- [ ] **Step 2: Confirm `PayloadTooLargeError` import path**

Run: `cd backend && python -c "from services.llm_service import PayloadTooLargeError; print('ok')"`
Expected: `ok`. If it lives elsewhere, grep and use the real module: `grep -rn "class PayloadTooLargeError" backend/services`.

### Task 2.2: Add `_previous_rows()` and `_compact()`

**Files:**
- Modify: `backend/services/message_processor.py`

- [ ] **Step 1: Split the query into `_previous_rows()` and rewrite `get_previous_messages()` to render it**

Replace `get_previous_messages` (785–860). Add `_previous_rows` and keep the render in `get_previous_messages`:

```python
    def _previous_rows(self) -> list:
        """Watermark-bounded transcript rows for this channel (design §3.7).

        SELECT * FROM transcript WHERE channel=? AND id > <watermark> ORDER BY id ASC
        No LIMIT (fixes the 20-row bug). Empty for suppress_history channels and
        post-compaction turns (the compaction row's own id IS the watermark)."""
        if self.config.suppress_history:
            return []
        from services import compaction_persistence, transcript_service  # noqa: PLC0415
        compaction = compaction_persistence.get_compaction(self.config.channel)
        watermark = compaction["compacted_up_to_id"] if compaction else 0
        return transcript_service.get_recent(self.config.channel, since_id=watermark)
```

`get_previous_messages` now calls `_previous_rows()` for the rows and keeps the existing durable-tool-call interleave + render loop (lines 826–860 unchanged), but drops the summary prepend (lines 836–837) — the checkpoint envelope (`_wrap_with_checkpoint`) is the single source for the summary (design §3.7):

```python
    def get_previous_messages(self) -> str:
        """Render the ## Previous Messages block from _previous_rows()."""
        if self.config.suppress_history:
            return ""
        from services.tool_call_service import ToolCallService  # noqa: PLC0415
        entries = self._previous_rows()
        if not entries:
            return ""
        all_ids = [e["id"] for e in entries if e.get("id")]
        durable_by_id = ToolCallService().get_by_transcript_ids(all_ids, include_ephemeral=False) if all_ids else {}
        lines: list[str] = []
        for entry in entries:
            ts = _format_ts(entry.get("created_at"), row_kind="transcript", row_id=entry.get("id"))
            raw_role = entry.get("role") or "unknown"
            role_label = "Assistant" if raw_role == "assistant" else raw_role
            content = (entry.get("content") or "").replace("\n", " ").strip()
            lines.append(f"[{ts}] {role_label}: {content}")
            for tc in durable_by_id.get(entry.get("id"), []):
                tc_name = tc.get("tool_name") or tc.get("name") or "tool"
                if tc_name in _NEVER_RENDER_IN_PREVIOUS:
                    continue
                tc_params = _parse_tc_params(tc.get("params"))
                tc_result = tc.get("result") or ""
                lines.append(_render_tool_call_for_previous(tc_name, tc_params, tc_result))
        return "\n".join(lines)
```

- [ ] **Step 2: Add `_compact()` (replaces `_compact_trail` + `_compact_history`)**

```python
    def _compact(self) -> None:
        """Compact history (→transcript role='compaction') and, when a trail
        exists, the act-trail (→tool_calls trail_compaction). Design §3.5.

        The history summary's transcript row id IS the new watermark, so the
        next read of _previous_rows() returns nothing through it. Each sub-summary
        is produced by a fresh MessageProcessor.process() compaction loop. A
        no-op (nothing to summarise) leaves the watermark untouched — the retry
        cap in _loop bounds the universal case."""
        from configs.channels import CompactionConfig, TrailHandoverConfig  # noqa: PLC0415
        from services import transcript_service  # noqa: PLC0415

        # 1. History → transcript (the writer's row id is the watermark).
        prev = self.get_previous_messages()
        if prev.strip():
            summary = MessageProcessor.process(prev, CompactionConfig())
            if summary and summary.strip():
                transcript_service.write_input_row(self.config.channel, "compaction", summary)

        # 2. Trail → tool_calls (only when a non-compaction trail row exists).
        if self._has_trail():
            from services.act_trail import ActTrail  # noqa: PLC0415
            from abilities._event_emitter import ActEventEmitter  # noqa: PLC0415
            trail_text = self._render_act_trail()
            handover = MessageProcessor.process(trail_text, TrailHandoverConfig())
            if handover and handover.strip():
                emitter = ActEventEmitter(self.config)
                emitter.emit({"type": "act_tool_start", "name": "trail_compaction", "id": "compact", "summary": "compacting context"})
                ActTrail().record(
                    tool_name="trail_compaction", params={}, result=handover,
                    transcript_id=self.uid, ephemeral=True,
                )
                emitter.emit({"type": "act_tool_end", "name": "trail_compaction", "id": "compact", "ok": True})

        # 3. Reset the iteration counter for the retried turn.
        self.current_iteration = 0
```

Delete `_compact_trail` (920–965), `_compact_history` (967–990), and `_run_full_compaction` (277–367). Also delete the module helpers only those used: `_format_compaction_entry` (1031), `_build_compaction_input` (1049), `_write_compaction_audit_row` (1063) — grep first to confirm no other callers:

Run: `cd backend && grep -rn "_run_full_compaction\|_format_compaction_entry\|_build_compaction_input\|_write_compaction_audit_row\|_compact_trail\|_compact_history" --include="*.py" . | grep -v __pycache__`
Expected after deletion: only test references (handled in their own tasks). `_extract_compaction_summary` MAY still be used by CompactionConfig output parsing — confirm before deleting; keep if referenced.

- [ ] **Step 3: Remove the dead `_overflow_recovered_this_turn` attribute**

Delete `self._overflow_recovered_this_turn` (line 227) and its comment block (219–227).

- [ ] **Step 4: Run import + watermark tests**

Run: `cd backend && pytest tests/test_compaction_watermark.py::test_previous_rows_excludes_through_watermark_and_has_no_limit -v`
Expected: PASS (the SQL is unchanged for this read; `since_id` was already correct).

- [ ] **Step 5: Commit**

```bash
git add backend/services/message_processor.py
git commit -m "refactor(compaction): single _compact() + pre-flight loop; delete two-threshold branching"
```

---

## Phase 3 — Watermark home = `transcript role='compaction'`

### Task 3.1: Repoint `compaction_persistence.get_compaction`

**Files:**
- Modify: `backend/services/compaction_persistence.py`

- [ ] **Step 1: Rewrite `get_compaction` to read the transcript**

```python
def get_compaction(channel: str) -> Optional[Dict]:
    """Latest history-compaction summary for a channel, or None.

    Canonical watermark reader (design §3.6): the newest transcript row with
    role='compaction'. Its OWN id is the watermark (compacted_up_to_id), so
    _previous_rows()'s `id > watermark` naturally excludes it and everything
    before it. Never raises — DB errors are logged and treated as 'no compaction'."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, content, created_at FROM transcript
                WHERE channel = ? AND role = 'compaction'
                ORDER BY id DESC LIMIT 1
                """,
                (channel,),
            ).fetchone()
        if not row:
            return None
        return {
            "compacted_text": row[0 + 1],         # content
            "compacted_up_to_id": row[0],         # the row's own id
            "tool_call_id": None,                 # legacy field; no longer a tool_call
            "created_at": row[2],
        }
    except Exception as exc:
        logger.warning("%s Failed to get compaction for %s: %s", LOG_PREFIX, channel, exc)
        return None
```

Update the module docstring (lines 1–12) to describe the `transcript role='compaction'` source and drop the `tool_calls` narrative.

- [ ] **Step 2: Run the get_compaction feature test**

Run: `cd backend && pytest tests/test_compaction_watermark.py::test_get_compaction_reads_transcript_role_compaction -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/services/compaction_persistence.py
git commit -m "refactor(compaction): watermark = newest transcript role='compaction' row"
```

### Task 3.2: Add `TrailHandoverConfig`

**Files:**
- Create: `backend/configs/channels/trail_handover.py`
- Modify: `backend/configs/channels/__init__.py` (export)

- [ ] **Step 1: Create the config (mirrors `CompactionConfig`)**

```python
# backend/configs/channels/trail_handover.py
from __future__ import annotations

from services.processor_config import ProcessorConfig

_SYSTEM_PROMPT = (
    "Extract a handover summary of what is available in the user-message. "
    "Keep your response concise."
)


class TrailHandoverConfig(ProcessorConfig):
    """Act-trail handover compaction — bounded loop, no tools, no transcript. §3.5."""

    def __init__(self) -> None:
        super().__init__(
            channel="compaction",
            role="compaction",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=30,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        return _SYSTEM_PROMPT
```

- [ ] **Step 2: Export it**

In `backend/configs/channels/__init__.py`, add `TrailHandoverConfig` to the imports + `__all__` next to `CompactionConfig`.

- [ ] **Step 3: Verify**

Run: `cd backend && python -c "from configs.channels import TrailHandoverConfig; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add backend/configs/channels/trail_handover.py backend/configs/channels/__init__.py
git commit -m "feat(configs): TrailHandoverConfig for act-trail handover compaction"
```

---

## Phase 4 — Retire the singleton (the three callers)

### Task 4.1: `thinking` → internal ability + `ThinkingConfig`

**Files:**
- Create: `backend/abilities/thinking.py`
- Modify: `backend/services/message_processor.py` (delete hack; dispatch at turn 0)
- Modify: `backend/configs/channels/user.py` (delete CoT branch)
- Modify: the PolicyManager INTERNAL frozenset (locate via grep)
- Modify: `backend/services/processor_config.py` if a `thinking_mode` class-attr default is needed

- [ ] **Step 1: Create `ThinkingAbility` + `ThinkingConfig`**

```python
# backend/abilities/thinking.py
"""ThinkingAbility — internal high-deliberation exploration pass.

NEVER discoverable and NEVER in any always_available list: the model never sees
`thinking` in find_tools or its toolbox. The orchestrator dispatches it
programmatically at turn 0 when the thinking-gate resolves 'high' (like the
memory.recall turn-0 seed). It fires its own MessageProcessor.process() loop with
ThinkingConfig, which RETAINS the parent channel's full tool surface so the model
can reason about which tools would help — but is told not to invoke them. The
result is recorded into the parent's act-trail by the dispatch path and flows back
into the next get_user_prompt via _render_act_trail. No special attribute."""

from typing import ClassVar

from abilities._ability import Ability
from services.processor_config import ProcessorConfig

_EXPLORATION_PREFIX = (
    "Think out loud about the user's request before responding.\n\n"
    "Consider:\n"
    "- What does the ideal response look like? What would make it genuinely useful?\n"
    "- Do you already know enough to answer well, or are there gaps?\n"
    "- Would any of your available tools fill those gaps? Which ones, in what order?\n"
    "- Is there anything non-obvious about this request you might miss on a first read?\n\n"
    "Whatever you output here will be shown to you as Chain of Thought on the next "
    "pass — write to your future self. Be specific: name the tools you plan to use, "
    "flag uncertainties, note key facts you want to remember to include.\n\n"
    "If the request is straightforward and you have nothing useful to say to yourself, "
    "output exactly: NOTHING\n\n"
    "DO NOT INVOKE TOOLS — they are disabled in this phase. Think only.\n\n---\n\n"
)


class ThinkingConfig(ProcessorConfig):
    """Single-pass high-deliberation exploration. Retains the parent's tool
    surface (so the catalogue is visible) but the prompt forbids invocation."""

    thinking_mode: ClassVar[str] = "high"

    def __init__(self, always_available, discoverable, policy_channel) -> None:
        super().__init__(
            channel="thinking",
            role="thinking",
            policy_channel=policy_channel,
            always_available=list(always_available or []),
            discoverable=list(discoverable or []),
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return _EXPLORATION_PREFIX + (mp._raw_input or "")

    def get_system_prompt(self, mp) -> str:
        from services.system_message_prompt import UserSystemPrompt  # noqa: PLC0415
        return UserSystemPrompt().get_prompt()  # confirm the real user-channel system prompt builder


class ThinkingAbility(Ability):
    NAME = "thinking"
    SEARCH_TOOLTIP = "internal deliberation pass"
    SUMMARY = "Internal-only high-deliberation exploration. Never user-invocable."
    EXAMPLES: ClassVar[list] = []
    INPUT_SCHEMA: ClassVar[dict] = {"type": "object", "properties": {}}

    def run(self, params: dict) -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        parent = self.MessageProcessor
        result = MessageProcessor.process(
            parent._raw_input,
            ThinkingConfig(
                parent.config.always_available,
                parent.config.discoverable,
                parent.config.policy_channel,
            ),
        )
        text = (result or "").strip()
        if text.upper() == "NOTHING":
            return {"status": "success", "result": ""}
        return {"status": "success", "result": text}
```

> **VERIFY before coding:** the real user-channel system-prompt builder class name. Grep `class .*SystemPrompt` in `backend/services/system_message_prompt.py` and use the one `UserConfig.get_system_prompt` uses. Do not invent `UserSystemPrompt` if it differs.

- [ ] **Step 2: Make `send()` honour `config.thinking_mode`**

Already wired in Task 1.1 Step 2 (`thinking_mode = getattr(mp.config, "thinking_mode", None) or mp.thinking_level`). Confirm `ProcessorConfig` (frozen dataclass) tolerates a subclass `ClassVar` `thinking_mode` — `ClassVar` is not a dataclass field, so it is allowed. No base change needed.

- [ ] **Step 3: Dispatch `thinking` at turn 0; delete the exploration hack**

In `_seed_turn_zero` (629–663), after the memory-seed block, add:

```python
        # c. High-deliberation thinking pass — programmatic, never model-visible.
        if getattr(self, "thinking_level", "low") == "high":
            dispatcher.dispatch("thinking", {})
```

In `_run_thinking_gate` (375–458): delete the exploration ThreadPool block (413–435) and every `self._thinking_exploration = …` assignment (401, 427, 433, 437, 458). The gate now only sets `self._thinking_level`. Delete `_run_thinking_exploration` (460–531) and `_persist_exploration_to_tool_calls` (533–554) entirely. Delete `self._thinking_exploration` (218) from `__init__`. In `process()` delete `mp.thinking_exploration = None` (585). In `_setup` delete the `self.thinking_exploration = self._thinking_exploration` sync (622–625).

Grep every other writer and delete them:

Run: `cd backend && grep -rn "thinking_exploration" --include="*.py" . | grep -v __pycache__`
Delete the `= None` lines in `services/subconscious_worker.py` (415, 579), `services/transcript_service.py` (395), `services/skill_suggestion_message_processor.py` (76), and `message_processor.py` (988 inside the now-deleted `_compact_history`). After this grep returns **zero** hits.

- [ ] **Step 4: Delete the `## Chain of Thought` branch in `user.py`**

In `backend/configs/channels/user.py`, delete lines 223–237 (the `exploration = getattr(mp, "thinking_exploration", None)` block); `body` is returned bare. The CoT now arrives via the act-trail render already present in the same builder.

- [ ] **Step 5: Add `thinking` to the PolicyManager INTERNAL bypass**

The dispatch at turn 0 runs through `PolicyManager.wrap` with `policy_channel=user`. Internal programmatic tools bypass the gate via the INTERNAL frozenset (TKT-797). Locate and add `"thinking"`:

Run: `cd backend && grep -rln "INTERNAL" --include="*.py" backend/services | grep -i polic`
Add `"thinking"` to that frozenset so authorize() short-circuits before any DB lookup.

- [ ] **Step 6: Remove `thinking` from `_NEVER_RENDER_IN_PREVIOUS`**

In `message_processor.py` (1164) change `frozenset({'compaction', 'thinking'})` → `frozenset({'compaction'})`. The thinking row is now ephemeral (recorded by dispatch with `ephemeral=True`) and is purged at turn end, so it never reaches the previous-messages read — the filter entry is dead.

- [ ] **Step 7: Keep `thinking` out of the find_tools index (optional safety)**

`find_tools` already filters strictly to `config.discoverable` ([find_tools.py:137](../../backend/abilities/find_tools.py)), and `thinking` is in no `discoverable` list, so it can never be surfaced. To also keep it out of the semantic ranking, exclude it in the index build:

In `backend/utils/build_ability_db.py` where `abilities = list(AbilityRegistry.all())` (line 144), filter: `abilities = [a for a in AbilityRegistry.all() if a.NAME != "thinking"]`.

- [ ] **Step 8: Rebuild the ability DB**

```bash
cd backend && python -m utils.build_ability_db
```
Commit the regenerated `backend/abilities/assets/abilities.sqlite` + `backend/resources/pre-trained/abilities_sha.json`.

- [ ] **Step 9: Feature test — thinking dispatched, invisible, recorded**

```python
# backend/tests/test_thinking_internal_tool.py
import pytest
from abilities._registry import AbilityRegistry
from configs.channels import UserConfig  # or the real user config import

pytestmark = pytest.mark.integration


def test_thinking_never_discoverable():
    # thinking is registered but in no discoverable/always_available list.
    assert "thinking" in {a.NAME for a in AbilityRegistry.all()}
    cfg = UserConfig()
    assert "thinking" not in (cfg.always_available or [])
    assert "thinking" not in (cfg.discoverable or [])


def test_thinking_config_retains_parent_tool_surface():
    from abilities.thinking import ThinkingConfig
    from configs.channels import UserConfig
    parent = UserConfig()
    tc = ThinkingConfig(parent.always_available, parent.discoverable, parent.policy_channel)
    assert tc.always_available == list(parent.always_available or [])
    assert tc.discoverable == list(parent.discoverable or [])
    assert tc.thinking_mode == "high"
    assert tc.max_iterations == 1
```

Run: `cd backend && pytest tests/test_thinking_internal_tool.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/abilities/thinking.py backend/services/message_processor.py backend/configs/channels/user.py backend/services/subconscious_worker.py backend/services/transcript_service.py backend/services/skill_suggestion_message_processor.py backend/abilities/assets/abilities.sqlite backend/resources/pre-trained/abilities_sha.json backend/utils/build_ability_db.py backend/tests/test_thinking_internal_tool.py
# + the PolicyManager file
git commit -m "refactor(thinking): internal never-discoverable ability dispatched at turn 0; delete exploration hack"
```

### Task 4.2: `skill_association` → own MP loop

**Files:**
- Create: `backend/configs/channels/skill_association.py`
- Modify: `backend/configs/channels/__init__.py` (export)
- Modify: `backend/services/skill_association_service.py`

- [ ] **Step 1: Create `SkillAssociationConfig`**

```python
# backend/configs/channels/skill_association.py
from __future__ import annotations

from services.processor_config import ProcessorConfig

_SYSTEM_PROMPT = """You map behavioral patterns to skill playbooks.

Given a list of the user's behavioral patterns and a list of available skills,
identify which patterns are relevant to which skills and produce a personalisation
rule for each match.

A personalisation rule is a single sentence describing how the skill should be
adapted based on the pattern. Only produce rules where the pattern genuinely
informs how the skill should be executed differently.

Respond with a JSON array of objects:
[{"skill_id": <int>, "pattern_name": "<str>", "rule": "<str>"}]

If no patterns match any skills, respond with an empty array: []"""


class SkillAssociationConfig(ProcessorConfig):
    """Subconscious pattern→skill association — own MP loop, no tools, no transcript."""

    def __init__(self) -> None:
        super().__init__(
            channel="skill_association",
            role="skill_association",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        return _SYSTEM_PROMPT
```

> **VERIFY:** the job used for provider resolution. The old call used `job='subconscious'`. `ProcessorConfig.job` is derived from the channel/usage_class — confirm `SkillAssociationConfig.job` resolves to the subconscious provider, or set `usage_class`/`job` explicitly to match the prior `job='subconscious'`. Grep `def job` / `usage_class` in `processor_config.py`.

- [ ] **Step 2: Rewrite `_request_associations` to use the MP loop**

In `skill_association_service.py`, replace the `Providers.instance().send(...)` block (119–138) with:

```python
        user_prompt = (
            f"## Behavioral Patterns\n{json.dumps(pattern_list)}\n\n"
            f"## Available Skills\n{json.dumps(skill_list)}"
        )

        from services.message_processor import MessageProcessor
        from configs.channels import SkillAssociationConfig
        try:
            text = MessageProcessor.process(user_prompt, SkillAssociationConfig())
        except Exception as exc:
            exc_str = str(exc).lower()
            if "context" in exc_str or "token" in exc_str or "length" in exc_str:
                logger.error(
                    f"{LOG_PREFIX} prompt exceeds provider context window — "
                    f"patterns={len(patterns)} skills={len(skills)}: {exc}"
                )
            else:
                logger.error(f"{LOG_PREFIX} LLM call failed: {exc}")
            return None

        return _parse_associations(text)
```

Remove the now-unused `_SYSTEM_PROMPT` constant from `skill_association_service.py` (it moved into the config) and the `from services.providers import Providers` import.

- [ ] **Step 3: Export the config**

Add `SkillAssociationConfig` to `backend/configs/channels/__init__.py` imports + `__all__`.

- [ ] **Step 4: Verify import + run the existing skill-association feature/integration test if present**

Run: `cd backend && python -c "from configs.channels import SkillAssociationConfig; print('ok')"`
Run: `cd backend && grep -rln "SkillAssociationService" tests/ || echo "no existing test"`

- [ ] **Step 5: Commit**

```bash
git add backend/configs/channels/skill_association.py backend/configs/channels/__init__.py backend/services/skill_association_service.py
git commit -m "refactor(skill-assoc): own MessageProcessor loop via SkillAssociationConfig"
```

### Task 4.3: `substitute_provider_content_field` → inline via `mp.providers`

**Files:**
- Modify: `backend/configs/channels/_common.py`
- Modify: `backend/configs/channels/user.py` (call site 135)
- Modify: `backend/configs/channels/external_agent.py` (call site 106)

- [ ] **Step 1: Rewrite the helper to take `mp`**

```python
def substitute_provider_content_field(body: str, mp) -> str:
    """Replace {{provider_content_field_name}} with the active provider's
    CONTENT_FIELD_LABEL, read through the mp-owned providers gateway. Best-effort:
    placeholder absent or resolution fails → body unchanged (design §6.3)."""
    if _CONTENT_FIELD_PLACEHOLDER not in body:
        return body
    try:
        label = mp.providers.selected_provider().CONTENT_FIELD_LABEL
    except Exception:
        label = None
    if not label:
        return body
    return body.replace(_CONTENT_FIELD_PLACEHOLDER, label)
```

Remove the dead `from services.providers import Providers` / `Providers.instance()._resolve(job)` lines and the `job` parameter.

- [ ] **Step 2: Update both call sites**

`user.py:135`: `prompt = substitute_provider_content_field(prompt, mp)`
`external_agent.py:106`: `body = substitute_provider_content_field(body, mp)`
(Both are inside `get_system_prompt(self, mp)` — `mp` is in scope.)

- [ ] **Step 3: Verify the singleton is fully dead**

Run: `cd backend && grep -rn "Providers.instance()" --include="*.py" . | grep -v __pycache__`
Expected: **zero hits.** If any remain, they are unconverted callers — STOP and surface them.

- [ ] **Step 4: Feature test — substitution resolves the real provider label**

```python
# append to backend/tests/test_compaction_watermark.py or a new test file
def test_substitute_provider_content_field_uses_mp_providers():
    from configs.channels._common import substitute_provider_content_field, _CONTENT_FIELD_PLACEHOLDER
    from services.message_processor import MessageProcessor
    from configs.channels import UserConfig
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "hi", None)
    mp.config = UserConfig()
    out = substitute_provider_content_field(f"write into {_CONTENT_FIELD_PLACEHOLDER}", mp)
    assert _CONTENT_FIELD_PLACEHOLDER not in out  # replaced with the active provider's label
```

Run: `cd backend && pytest tests/test_compaction_watermark.py -k substitute -v`
Expected: PASS (requires a configured provider; this is the real stack).

- [ ] **Step 5: Commit**

```bash
git add backend/configs/channels/_common.py backend/configs/channels/user.py backend/configs/channels/external_agent.py backend/tests/test_compaction_watermark.py
git commit -m "refactor(prompt): substitute_provider_content_field reads label via mp.providers; singleton retired"
```

---

## Phase 5 — Consumers + observability

### Task 5.1: Exclude `role='compaction'` from the user conversation view

**Files:**
- Modify: `backend/api/conversation.py` (46)

- [ ] **Step 1: Add `compaction` to the role filter**

```python
            "WHERE channel = 'user' AND role NOT IN ('subagent_return', 'compaction') "
```

- [ ] **Step 2: Verify no other transcript reader surfaces compaction rows unexpectedly**

Run: `cd backend && grep -rn "FROM transcript" --include="*.py" backend/api backend/services | grep -i "role" | grep -v __pycache__`
Confirm any user-facing reader excludes `compaction`; episode/extraction readers are out of scope but note any that would ingest a compaction row (see Risk R1).

- [ ] **Step 3: Commit**

```bash
git add backend/api/conversation.py
git commit -m "fix(conversation): hide role='compaction' summary rows from the chat view"
```

### Task 5.2: Repoint the Brain observability compaction source

**Files:**
- Modify: the Brain API/endpoint that reads compaction (locate via grep)

- [ ] **Step 1: Find the reader**

Run: `cd backend && grep -rn "tool_name.*compaction\|'compaction'\|\"compaction\"" --include="*.py" backend/api | grep -v __pycache__`
Repoint any Brain observability query from `tool_calls tool_name='compaction'` to `transcript role='compaction'` (mirror `get_compaction`). If the Brain tab already calls `compaction_persistence.get_compaction`, it is automatically correct — verify and note.

- [ ] **Step 2: Commit (if changed)**

```bash
git add <brain reader file>
git commit -m "fix(brain): read compaction summary from transcript role='compaction'"
```

---

## Phase 6 — Cleanup, tests, docs

### Task 6.1: Delete obsolete compaction tests + reconcile references

- [ ] **Step 1: Find tests referencing deleted symbols**

Run: `cd backend && grep -rln "_run_full_compaction\|_compact_trail\|_compact_history\|_thinking_exploration\|Providers.instance\|\.calculate(" tests/ | grep -v __pycache__`
For each: if it asserts deleted behaviour, delete the test (the behaviour is gone — scenario 120 + the new feature tests cover the replacement). If it asserts surviving behaviour through a deleted symbol, rewrite it onto the new entry point. Do NOT relax assertions to make code pass (Feature Test Acceptance Criteria #2).

- [ ] **Step 2: Run the unit gate**

Run: `cd backend && pytest -m unit -q`
Expected: green. Fix real regressions in code, never by weakening a test.

- [ ] **Step 3: Commit**

```bash
git add backend/tests
git commit -m "test: remove obsolete compaction/exploration tests superseded by redesign"
```

### Task 6.2: /taskie Tools docs + design-doc reconciliation

- [ ] **Step 1: Update Tools docs** — `thinking` is new but internal/never-discoverable: add a Tools doc noting it is programmatic-only (not user-invocable, not in find_tools). Update the Tools index. No policy-label UI entry is needed (it is never user-facing), but confirm the Brain Policy tab does not error on an unknown action.
- [ ] **Step 2: Update `docs/04-ARCHITECTURE.md` / `docs/13-MESSAGE-FLOW.md`** compaction + thinking sections to the new model (pre-flight loop, transcript watermark, internal thinking tool).
- [ ] **Step 3: Commit**

```bash
git add docs/
git commit -m "docs: reconcile compaction redesign (pre-flight loop, transcript watermark, internal thinking)"
```

---

## Phase 7 — Gates (chalie-feature)

### Task 7.1: Pre-critic nightly run
- [ ] Push; run scenario 120 targeted; collect evidence; hand to critic. (chalie-feature Step 2b)

### Task 7.2: Pass gate
- [ ] Scenario 120 must PASS. If >2 fix rounds, STOP and escalate. (chalie-feature Step 3)

### Task 7.3: Post-push SonarQube gate
- [ ] Resolve any new code smells (ruff, vulture, Sonar). Hard cap 3 attempts. (chalie-feature Step 7c)

### Task 7.4: Notify Chalie
- [ ] `mcp__chalie__talk_to_chalie` with the rc-0.9.0 shipped-work summary. (chalie-feature Step 7d)

---

## Risks / verify-continuously (chalie-feature: surface every breakage)

- **R1 — `write_input_row` side effects on a `role='compaction'` row.** `write_input_row` calls `_maybe_trigger_extraction(channel, row_id)` and `_resolve_location`. A compaction row will now fire the rolling episode-extraction trigger. VERIFY `_maybe_trigger_extraction` ignores `role='compaction'` (or is harmless on it); if it ingests the summary as conversational content, add a role guard. Evidence required before closing.
- **R2 — `CompactionConfig` output parsing.** `_compact()` writes `MessageProcessor.process(prev, CompactionConfig())` verbatim as the summary. The old `_run_full_compaction` extracted `<summary>` tags via `_extract_compaction_summary`. CONFIRM whether `CompactionConfig`'s system prompt still wraps output in `<summary>`; if so, either strip tags in `_compact()` or update the prompt. Do not write raw `<summary>…</summary>` into the transcript.
- **R3 — proactive `len(_previous_rows()) > 50` cost.** Runs every iteration. `_previous_rows()` is one indexed `id > watermark` query; acceptable, but confirm it is not called twice per iteration (it is also inside `get_previous_messages`). Acceptable duplication; note if profiling shows otherwise.
- **R4 — `build_request_body` / `estimate_tokens` availability on every provider.** `pre_flight_check` calls `provider.build_request_body` + `estimate_tokens`. The old `calculate()` used the same pair, so parity holds — but re-confirm both exist on the Fallback wrapper and each concrete provider.
- **R5 — thinking dispatch ordering.** `_run_thinking_gate` (sets `thinking_level`) runs in `_setup` BEFORE `_seed_turn_zero` (verified: 620–627). The turn-0 dispatch reads `self.thinking_level`; ensure the gate has run for the user channel before seeding.
- **R6 — `thinking` policy bypass.** If `thinking` is NOT added to the INTERNAL frozenset, the turn-0 dispatch is gated under the user policy channel and may be denied / prompt the user. Must be in the bypass set (Task 4.1 Step 5).

---

## Self-review notes

- **Spec coverage:** §3.1 (Task 1.1), §3.2 (2.1), §3.3 (1.1 pre_flight_check), §3.4 413 (2.1 catch), §3.5 (_compact 2.2), §3.6 watermark (3.1), §3.7 (2.2 _previous_rows), §4 consumers (5.1/5.2), §5 deletions (2.2/4.1), §6.1 thinking (4.1), §6.2 skill_assoc (4.2), §6.3 substitution (4.3), §7 configs (3.2/4.1/4.2). All covered.
- **Type consistency:** `Providers(mp)` / `mp.providers` / `selected_provider()` / `_previous_rows()` / `_compact()` / `_compaction_retries` / `ContextOverflowError` / `ThinkingConfig(always_available, discoverable, policy_channel)` used identically across tasks.
- **Open VERIFY items folded into steps:** real user system-prompt class (4.1 S1), `PayloadTooLargeError` import (2.1 S2), `_extract_compaction_summary` survival (2.2 S2 / R2), `job` resolution for SkillAssociationConfig (4.2 S1), INTERNAL frozenset location (4.1 S5).
