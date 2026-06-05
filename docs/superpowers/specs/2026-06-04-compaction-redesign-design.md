# Compaction Redesign — Design

- **Date:** 2026-06-04
- **Branch:** rc-0.9.0
- **Status:** Approved design (implementation not started)
- **Author:** Dylan + Claude Code

---

## 1. Problem

Compaction on `rc-0.9.0` has several confirmed bugs and a tangled control flow. Evidence from the current code:

1. **413 is never handled.** `OllamaService` raises `PayloadTooLargeError` (a `NonRetryableError`) on HTTP 413; `FallbackLLMService.send_messages` re-raises it; **nothing in `MessageProcessor` catches it** — it propagates unhandled out of `_loop()`. The `_overflow_recovered_this_turn` attribute set in `__init__` is dead (no reader/writer).

2. **`get_previous_messages()` truncates at 20 rows.** It calls `transcript_service.get_recent(channel, since_id=watermark)` with **no `limit`**, so the `limit=20` default applies (`transcript_service.py` `get_recent`). Any channel with >20 turns since the last compaction silently loses older history.

3. **Two-threshold branching is complex.** `_loop()` calls `Providers.calculate()` for a fraction, then branches `>0.90 trail` / `>0.80 history` across `_compact_trail()` / `_compact_history()` / `_run_full_compaction()` with a `_COMPACTION_CHANNELS` recursion guard.

4. **The general compaction summary lives in `tool_calls`** (`tool_name='compaction'`, watermark in a params JSON), read back via a `tool_calls → transcript` join. The summary is double-sourced: prepended inside `get_previous_messages()` **and** surfaced via `_wrap_with_checkpoint()`.

5. **Raw params are threaded into `Providers`.** `send`/`send_messages`/`calculate` take `(system_prompt, messages, tools, job, …, mp)` — context leaks out of the orchestrator instead of being scaffolded from it.

---

## 2. Principles

- **`MessageProcessor` (`mp`) is THE orchestrator** — the context/glue of the whole request lifecycle.
- **Collaborators receive only `mp` and scaffold internally** — never raw payload params (mirrors `ProcessorConfig.get_system_prompt(mp)`). See `feedback_orchestrator_pass_mp_only`.
- **Reuse existing code paths** — transcript writers, `ActTrail`, `ActEventEmitter`, the canonical watermark reader. No new helper where one exists.
- **Lean / SRP / net-negative SLOC.** Deleting branching and audit machinery should outweigh the added pre-flight + retry.

---

## 3. Design

### 3.1 Ownership graph — `mp` owns a bound `Providers` instance

- `Providers` stops being a singleton. `Providers.instance()` is **deleted**.
- `MessageProcessor.__init__` (or `process()` immediately after `config` is assigned) constructs `self.providers = Providers(self)`.
- `Providers.__init__(self, mp)` stores `self.mp = mp`. Provider resolution stays lazy (resolved on demand from `self.mp.config.job`).
- Every public method is **param-free** and reads `self.mp`:
  - `self.providers.send()` — scaffolds `system = mp.config.get_system_prompt(mp)`, `user = _wrap_with_checkpoint(mp.config.channel, mp.config.get_user_prompt(mp))`, `tools = AbilityRegistry.build_tools(mp)`, `job = mp.config.job`, `thinking_mode = mp.thinking_level`.
  - `self.providers.pre_flight_check()` — measures and raises `ContextOverflowError`.
  - `self.providers.selected_provider()`, `.count_tokens()`, `.get_context_limit()`.
- Tools / dispatch reach it via the orchestrator: `self.mp.providers.selected_provider()`.
- `_log_after_call` and the per-channel send counters read `self.mp` — token attribution is automatic (the instance *is* bound to the right `mp`).
- Each compaction sub-`mp` gets its own `Providers(mp)` bound to itself — correct and cheap.

### 3.2 Trigger relocation — provider detects, orchestrator reacts

`MessageProcessor._loop()` collapses to:

```python
def _loop(self):
    while True:
        if self._should_stop():
            return ""
        if len(self._previous_rows()) > 50:      # proactive turn-count compaction
            self._compact()
        try:
            response = self.providers.send()      # scaffolds + pre-flights internally
        except (ContextOverflowError, PayloadTooLargeError):
            if self._compaction_retries >= 2:     # cap → error
                raise
            self._compaction_retries += 1
            self._compact()
            continue                              # retry re-reads the now-compacted DB
        self._compaction_retries = 0              # reset on every successful send
        <dispatch response.tool_calls / return response.text>
```

- `self._compaction_retries` lives on `mp`, increments per reactive compaction, **resets to 0 after any successful send**.
- The `>50 turns since watermark` check is **proactive** (does not consume the retry budget) and reuses the prev-messages query: `len(self._previous_rows()) > 50`.

**Deleted from the loop:** the `>0.90/>0.80` branching, `_compact_trail`, `_compact_history`, the `Providers.calculate()` call, the `_COMPACTION_CHANNELS` guard, and the dead `_overflow_recovered_this_turn`.

### 3.3 `pre_flight_check()` — size/budget, inside `send()`

Runs before the real provider call, measuring the request `send()` just scaffolded:

```
body    = provider.build_request_body(system, [user], tools)
req     = estimate_tokens(body)
window  = self.get_context_limit()          # declared max_tokens, capped 200k (see below)
headroom = min(8000, int(0.10 * window))    # clamp: 8k on large windows, 10% on small
if window and (req >= 0.90 * window or (window - req) <= headroom):
    raise ContextOverflowError(...)
```

**Window source — DECISION (Dylan, 2026-06-05):** `get_context_limit()` must return the **declared `max_tokens` for the active provider/model, hard-capped at 200k** (`MAX_CONTEXT_WINDOW`). So even a 1M-context model (e.g. Opus) is treated as 200k. Today `Providers.get_context_limit()` ignores the persisted column and re-calls the live provider method (`Anthropic.get_context_limit()` hardcodes `return 200_000`), so the backfilled `providers.max_tokens` is dead at runtime and a misconfigured small-window model would 413 instead of pre-empting. Evidence:
- `providers.max_tokens` is **already** populated with each model's declared window, capped 200k, by `provider_token_limits.backfill_one` (`services/provider_token_limits.py:51-58` — `max_tokens = min(int(svc.get_context_limit()), MAX_CONTEXT_WINDOW)`), run on provider create/boot.
- `ProviderCacheService.get_selected_provider()` (`services/provider_cache_service.py:128`) returns platform/model/host/api_key/dimensions/timeout but **omits `max_tokens`** — must add it.

**New `get_context_limit()` (param-free, on the mp-bound `Providers`):**
```
config   = ProviderCacheService.get_selected_provider() or {}
declared = config.get('max_tokens')
if declared and declared > 0:
    return min(int(declared), MAX_CONTEXT_WINDOW)
return min(self._resolve(self.mp.config.job).get_context_limit(), MAX_CONTEXT_WINDOW)  # first-boot fallback, pre-backfill
```
Normal operation is unchanged (Anthropic backfills to 200k → still 200k; Ollama backfills to its live window → same value), but the window is now DB-driven and overridable — which (a) fixes the misconfig/413 risk and (b) gives the nightly scenario a deterministic knob: `UPDATE providers SET max_tokens = <N>`.

**Why the clamp (evidence):** an unclamped 8k floor (`window - req <= 8000`) is `req >= window - 8000`; for a window ≤ 8000 that is `req >= 0`, so **every** request would trip overflow → `_compact()` no-ops on a fresh small turn → 2-retry cap → hard error. `min(8000, 0.10*window)` keeps the literal 8k headroom on production-size windows and degrades to a 10% headroom on tiny ones, eliminating the pathology at every window size.

**Floor caveat (scenario calibration).** A window cannot be set below the fixed per-turn overhead (system prompt + tool schemas + minimal input). If a single turn's `req` exceeds `window`, `_compact()` cannot bring it under (the overhead is irreducible) → 2-retry cap → `ContextOverflowError`. The nightly trigger therefore sets `max_tokens` **above** the measured single-turn overhead and accumulates history across several turns to cross `0.90*window`. The exact value is calibrated empirically from `llm_call_log` prompt-token counts during the pre-development baseline (the one phase where scenario edits are still permitted).

**Note:** `compact_at` (provider DB column) remains **not read anywhere in the runtime path** — only written by `provider_token_limits.backfill_one`. The redesign's trigger basis is window-derived via the declared `max_tokens` above; `compact_at` stays a dead column (out of scope to remove).

**No compaction exemption.** If a compaction sub-`mp`'s own request is too big, its `_previous_rows()` is empty (the `compaction` channel has no history) and it has no trail, so `_compact()` is a no-op → it hits the 2-retry cap → raises. The cap is the universal bound; unbounded recursion is structurally impossible.

### 3.4 413 handling

The existing `PayloadTooLargeError` (raised by `OllamaService` on HTTP 413, re-raised by `FallbackLLMService`) propagates out of `providers.send()` and is caught by the same `except` block — compact + retry, max 2, then raise.

### 3.5 `_compact()` — reuse write + emit paths

1. **History → `transcript`** (reuses the existing writer):
   `raw = MessageProcessor.process(self.get_previous_messages(), CompactionConfig())`
   `summary = _extract_compaction_summary(raw)`
   `transcript_service.write_input_row(channel, 'compaction', summary)`.
   The written row's **own id is the watermark**.
   Skip when there is nothing to summarise (or when extraction yields no `<summary>`).

   **Decision (2026-06-05, Dylan — "retain old behaviour in terms of compaction"):**
   `CompactionConfig`'s system prompt (`ContinuityCompactionSystemPrompt`) emits
   `<analysis>…</analysis><summary>…</summary>`. The `<analysis>` is a
   discard-after-use reconciliation scaffold (audit each prior fact
   still-true/updated/contradicted/dead before compressing — the guard against
   lossy or fresh-start summaries); only the dense `<summary>` body is the
   durable artifact injected into later turns via `_wrap_with_checkpoint`.
   Persist the **extracted** `<summary>`, never the raw blob — writing the raw
   output leaks the scratchpad and literal XML tags into every subsequent
   prompt. This restores the pre-redesign `_run_full_compaction` behaviour and
   re-wires `_extract_compaction_summary` into the live path. **Resolves the
   plan's Risk R2.** `TrailHandoverConfig` output (item 2) stays raw — its
   prompt emits no tags. (Implemented in commit `1fccaf07`.)

2. **Trail → `tool_calls`** (only when a trail exists, i.e. not turn-0):
   `handover = MessageProcessor.process(self._render_act_trail(), TrailHandoverConfig())`
   `ActTrail().record(tool_name='trail_compaction', params={}, result=handover, transcript_id=self.uid, ephemeral=True)`.
   `TrailHandoverConfig` system prompt: *"Extract a handover summary of what is available in the user-message. Keep your response concise."*

3. **Emit to the act-trail** (reuses the existing record+emit pair, the `_record_narration` pattern):
   `ActEventEmitter(self.config).emit({...})` alongside the `ActTrail().record(...)`.
   **Note (evidence):** `ActTrail.record()` is a pure INSERT and does **not** emit WS (`act_trail.py`). The WS emit lives in `ActEventEmitter`, used by `_record_narration` (`message_processor.py`) and `ToolDispatcher._execute`. Compaction reuses that same explicit emit; it is not a side-effect of `record`.

4. Reset `self.current_iteration = 0`.

### 3.6 Watermark = the compaction row's own id

- `compaction_persistence.get_compaction(channel)` is **repointed** from `tool_calls` to: latest `transcript` row `WHERE channel=? AND role='compaction' ORDER BY id DESC LIMIT 1`. Returns `compacted_text = row.content`, `compacted_up_to_id = row.id`.
- It remains the single canonical watermark reader for its three consumers: `get_previous_messages`, `_wrap_with_checkpoint`, `transcript_service.cleanup_unlinked_entries`.
- The trail watermark is unchanged — positional, via `_from_last_compaction()` slicing from the last `trail_compaction` row.

### 3.7 `get_previous_messages()` — the literal query

```sql
SELECT * FROM transcript WHERE channel = ? AND id > <watermark> ORDER BY id ASC
```

- **No `LIMIT`** (fixes the 20-row bug).
- **No summary prepend** — the summary is surfaced once, via `### Checkpoint` (`_wrap_with_checkpoint`).
- **No durable tool-call interleave** — tool history lives in the act-trail.
- Split into `_previous_rows()` (the query, reused by the 50-turn count) + a render step.
- Post-compaction `id > watermark` is empty (the compaction row's id *is* the watermark, so it and everything before it are excluded).

---

## 4. Consumers to update / breakages

1. **`api/conversation.py`** — the user conversation query `WHERE channel='user' AND role NOT IN ('subagent_return')` must add `'compaction'`, else the summary row leaks into the chat view.
2. **Brain observability tab** — reads compaction from `tool_calls`; repoint to the `transcript` `role='compaction'` source.
3. **Nightly scenario `120-compaction-continuity-long-history.yaml`** — two changes, both before code (must **fail at the pre-development baseline**):
   - **Assertion:** the poll + db_query assert a `tool_calls` `tool_name='compaction'` success row. Under the new model the summary is a `transcript` `role='compaction'` row → repoint both to `SELECT … FROM transcript WHERE channel='user' AND role='compaction'`.
   - **Trigger:** the old `UPDATE providers SET compact_at = 400` step is a **no-op** (compact_at is unread; see §3.3). Re-point the trigger to `UPDATE providers SET max_tokens = <N>` — which, once `get_context_limit()` reads the declared `max_tokens` (§3.3 decision), deterministically lowers the window so accumulated history crosses `0.90*N`. `<N>` is set **above** the measured single-turn overhead (else a single turn can never fit → hard error; see §3.3 floor caveat) and the seeded history is sent across enough turns to exceed it. Calibrate `<N>` and the per-message sizes from `llm_call_log` prompt-token counts during the pre-development baseline. Restore with `UPDATE providers SET max_tokens = NULL` (runtime then falls back to the live method). Drop the `compact_at` UPDATE / reset steps.

## 5. Deletions (SLOC reduction targets)

- `Providers.calculate()` and the singleton `instance()` / `_instance` / `_lock` — **fully removed** once §6's three callers are converted (no remaining `Providers.instance()` references).
- `_compact_trail()`, `_compact_history()`.
- `_write_compaction_audit_row()`, `_build_compaction_input()`, `_format_compaction_entry()`, and the `_run_full_compaction` audit path (history compaction is now a single transcript write).
- `_overflow_recovered_this_turn`, `_COMPACTION_CHANNELS`.
- The summary prepend + tool-call interleave inside `get_previous_messages()`.
- **Thinking hack (§6.1):** `_run_thinking_exploration()`, `_persist_exploration_to_tool_calls()`, `self._thinking_exploration`, `mp.thinking_exploration`, the six `=None` resets, and the `## Chain of Thought` branch in `configs/channels/user.py`.

---

## 6. Singleton retirement — the three remaining `Providers` callers

These three callers are the only remaining users of `Providers.instance()`. They are **in scope**: converting them removes the last singleton references so `Providers.instance()` can be deleted entirely (§3.1, §5). Each was implemented as a one-off workaround; each collapses onto an existing pattern.

### 6.1 `_run_thinking_exploration()` → internal `thinking` ability

The high-thinking exploration is currently a hack: a one-off `Providers.send_messages(..., thinking_mode='high')` whose result is stashed on `self._thinking_exploration`, persisted to `tool_calls` via `_persist_exploration_to_tool_calls()`, **and** special-cased back into the prompt as a `## Chain of Thought` prepend in `configs/channels/user.py`. The attribute is threaded through `mp.__init__` and reset to `None` in six files.

**Redesign** — a delegate-style ability, dispatched programmatically:

- New `ThinkingAbility` + `ThinkingConfig`, mirroring the `web_search` template (`run()` calls `MessageProcessor.process(goal, ThinkingConfig(...))`). No MP subclass.
- **`ThinkingAbility` is never exposed to the model** — not in `DEFAULT_DISCOVERABLE`, not in any `always_available`. The LLM never sees `thinking` in `find_tools` or its toolbox. It is dispatched **programmatically at turn 0** when the thinking-gate resolves `high`, exactly like the `memory.recall` turn-0 seed (`_seed_turn_zero`). It is a convenience tool the orchestrator fires, not a discoverable capability.
- **`ThinkingConfig` retains the parent channel's full tool surface** — constructed from the parent config's `always_available` + `discoverable` (and `policy_channel`, inherited from the caller exactly as `WebSearchConfig` does). The model still sees the tool catalogue so it can reason about which tools would help; the prompt forbids invoking them and any `tool_calls` in the response are discarded. `thinking_mode='high'`, single pass.
- The exploration is recorded into the act-trail by the normal dispatch → `ActTrail.record` path (`tool_name='thinking'`, durable), on the **parent**'s transcript id, so it surfaces through `_render_act_trail` and flows back into the next `get_user_prompt` **for free** — no special attribute, no re-injection branch.

**Deletions:** `_run_thinking_exploration`, `_persist_exploration_to_tool_calls`, `self._thinking_exploration`, `mp.thinking_exploration`, the six `=None` resets (`subconscious_worker` ×2, `transcript_service`, `skill_suggestion_message_processor`, `message_processor` init + reset), and the `## Chain of Thought` branch in `configs/channels/user.py`.

### 6.2 `skill_association_service` → own MP loop

`services/skill_association_service.py` sends from the subconscious worker via `Providers.instance().send(user_prompt=…, system_prompt=_SYSTEM_PROMPT, job='subconscious', tools=[])` — no `mp`.

**Redesign** — its own MessageProcessor loop with a `SkillAssociationConfig`:

```python
result = MessageProcessor.process(pattern_context, SkillAssociationConfig())
self._write_associations(self._parse_associations(result))
```

`SkillAssociationConfig.get_system_prompt` returns `_SYSTEM_PROMPT`; `get_user_prompt` returns the patterns/skill-index prompt the service already builds. Loading, parsing, and writing associations stay in the service (its SRP) — only the send becomes a real `mp` loop. Removes the last `Providers.instance().send()` caller.

### 6.3 `substitute_provider_content_field` → inline via `mp.providers`

This is **not a send** — it swaps the `{{provider_content_field_name}}` placeholder for the active provider's `CONTENT_FIELD_LABEL` (`content[].text`, `message.content`, …) during system-prompt assembly, so the model is told the JSON field its prose lands in. It only touched `Providers.instance()` to read that one label.

**Redesign** — read the label through the mp-owned instance, no singleton:

- `CONTENT_FIELD_LABEL` already lives on every provider sub-class (verified: `AnthropicService`, the OpenAI service, `GeminiService`, `OllamaService`) — no change there.
- The helper takes `mp` and reads the resolved provider:
  ```python
  def substitute_provider_content_field(body, mp):
      if _CONTENT_FIELD_PLACEHOLDER not in body:
          return body
      label = mp.providers.selected_provider().CONTENT_FIELD_LABEL
      return body.replace(_CONTENT_FIELD_PLACEHOLDER, label)
  ```
- Both call sites (`configs/channels/user.py:135`, `configs/channels/external_agent.py:106`) are already inside `get_system_prompt(self, mp)` builders, so `mp` is in scope — the call becomes `substitute_provider_content_field(prompt, mp)`. `selected_provider()` is the param-free §3.1 method (resolves from `mp.config.job`).

---

## 7. New configs

- `TrailHandoverConfig` — `compaction`-style `ProcessorConfig` (no tools, no transcript, suppress history), system prompt = the handover instruction in §3.5. Reuses `CompactionConfig` shape; differs only in the system prompt.
- `ThinkingConfig` (§6.1) — single-pass, `thinking_mode='high'`, retains the **parent channel's** `always_available` + `discoverable` + `policy_channel` (constructor args, as `WebSearchConfig` takes `policy_channel`); prompt forbids tool invocation, `tool_calls` discarded.
- `SkillAssociationConfig` (§6.2) — `subconscious`-job config; `get_system_prompt` = `_SYSTEM_PROMPT`, `get_user_prompt` = the patterns/skill-index prompt; no tools, no transcript writes.
