"""
SubconsciousWorker — idle-gated 5-minute cognition tick.

A single daemon thread that owns latent cognition: super-episode consolidation,
decay, pattern extraction, user synthesis, and background DMN reflection.
Fires only when the user is not active.

Spec: ``/Volumes/llm/chalie-plans/v0.5.0/2026-04-24-subconscious-worker-design.md``.

Tick body (sequential, per §5.3):
    1. Consolidate episodes → super-episodes  (channel='user' only — §B masterplan).
    2. Run decay engine.
    3. Run pattern match.
    4. Run user synthesis.
    5. Run DMN reflection (skipped when no user_summary row exists).

Gates (both must pass — §5.2):
    - User-active: ``last_user_message_at`` is older than 30 minutes.
    - Already-fired: ``subconscious_last_fired_at > last_user_message_at``.

Each step is wrapped in ``try/except``; one bad step does not skip the rest.
Step 5 (DMN) self-gates when ``user_summary`` is absent in data_graph.

State persistence — ``subconscious_last_fired_at``:
    - MemoryStore key ``subconscious:last_fired_at`` (fast read).
    - data_graph row ``kind='system' key='subconscious_last_fired_at'``
      (durable across restarts; reloaded into MemoryStore on first run).
"""

import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"

# ── Tunables (env-overridable) ────────────────────────────────────────────────

DEFAULT_TICK_SEC = 300              # 5 minutes — spec §5.1
DEFAULT_IDLE_WINDOW_SEC = 1800      # 30 minutes — spec §5.2 user-active gate

_MEMORY_KEY_LAST_FIRED = "subconscious:last_fired_at"
_DG_KEY_LAST_FIRED = "subconscious_last_fired_at"

# Keys checked to determine whether DMN has a synthesis to work from.
# SubconsciousWorker._step_dmn() skips when neither row is present.
_DMN_SYNTHESIS_KEYS = ('user_summary', 'user_summary_long')


def _env_int(name: str, default: int) -> int:
    """Read an int env-var with a default fallback. Invalid values fall through."""
    try:
        raw = os.environ.get(name)
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


class SubconsciousWorker:
    """Idle-gated tick orchestrator. Stateless across ticks, except for last_fired_at.

    Public API:
        - ``run_once()`` — fire one tick (gates, steps, state update).
        - ``last_fired_at`` property — current persisted value (or None).

    Re-entrancy: a non-blocking lock guards against overlapping ticks. Cheap
    insurance even though the 5-minute spacing makes overlap unlikely.
    """

    def __init__(
        self,
        tick_sec: int = DEFAULT_TICK_SEC,
        idle_window_sec: int = DEFAULT_IDLE_WINDOW_SEC,
    ):
        self.tick_sec = tick_sec
        self.idle_window = timedelta(seconds=idle_window_sec)
        self._lock = threading.Lock()
        self._cached_last_fired: Optional[datetime] = None
        # Single DecayEngineService instance — the engine reads
        # ``ConfigService.get_agent_config('episodic-memory')`` in __init__,
        # so re-instantiating per tick would burn a config read every 5 min.
        # Lazy-built on first use so import failures surface as a step error.
        self._decay_engine = None
        # Hydrate from durable state on construction so the first tick after a
        # restart sees the correct already-fired value. Failure is non-fatal —
        # worst case we run one extra tick on first idle window after a crash.
        try:
            self._cached_last_fired = self._load_last_fired_from_storage()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} hydrate last_fired_at failed: {exc}")

    # ── Public entry ─────────────────────────────────────────────────────────

    def run_once(self) -> dict:
        """Run one full tick. Returns a structured summary.

        Summary keys:
            - ``skipped``: gate name when both-gates check rejects the tick
              (``user_active`` | ``already_fired`` | ``re_entrant``); absent
              when steps run.
            - ``steps``: dict per step with ``status`` (``ok`` | ``skipped`` |
              ``error``) and optional ``detail`` / ``error`` fields.
            - ``last_fired_at``: ISO string when state was bumped, else ``None``.
        """
        if not self._lock.acquire(blocking=False):
            logger.info(f"{LOG_PREFIX} tick already running — skipping re-entry")
            return {"skipped": "re_entrant", "steps": {}, "last_fired_at": None}

        try:
            gate_skip = self._check_gates()
            if gate_skip is not None:
                logger.info(f"{LOG_PREFIX} tick skipped: {gate_skip}")
                return {"skipped": gate_skip, "steps": {}, "last_fired_at": None}
            return self._tick()
        finally:
            self._lock.release()

    @property
    def last_fired_at(self) -> Optional[datetime]:
        """Current cached last-fired timestamp. ``None`` when never fired."""
        return self._cached_last_fired

    # ── Tick orchestration ───────────────────────────────────────────────────

    def _tick(self) -> dict:
        """Body of one tick after gates have been cleared.

        Steps are strictly sequential — each step completes before the next
        begins. Step 4 (synthesis) writes the user_summary row that step 5
        (DMN) reads; the sequential contract ensures step 5 always sees the
        latest synthesis output.
        """
        steps: dict = {}
        steps["consolidate"] = self._safe_step("consolidate", self._step_consolidate)
        steps["decay"] = self._safe_step("decay", self._step_decay)
        steps["pattern_match"] = self._safe_step("pattern_match", self._step_pattern_match)
        steps["synthesis"] = self._safe_step("synthesis", self._step_synthesis)
        steps["dmn"] = self._safe_step("dmn", self._step_dmn)

        now = utc_now()
        try:
            self._persist_last_fired(now)
            self._cached_last_fired = now
            last_iso = now.isoformat()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} persist last_fired_at failed: {exc}")
            last_iso = None

        logger.info(
            f"{LOG_PREFIX} tick complete: "
            f"consolidate={steps['consolidate']['status']} "
            f"decay={steps['decay']['status']} "
            f"pattern_match={steps['pattern_match']['status']} "
            f"synthesis={steps['synthesis']['status']} "
            f"dmn={steps['dmn']['status']}"
        )
        return {"steps": steps, "last_fired_at": last_iso}

    def _safe_step(self, name: str, fn) -> dict:
        """Run a single step under try/except. Returns step status dict."""
        try:
            detail = fn()
            return {"status": "ok", "detail": detail}
        except Exception as exc:
            logger.exception(f"{LOG_PREFIX} step '{name}' failed")
            return {"status": "error", "error": str(exc)}

    # ── Gates ────────────────────────────────────────────────────────────────

    def _check_gates(self) -> Optional[str]:
        """Return the name of the failing gate, or ``None`` when both pass.

        Gate 1 (user-active) — Skip when ``last_user_message_at`` is within the
        idle window. We are conservative: when the snapshot does not yet have
        a user-message timestamp (cold boot, no traffic), Gate 1 passes (we
        treat the system as idle so latent cognition can run).

        Gate 2 (already-fired) — Skip when ``last_fired_at`` is newer than
        ``last_user_message_at``. The worker has already covered this idle
        window; a new user message must reset the comparison before it fires
        again.

        Post-restart hydrated-state edge case — when ``last_fired_at`` is
        present (loaded from durable storage) but ``last_user_message_at`` is
        still ``None`` (WorldState is cold and no user has talked since boot),
        we have already fired during the previous process lifetime and
        nothing has changed since. Skip with ``already_fired`` until a real
        user message arrives.
        """
        from services.world_state import world_state

        snapshot = world_state.snapshot()
        last_msg = snapshot.get("last_user_message_at")
        now = utc_now()

        if last_msg is not None and now - last_msg < self.idle_window:
            return "user_active"

        last_fired = self._cached_last_fired
        if last_fired is not None:
            # Hydrated last_fired with no user activity since boot — the tick
            # has nothing new to consolidate. Wait for a real signal.
            if last_msg is None or last_fired > last_msg:
                return "already_fired"

        return None

    # ── Steps ────────────────────────────────────────────────────────────────

    def _step_consolidate(self) -> str:
        """Step 1 — consolidate apex episodes into super-episodes (channel='user' only).

        Iterates only channel='user' episodes. This is both a performance
        optimisation (the only channel that produces episodes post-masterplan,
        since _maybe_trigger_extraction is gated to channel='user' upstream)
        and a correctness guarantee: legacy pre-migration channels with residual
        episodes are intentionally excluded.

        SuperEpisodeEncoderProcessor is self-gating: ``send()``
        returns '' immediately when ``find_super_candidates(channel)`` finds
        nothing, so the call is cheap when nothing has accumulated.
        """
        from services.super_episode_encoder_processor import SuperEpisodeEncoderProcessor

        try:
            summary = SuperEpisodeEncoderProcessor(channel='user').send()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} consolidate channel=user failed: {exc}")
            raise
        return summary if summary else "checked channel=user, no clusters formed"

    def _step_decay(self) -> str:
        """Step 2 — run the unified decay cycle.

        DecayEngineService owns episodic + data_graph + transcript cleanup +
        tool_calls purge + behavioural-pattern stale flips. Engine logic
        unchanged; only the trigger surface lives here now.

        The engine is cached on the worker instance so we avoid the
        ``ConfigService.get_agent_config`` read on every 5-minute tick.
        """
        if self._decay_engine is None:
            from services.decay_engine_service import DecayEngineService
            self._decay_engine = DecayEngineService()
        self._decay_engine.run_once()
        return "ok"

    def _step_pattern_match(self) -> str:
        """Step 3 — single-pass LLM pattern matcher over a transcript-id window.

        Reads a cursor row from data_graph (kind='system'
        key='pattern_match_cursor'). If MAX(transcripts.id) - cursor < 50,
        skip. Else fire PatternMatchProcessor over the (cursor, latest] window
        and advance the cursor on success.
        """
        from services.data_graph_service import get_data_graph_service
        from services.database_service import get_shared_db_service
        from services.pattern_match_processor import PatternMatchProcessor

        _DG_KEY_CURSOR = "pattern_match_cursor"
        _MIN_DELTA = 50

        db = get_shared_db_service()

        # 1. Read cursor — newest active row wins. The deterministic
        # ORDER BY id DESC defends against historical / concurrent writes
        # leaving more than one active row (UPSERT collisions in
        # data_graph_service.store path); SQLite's default ordering is
        # implementation-defined and would silently pick the wrong row.
        cursor = 0
        with db.connection() as conn:
            row = conn.execute(
                "SELECT value FROM data_graph "
                "WHERE kind='system' AND key=? "
                "AND active=1 AND deleted_at IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (_DG_KEY_CURSOR,),
            ).fetchone()
            if row and row[0]:
                try:
                    cursor = int(row[0])
                except (TypeError, ValueError):
                    cursor = 0

        # 2. Read latest transcript id
        with db.connection() as conn:
            latest_row = conn.execute("SELECT MAX(id) FROM transcript").fetchone()
        latest = (latest_row[0] if latest_row else None) or 0

        delta = latest - cursor
        if delta < _MIN_DELTA:
            logger.info(
                f"{LOG_PREFIX} pattern_match_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # 3. Fire processor
        PatternMatchProcessor(window_start=cursor, window_end=latest).send()

        # 4. Advance cursor on success.
        # Cursor write race / known risk: a crash here leaves the cursor
        # pinned to its previous value while the processor has already
        # decayed every untouched pattern. The next tick re-fires the same
        # window and re-decays — visible in tests as confidence drifting
        # below the "−0.005 per cycle" expectation. Spec accepts this as a
        # minor double-fire risk; logging at WARNING so an unexpected
        # cursor-stuck pattern is observable in operator logs.
        try:
            get_data_graph_service().store(
                kind="system",
                key=_DG_KEY_CURSOR,
                value=str(latest),
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} pattern_match cursor write failed "
                f"cursor={cursor}->{latest} — next tick will re-fire same "
                f"window and re-decay untouched patterns: {exc}"
            )
            raise
        return f"fired cursor={cursor}->{latest} delta={delta}"

    def _step_synthesis(self) -> str:
        """Step 4 — refresh the user synopsis (short + long).

        UserSummaryProcessor self-gates via ``_should_synthesise()`` — when no
        new traits or behavioural patterns have arrived since the last
        synthesis it silently returns ''.  Inputs are now the union of
        Episodes + Data Graph + Extracted Patterns.

        The resulting ``user_summary`` / ``user_summary_long`` rows in
        data_graph are the prerequisite for step 5 (DMN). Sequential execution
        guarantees step 5 sees the freshest synthesis output.
        """
        from services.user_summary_processor import UserSummaryProcessor

        result = UserSummaryProcessor().send()
        return "ok" if result else "no new traits/patterns; skipped"

    def _step_dmn(self) -> str:
        """Step 5 — background DMN reflection via DMNMessageProcessor.

        Runs DMNMessageProcessor which reads user synthesis + recent episodes,
        acts on open threads using news/search/browser/memory tools, and saves
        all findings to data_graph via the memory tool. No UI broadcast.

        Prerequisites (checked before constructing the processor):
            - ``user_summary`` or ``user_summary_long`` must exist in data_graph.
              Step 4 (synthesis) runs in the same tick and may have just produced
              this row; this check runs after step 4 completes.

        Skips (returns early) when no synthesis row exists.
        DMN has nothing meaningful to reflect on without a user model.
        """
        synthesis = self._load_user_synthesis()
        if not synthesis:
            logger.info(f"{LOG_PREFIX} Skipping DMN — no user synthesis available")
            return "skipped: no user synthesis"

        from services.dmn_message_processor import DMNMessageProcessor
        DMNMessageProcessor(raw_input='').send()
        return "ok"

    def _load_user_synthesis(self) -> Optional[str]:
        """Check whether a user_summary row exists in data_graph.

        Returns the synthesis value when present, None otherwise.
        Prefers user_summary_long for completeness but falls back to
        user_summary — matching the same preference logic used by
        DMNMessageProcessor._fetch_user_synthesis().
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT key, value FROM data_graph "
                    "WHERE kind = 'system' "
                    "  AND key IN ('user_summary', 'user_summary_long') "
                    "  AND active = 1 AND deleted_at IS NULL",
                ).fetchall()
            by_key = {row[0]: row[1] for row in rows if row[1]}
            return by_key.get('user_summary_long') or by_key.get('user_summary')
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} _load_user_synthesis failed: {exc}")
            return None

    # ── State persistence ───────────────────────────────────────────────────

    def _load_last_fired_from_storage(self) -> Optional[datetime]:
        """Read last_fired_at from MemoryStore first, fall back to data_graph.

        MemoryStore is the fast path; data_graph survives MemoryStore eviction
        and process restarts.
        """
        # Fast path — MemoryStore.
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            raw = store.get(_MEMORY_KEY_LAST_FIRED)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                return parse_utc(raw)
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} memory hydrate skipped: {exc}")

        # Durable fallback — data_graph kind='system'.
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM data_graph "
                    "WHERE kind='system' AND key=? AND active=1 AND deleted_at IS NULL "
                    "LIMIT 1",
                    (_DG_KEY_LAST_FIRED,),
                ).fetchone()
            if row and row[0]:
                return parse_utc(row[0])
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} data_graph hydrate skipped: {exc}")
        return None

    def _persist_last_fired(self, when: datetime) -> None:
        """Write last_fired_at to MemoryStore + data_graph.

        Best-effort across both stores; either failure is logged but does not
        abort the tick. The next tick's gate logic still sees the cached
        in-process value via ``self._cached_last_fired``.

        Both write paths log at WARNING — split-brain (one written, one not)
        is real state divergence we want operators to see. DEBUG would have
        made it silent.
        """
        iso = when.isoformat()
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            store.set(_MEMORY_KEY_LAST_FIRED, iso)
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} memory persist skipped: {exc}")

        try:
            from services.data_graph_service import get_data_graph_service, KIND_SYSTEM
            get_data_graph_service().store(
                kind=KIND_SYSTEM,
                key=_DG_KEY_LAST_FIRED,
                value=iso,
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} data_graph persist skipped: {exc}")


# ── Module-level worker entry (registered in run.py) ─────────────────────────

_DEFAULT_INSTANCE: Optional[SubconsciousWorker] = None
_DEFAULT_INSTANCE_LOCK = threading.Lock()


def get_subconscious_worker() -> SubconsciousWorker:
    """Return the process-wide SubconsciousWorker singleton (lazy-init)."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None:
        with _DEFAULT_INSTANCE_LOCK:
            if _DEFAULT_INSTANCE is None:
                tick_sec = _env_int("SUBCONSCIOUS_TICK_SEC", DEFAULT_TICK_SEC)
                idle_sec = _env_int("SUBCONSCIOUS_IDLE_WINDOW_SEC", DEFAULT_IDLE_WINDOW_SEC)
                _DEFAULT_INSTANCE = SubconsciousWorker(
                    tick_sec=tick_sec,
                    idle_window_sec=idle_sec,
                )
    return _DEFAULT_INSTANCE


def subconscious_worker():
    """WorkerManager entry point. Tick loop with a stable cadence.

    The first tick is delayed by the configured tick interval so the worker
    does not fire during boot before any user message has arrived (which
    would defeat the user-active gate's intent).
    """
    worker = get_subconscious_worker()
    interval = max(1, worker.tick_sec)
    logger.info(f"{LOG_PREFIX} Service started (tick={interval}s)")

    next_tick = time.monotonic() + interval
    while True:
        try:
            sleep_secs = max(0.0, next_tick - time.monotonic())
            time.sleep(sleep_secs)
            worker.run_once()
            # Schedule the next tick from the moment the current one finished.
            # If a tick takes longer than ``interval`` (e.g. a slow consolidate
            # step), the previous ``next_tick += interval`` would land in the
            # past and immediately re-fire — burning CPU and starving idle
            # gates. Anchoring on ``monotonic()`` after the work runs gives
            # us a stable cadence under variable workload.
            next_tick = time.monotonic() + interval
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            return
        except Exception as exc:
            # Worker.run_once() already swallows step exceptions; this catch
            # only matters for un-anticipated failures (e.g. import errors).
            logger.exception(f"{LOG_PREFIX} unexpected tick error: {exc}")
            next_tick = time.monotonic() + interval
