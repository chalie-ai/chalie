"""
SubconsciousWorker — idle-gated 5-minute cognition tick.

A single daemon thread that owns latent cognition: super-episode consolidation,
decay, pattern extraction, user synthesis, and background DMN reflection.
Fires only when the user is not active.

Tick body (sequential, per §5.3):
    1. Consolidate episodes → super-episodes  (channel='user' only — §B masterplan).
    2. Run decay engine.
    3. Run pattern match.
    4. Run user synthesis.
    5. Run DMN reflection (skipped when no user_summary row exists).
    6. Capability sync (IMAP/CalDAV/CardDAV via MailCapability._do_monitor).
    7. Geo-spatial pattern extraction from location-tagged transcripts.

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
        # Single DecayEngineService instance — shared across ticks.
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
        steps["capability_sync"] = self._safe_step("capability_sync", self._step_capability_sync)
        steps["geo_patterns"] = self._safe_step("geo_patterns", self._step_geo_patterns)

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
            f"dmn={steps['dmn']['status']} "
            f"capability_sync={steps['capability_sync']['status']} "
            f"geo_patterns={steps['geo_patterns']['status']}"
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

        §3c / O2: cluster loop lives here, not inside the processor.  One
        MessageProcessor.process() call per cluster.

        Iterates only channel='user' episodes. This is both a performance
        optimisation (the only channel that produces episodes post-masterplan,
        since _maybe_trigger_extraction is gated to channel='user' upstream)
        and a correctness guarantee: legacy pre-migration channels with residual
        episodes are intentionally excluded.
        """
        from configs.channels import SuperEpisodeConfig  # noqa: PLC0415
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.embedding_service import get_embedding_service  # noqa: PLC0415
        from services.episodic_constants import SUPER_EPISODE_MIN_CLUSTER  # noqa: PLC0415
        from services.episodic_service import (  # noqa: PLC0415
            EpisodicService,
            find_super_candidates,
            _fetch_novelty_comparison_set,
            compute_novelty,
        )
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        from services.salience_service import compute_salience  # noqa: PLC0415
        from configs.channels import (  # noqa: PLC0415
            _safe_json_load_object,
            _collect_transcript_ids,
            _fetch_transcript_spans,
        )

        try:
            clusters = find_super_candidates("user")
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} find_super_candidates failed: {exc}")
            raise

        if not clusters:
            return "checked channel=user, no clusters formed"

        db = get_shared_db_service()
        episodic_svc = EpisodicService(db)
        emb_svc = get_embedding_service()

        try:
            prior_embeddings = _fetch_novelty_comparison_set("user")
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} _fetch_novelty_comparison_set failed: {exc}")
            prior_embeddings = []

        supers_written = 0

        for cluster_ids in clusters:
            try:
                sources = [
                    ep for ep in (
                        episodic_svc.get_episode_by_id(eid) for eid in cluster_ids
                    )
                    if ep
                ]
                if len(sources) < SUPER_EPISODE_MIN_CLUSTER:
                    continue

                all_t_ids = _collect_transcript_ids(sources)
                transcript_spans = _fetch_transcript_spans(all_t_ids, db)

                config = SuperEpisodeConfig("user", sources, transcript_spans)
                response = MessageProcessor.process("", config)

                if not response:
                    logger.warning(
                        f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response "
                        f"for cluster {cluster_ids}"
                    )
                    continue

                super_ep = _safe_json_load_object(response)
                if not super_ep or not super_ep.get("gist"):
                    logger.warning(
                        f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty "
                        f"gist for cluster {cluster_ids}"
                    )
                    continue

                super_ep["channel"] = "user"
                unique_t_ids = sorted(all_t_ids)
                super_ep["transcript_ids"] = unique_t_ids
                super_ep["transcript_id_start"] = min(unique_t_ids) if unique_t_ids else None
                super_ep["transcript_id_end"] = max(unique_t_ids) if unique_t_ids else None
                super_ep["consolidated_from"] = [ep["id"] for ep in sources]

                gist = super_ep["gist"]
                embedding = emb_svc.generate_embedding(gist)
                novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
                super_ep["salience"] = compute_salience(
                    valence=float(super_ep.get("emotional_valence") or 0.0),
                    arousal=float(super_ep.get("emotional_arousal") or 0.0),
                    has_open_loop=bool(super_ep.get("has_open_loop", False)),
                    novelty=novelty,
                )
                super_ep.pop("has_open_loop", None)

                new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
                for src_id in cluster_ids:
                    episodic_svc.set_consolidated_into(src_id, new_id)

                logger.info(
                    f"{LOG_PREFIX} Super-episode {new_id} created from cluster {cluster_ids}"
                )
                supers_written += 1

            except Exception as exc:
                logger.warning(
                    f"{LOG_PREFIX} Super-episode creation failed for cluster {cluster_ids}: {exc}"
                )

        if supers_written == 0:
            return f"checked channel=user, {len(clusters)} cluster(s) found, 0 written"
        return f"consolidated {len(clusters)} cluster(s), {supers_written} super-ep(s) written"

    def _step_decay(self) -> str:
        """Step 2 — run the unified decay cycle.

        DecayEngineService owns episodic + data_graph + transcript cleanup +
        tool_calls purge + behavioural-pattern stale flips. Engine logic
        unchanged; only the trigger surface lives here now.

        The engine is cached on the worker instance to reuse the same instance across ticks.
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

        # 3. Fire processor via flat path (§3b / T8 migration).
        # get_user_prompt lazily inits _save_pattern_calls / _touched_pattern_ids etc.
        from configs.channels import PatternConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = PatternConfig(cursor, latest)
        # Build the mp manually so we can inspect _touched_pattern_ids after _run().
        pmp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(pmp, "", None)
        pmp.config = config
        pmp.uid = None
        pmp.current_iteration = 0
        pmp.cancel_event = threading.Event()
        pmp.thinking_level = "low"
        pmp._run()

        # Layer 2: update skill-personalisation associations for touched patterns
        touched = getattr(pmp, "_touched_pattern_ids", set())
        if touched:
            try:
                from services.skill_association_service import SkillAssociationService
                SkillAssociationService().run_pass(touched)
            except Exception as exc:
                logger.warning(
                    f"{LOG_PREFIX} skill_association pass failed: {exc}"
                )

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

        §3c / O1: _should_synthesise() gate lives HERE, not inside the config.
        When no new traits or behavioural patterns have arrived since the last
        synthesis, skip.  Inputs are the union of Episodes + Data Graph + Patterns.

        The resulting ``user_summary`` / ``user_summary_long`` rows in
        data_graph are the prerequisite for step 5 (DMN). Sequential execution
        guarantees step 5 sees the freshest synthesis output.
        """
        from configs.channels import UserSummaryConfig, _should_synthesise  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        if not _should_synthesise():
            logger.info(f"{LOG_PREFIX} No new traits since last synthesis; skipping")
            return "no new traits/patterns; skipped"

        config = UserSummaryConfig()
        MessageProcessor.process("", config)
        return "ok"

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

        from configs.channels import DmnConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        MessageProcessor.process("", DmnConfig())
        return "ok"

    def _step_capability_sync(self) -> str:
        """Step 6 — IMAP / CalDAV / CardDAV server sync.

        Delegates to every connected capability's ``monitor()`` method.
        MailCapability._do_monitor() manages per-protocol cadence internally
        (IMAP every call, CalDAV every 3rd, CardDAV every 12th).
        """
        from capabilities import load_capabilities

        synced = []
        for cap in load_capabilities().values():
            if cap.is_connected():
                cap.monitor()
                synced.append(cap.get_id())
        return f"synced: {', '.join(synced)}" if synced else "no connected capabilities"

    def _step_geo_patterns(self) -> str:
        """Step 7 — single-pass LLM geo-spatial pattern extractor.

        Reads a cursor from data_graph (kind='system' key='geo_pattern_cursor').
        If the count of location-tagged transcripts beyond the cursor is below
        the minimum delta (30), skip. Else fire via flat MessageProcessor.process()
        with GeoConfig and advance the cursor on success.
        """
        from services.data_graph_service import get_data_graph_service
        from services.database_service import get_shared_db_service

        _DG_KEY_CURSOR = "geo_pattern_cursor"
        _MIN_DELTA = 30

        db = get_shared_db_service()

        cursor = 0
        try:
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

            with db.connection() as conn:
                latest_row = conn.execute(
                    "SELECT MAX(id) FROM transcript "
                    "WHERE location_lat IS NOT NULL AND location_lon IS NOT NULL"
                ).fetchone()
            latest = (latest_row[0] if latest_row else None) or 0
        except Exception as exc:
            logger.debug(f"{LOG_PREFIX} geo_patterns no db: {exc}")
            return "skip: no db"

        delta = latest - cursor
        if delta < _MIN_DELTA:
            logger.info(
                f"{LOG_PREFIX} geo_patterns_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # Fire via flat path (§3b / T8 migration).
        # get_user_prompt lazily inits _save_pattern_calls / _touched_pattern_ids etc.
        from configs.channels import GeoConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        config = GeoConfig(cursor, latest)
        gmp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(gmp, "", None)
        gmp.config = config
        gmp.uid = None
        gmp.current_iteration = 0
        gmp.cancel_event = threading.Event()
        gmp.thinking_level = "low"
        gmp._run()

        try:
            get_data_graph_service().store(
                kind="system",
                key=_DG_KEY_CURSOR,
                value=str(latest),
                source="subconscious_worker",
            )
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} geo_patterns cursor write failed "
                f"cursor={cursor}->{latest} — next tick will re-fire same "
                f"window: {exc}"
            )
            raise
        return f"fired cursor={cursor}->{latest} delta={delta}"

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
