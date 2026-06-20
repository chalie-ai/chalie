"""SubconsciousWorker — idle-gated 5-minute cognition tick."""

import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from services.database_service import DatabaseService
    from services.decay_engine_service import DecayEngineService
    from services.embedding_service import EmbeddingService
    from services.episodic_service import EpisodicService
    from services.data_graph_service import DataGraphService

from services.durable_timestamp import DurableTimestamp
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUBCONSCIOUS]"

# ── Tunables (env-overridable) ────────────────────────────────────────────────

DEFAULT_TICK_SEC = 300              # 5 minutes — spec §5.1
DEFAULT_IDLE_WINDOW_SEC = 1800      # 30 minutes — spec §5.2 user-active gate

_MEMORY_KEY_LAST_FIRED = "subconscious:last_fired_at"
_DG_KEY_LAST_FIRED = "subconscious_last_fired_at"
_SOURCE_LAST_FIRED = "subconscious_worker"

# Durable dual-write clock for the already-fired gate (MemoryStore + data_graph).
# Bidirectional dependency: services/durable_timestamp.py owns the persist/hydrate
# mechanism; this module supplies the key pair + provenance.
_LAST_FIRED_TIMESTAMP = DurableTimestamp(
    memory_key=_MEMORY_KEY_LAST_FIRED,
    data_graph_key=_DG_KEY_LAST_FIRED,
    source=_SOURCE_LAST_FIRED,
)

# Keys checked to determine whether DMN has a synthesis to work from.
# SubconsciousWorker._step_dmn() skips when neither row is present.
_DMN_SYNTHESIS_KEYS = ('user_summary', 'user_summary_long')

# Per-tick consolidation summarization cap: at most this many cluster→parent LLM
# summarization calls run per tick across all channels and both roll-up rounds,
# so a large backlog drains over several ticks instead of stalling one tick.
_SUMMARIZATION_CLUSTER_BUDGET = 5

# Fact-extraction step budget (§F / spec §4.6 mechanism 3). The backlog of
# episodes WHERE facts_extracted_at IS NULL drains at a fixed per-tick budget,
# measured in LLM calls so the tick stays bounded regardless of backlog size:
# one extraction call per episode, capped here. A fresh instance processes
# yesterday's episodes; a 30k-episode instance converges over weeks at the same
# rate, never blocking a tick.
_FACT_EXTRACTION_CALL_BUDGET = 20
# Similar data_graph rows shown to the model per episode for reconciliation.
_FACT_NEIGHBOUR_LIMIT = 10
# Provenance prefix stamped on every data_graph row the fact pipeline writes.
# The episode's channel is appended (``fact_extraction:<channel>``) so a fact's
# origin is recoverable; a channel-less episode degrades to the bare prefix.
_FACT_SOURCE = "fact_extraction"


def _fact_source_for(channel: Optional[str]) -> str:
    return f"{_FACT_SOURCE}:{channel}" if channel else _FACT_SOURCE
# Maps a data_graph upsert_fact() status to the fact-extraction telemetry
# counter. A new row (created) counts as an ADD; a contradicting value
# (superseded) counts as an UPDATE; an unchanged write (reinforced) is a NOOP.
# Unlisted statuses default to ADD at the call site.
_FACT_STATUS_COUNTER = {
    "created": "add",
    "superseded": "update",
    "reinforced": "noop",
}


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


class SubconsciousWorker:
    """Idle-gated tick orchestrator. Stateless across ticks, except for last_fired_at."""

    def __init__(
        self,
        tick_sec: int = DEFAULT_TICK_SEC,
        idle_window_sec: int = DEFAULT_IDLE_WINDOW_SEC,
    ) -> None:
        self.tick_sec = tick_sec
        self.idle_window = timedelta(seconds=idle_window_sec)
        self._lock = threading.Lock()
        self._cached_last_fired: Optional[datetime] = None
        # Per-tick consolidation summarization budget; reset at the start of
        # each _step_consolidate run (declared here so _write_round never reads
        # an unset attribute).
        self._summarization_budget_remaining = _SUMMARIZATION_CLUSTER_BUDGET
        # Single DecayEngineService instance — shared across ticks.
        # Lazy-built on first use so import failures surface as a step error.
        self._decay_engine: Optional["DecayEngineService"] = None
        # Hydrate from durable state on construction so the first tick after a
        # restart sees the correct already-fired value. Failure is non-fatal —
        # worst case we run one extra tick on first idle window after a crash.
        try:
            self._cached_last_fired = self._load_last_fired_from_storage()
        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} hydrate last_fired_at failed: {exc}")

    # ── Public entry ─────────────────────────────────────────────────────────

    def run_once(self) -> dict[str, object]:
        """Run one full tick. Returns a structured summary."""
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

    def _tick(self) -> dict[str, object]:
        """Body of one tick after gates have been cleared."""
        steps: dict[str, dict[str, object]] = {}
        steps["consolidate"] = self._safe_step("consolidate", self._step_consolidate)
        steps["fact_extraction"] = self._safe_step("fact_extraction", self._step_fact_extraction)
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
            f"fact_extraction={steps['fact_extraction']['status']} "
            f"decay={steps['decay']['status']} "
            f"pattern_match={steps['pattern_match']['status']} "
            f"synthesis={steps['synthesis']['status']} "
            f"dmn={steps['dmn']['status']} "
            f"capability_sync={steps['capability_sync']['status']} "
            f"geo_patterns={steps['geo_patterns']['status']}"
        )
        return {"steps": steps, "last_fired_at": last_iso}

    def _safe_step(self, name: str, fn: Callable[[], str]) -> dict[str, object]:
        """Run a single step under try/except. Returns step status dict."""
        try:
            detail = fn()
            return {"status": "ok", "detail": detail}
        except Exception as exc:
            logger.exception(f"{LOG_PREFIX} step '{name}' failed")
            return {"status": "error", "error": str(exc)}

    # ── Gates ────────────────────────────────────────────────────────────────

    def _check_gates(self) -> Optional[str]:
        """Return the name of the failing gate, or ``None`` when both pass."""
        from services.world_state import world_state

        snapshot = world_state.snapshot()
        last_msg_raw = snapshot.get("last_user_message_at")
        last_msg = cast(Optional[datetime], last_msg_raw)
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
        """Step 1 — consolidate apex episodes into super-episodes, per channel."""
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.embedding_service import get_embedding_service  # noqa: PLC0415

        db = get_shared_db_service()
        emb_svc = get_embedding_service()

        # Per-tick summarization budget shared across every channel and round.
        # Reset each tick so a backlog drains over successive ticks, never
        # blowing one tick's LLM budget.
        self._summarization_budget_remaining = _SUMMARIZATION_CLUSTER_BUDGET

        channels = self._consolidating_channels(db)
        total_clusters = 0
        supers_written = 0
        for channel in channels:
            found, written = self._consolidate_channel(channel, db, emb_svc)
            total_clusters += found
            supers_written += written

        if total_clusters == 0:
            return f"checked channels={channels}, no clusters formed"
        if supers_written == 0:
            return (
                f"checked channels={channels}, {total_clusters} cluster(s) found, "
                "0 written"
            )
        return (
            f"consolidated {total_clusters} cluster(s) across {len(channels)} "
            f"channel(s), {supers_written} super-ep(s) written"
        )

    @staticmethod
    def _consolidating_channels(db: "DatabaseService") -> list[str]:
        """Return the channels to consolidate: the HEAVY exact channels (user,"""
        from services.source_profiles import (  # noqa: PLC0415
            LIKE_EXTERNAL_AGENT,
            consolidating_exact_channels,
        )

        channels = list(consolidating_exact_channels())
        try:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT channel FROM episodes "
                    "WHERE deleted_at IS NULL "
                    "  AND consolidated_into IS NULL "
                    "  AND channel LIKE ?",
                    (LIKE_EXTERNAL_AGENT,),
                ).fetchall()
            channels.extend(r[0] for r in rows if r[0])
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} external-agent channel discovery failed: {exc}"
            )
        return channels

    def _consolidate_channel(self, channel: str, db: "DatabaseService", emb_svc: "EmbeddingService") -> tuple[int, int]:
        """Consolidate one channel across both hierarchy rounds. Returns"""
        from services.episodic_constants import ERA_DIGEST_TRIGGER  # noqa: PLC0415
        from services.episodic_service import (  # noqa: PLC0415
            EpisodicService,
            cluster_apex_embeddings,
            find_super_candidates,
            _fetch_apex_embeddings,
        )

        episodic_svc = EpisodicService(db)

        # Leaf round — count trigger lives inside find_super_candidates.
        try:
            leaf_clusters = find_super_candidates(channel)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} find_super_candidates failed (channel={channel}): {exc}"
            )
            leaf_clusters = []

        # Era round — cluster stored level-1 summary embeddings when enough
        # level-1 apexes have accumulated. The count gate is explicit here
        # because the era round reads its own (level=1) apex set.
        # Intentional one-tick lag: this reads level-1 apexes BEFORE the leaf
        # round below writes this tick's new level-1 parents, so a super is never
        # rolled into an era the same tick it is born — it waits one tick. Do not
        # "fix" this into a same-tick re-read.
        era_clusters: list[list[str]] = []
        try:
            l1_ids, l1_embs = _fetch_apex_embeddings(channel, level=1)
            if len(l1_ids) >= ERA_DIGEST_TRIGGER:
                era_clusters = cluster_apex_embeddings(l1_ids, l1_embs)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} era clustering failed (channel={channel}): {exc}"
            )

        # Leaf clusters → level-1 parents; era clusters → level-2 parents.
        rounds = ((leaf_clusters, 1), (era_clusters, 2))
        found = sum(len(clusters) for clusters, _ in rounds)
        if found == 0:
            return 0, 0

        written = 0
        for clusters, level in rounds:
            written += self._write_round(
                channel, clusters, level, db, emb_svc, episodic_svc
            )
        return found, written

    def _write_round(
        self,
        channel: str,
        clusters: list[list[str]],
        level: int,
        db: "DatabaseService",
        emb_svc: "EmbeddingService",
        episodic_svc: "EpisodicService",
    ) -> int:
        """Write one roll-up round's clusters at ``level``. Returns supers written."""
        from services.episodic_service import _fetch_novelty_comparison_set  # noqa: PLC0415

        if not clusters or self._summarization_budget_remaining <= 0:
            return 0

        try:
            prior_embeddings = _fetch_novelty_comparison_set(channel)
        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} _fetch_novelty_comparison_set failed "
                f"(channel={channel}): {exc}"
            )
            prior_embeddings = []

        written = 0
        for cluster_ids in clusters:
            if self._summarization_budget_remaining <= 0:
                break
            if self._write_super_episode(
                channel, cluster_ids, level, db, emb_svc, episodic_svc, prior_embeddings
            ):
                written += 1
                self._summarization_budget_remaining -= 1
        return written

    @staticmethod
    def _write_super_episode(
        channel: str,
        cluster_ids: list[str],
        level: int,
        db: "DatabaseService",
        emb_svc: "EmbeddingService",
        episodic_svc: "EpisodicService",
        prior_embeddings: list[bytes],
    ) -> bool:
        """Encode + store one parent episode for a cluster. Returns True on write."""
        from configs.channels import (  # noqa: PLC0415
            SuperEpisodeConfig,
            _collect_transcript_ids,
            _fetch_transcript_spans,
            _safe_json_load_object,
        )
        from services.episodic_constants import HDBSCAN_MIN_CLUSTER_SIZE  # noqa: PLC0415
        from services.episodic_service import compute_novelty  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        from services.salience_service import compute_salience  # noqa: PLC0415

        try:
            sources = [
                ep for ep in (
                    episodic_svc.get_episode_by_id(eid) for eid in cluster_ids
                )
                if ep
            ]
            if len(sources) < HDBSCAN_MIN_CLUSTER_SIZE:
                return False

            all_t_ids = _collect_transcript_ids(cast(list[object], sources))
            transcript_spans = _fetch_transcript_spans(all_t_ids, db)

            config = SuperEpisodeConfig(channel, cast(list[object], sources), transcript_spans)
            response = MessageProcessor.process("", config)

            if not response:
                logger.warning(
                    f"{LOG_PREFIX} SuperEpisodeEncoder returned empty response "
                    f"for cluster {cluster_ids}"
                )
                return False

            super_ep = _safe_json_load_object(response)
            if not super_ep or not super_ep.get("gist"):
                logger.warning(
                    f"{LOG_PREFIX} SuperEpisodeEncoder returned unparseable/empty "
                    f"gist for cluster {cluster_ids}"
                )
                return False

            super_ep["channel"] = channel
            super_ep["level"] = level
            unique_t_ids = sorted(all_t_ids)
            super_ep["transcript_ids"] = unique_t_ids
            super_ep["transcript_id_start"] = min(unique_t_ids) if unique_t_ids else None
            super_ep["transcript_id_end"] = max(unique_t_ids) if unique_t_ids else None
            super_ep["consolidated_from"] = [ep["id"] for ep in sources]

            gist = cast(str, super_ep["gist"])
            embedding = emb_svc.generate_embedding(gist)
            novelty = compute_novelty(embedding, prior_embeddings) if embedding else 1.0
            super_ep["salience"] = compute_salience(
                valence=float(cast(float, super_ep.get("emotional_valence") or 0.0)),
                arousal=float(cast(float, super_ep.get("emotional_arousal") or 0.0)),
                has_open_loop=bool(super_ep.get("has_open_loop", False)),
                novelty=novelty,
            )
            super_ep.pop("has_open_loop", None)

            new_id = episodic_svc.store_episode(super_ep, embedding=embedding)
            for src_id in cluster_ids:
                episodic_svc.set_consolidated_into(src_id, new_id)

            logger.info(
                f"{LOG_PREFIX} level-{level} episode {new_id} created from cluster "
                f"{cluster_ids} (channel={channel})"
            )
            return True

        except Exception as exc:
            logger.warning(
                f"{LOG_PREFIX} level-{level} consolidation failed for cluster "
                f"{cluster_ids} (channel={channel}): {exc}"
            )
            return False

    def _step_fact_extraction(self) -> str:
        """Step 2 — route hard facts from new episodes into data_graph."""
        from configs.channels import FactExtractionConfig, parse_fact_ops  # noqa: PLC0415
        from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
        from services.database_service import get_shared_db_service  # noqa: PLC0415
        from services.episodic_service import EpisodicService  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        episodic_svc = EpisodicService(get_shared_db_service())
        backlog = episodic_svc.fetch_fact_extraction_backlog(_FACT_EXTRACTION_CALL_BUDGET)
        if not backlog:
            return "no backlog"

        dg = get_data_graph_service()
        counters = {
            "episodes": 0, "add": 0, "update": 0, "delete": 0,
            "noop": 0, "unparseable": 0, "failed": 0,
        }

        for episode in backlog:
            try:
                self._extract_facts_for_episode(
                    episode, episodic_svc, dg, counters,
                    FactExtractionConfig, parse_fact_ops, MessageProcessor,
                )
            except Exception as exc:
                logger.warning(
                    f"{LOG_PREFIX} fact_extraction failed for episode "
                    f"{episode.get('id')}: {exc}"
                )

        return (
            f"episodes={counters['episodes']} add={counters['add']} "
            f"update={counters['update']} delete={counters['delete']} "
            f"noop={counters['noop']} unparseable={counters['unparseable']} "
            f"failed={counters['failed']}"
        )

    def _extract_facts_for_episode(
        self,
        episode: dict[str, object],
        episodic_svc: "EpisodicService",
        dg: "DataGraphService",
        counters: dict[str, int],
        config_cls: object,
        parse_ops: object,
        processor_cls: object,
    ) -> None:
        """Run the constrained-op pipeline for a single episode and stamp it."""
        gist = cast(str, episode.get("gist") or "")
        neighbours = dg.recall(gist, limit=_FACT_NEIGHBOUR_LIMIT) if gist else []

        from collections.abc import Callable as _Callable  # noqa: PLC0415
        config = cast(_Callable[..., object], config_cls)(gist, neighbours)
        response = cast(_Callable[[str, object], str], getattr(processor_cls, "process"))("", config)

        try:
            ops = cast(list[dict[str, object]], cast(_Callable[..., object], parse_ops)(response))
        except ValueError as exc:
            counters["unparseable"] += 1
            logger.warning(
                f"{LOG_PREFIX} fact_extraction unparseable output for episode "
                f"{episode.get('id')} — NOOP: {exc}"
            )
            ops = []

        # Provenance is channel-tagged so a fact's origin (user vs dmn vs a
        # specific external agent) is recoverable from data_graph.source. dmn and
        # external-agent facts are wanted, so there is no channel gate here — the
        # backlog feeds every episode-producing channel.
        source = _fact_source_for(cast(Optional[str], episode.get("channel")))
        for op in ops:
            self._apply_fact_op(op, dg, counters, source)

        episodic_svc.set_facts_extracted_at(cast(str, episode["id"]))
        counters["episodes"] += 1

    def _apply_fact_op(self, op: dict[str, object], dg: "DataGraphService", counters: dict[str, int], source: str) -> None:
        """Apply one validated constrained op to data_graph and count it."""
        from configs.channels.fact_extraction import OP_DELETE  # noqa: PLC0415

        verb = op["op"]
        try:
            if verb == OP_DELETE:
                dg.invalidate(cast(str, op["kind"]), cast(str, op["key"]))
                counters["delete"] += 1
                return
            result = dg.upsert_fact(cast(str, op["key"]), cast(str, op["value"]), source=source)
            if result is None:
                counters["noop"] += 1
                return
            counters[_FACT_STATUS_COUNTER.get(cast(str, result.get("status")), "add")] += 1
        except Exception as exc:
            counters["failed"] += 1
            logger.warning(
                f"{LOG_PREFIX} fact_extraction op {verb} key='{op.get('key')}' "
                f"failed: {exc}"
            )

    def _step_decay(self) -> str:
        """Step 3 — run the unified decay cycle."""
        if self._decay_engine is None:
            from services.decay_engine_service import DecayEngineService
            self._decay_engine = DecayEngineService()
        self._decay_engine.run_once()
        return "ok"

    def _step_pattern_match(self) -> str:
        """Step 4 — single-pass LLM pattern matcher over a transcript-id window."""
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

        # 2. Read latest transcript id.
        # The cursor must count only rows the pattern LOAD window (pattern.py)
        # actually reads — user-behaviour channels, no compaction rows. Counting
        # background-loop rows (dmn writes many) would advance the delta past the
        # _MIN_DELTA trigger and fire spurious pattern passes the load discards.
        from services.transcript_service import Transcript  # noqa: PLC0415
        latest = Transcript.latest_id(["user"], exclude_roles=("compaction",)) or 0

        delta = latest - cursor
        if delta < _MIN_DELTA:
            logger.info(
                f"{LOG_PREFIX} pattern_match_skip cursor={cursor} "
                f"latest={latest} delta={delta}"
            )
            return f"skip cursor={cursor} latest={latest} delta={delta}"

        # 3. Fire the pattern pass via the canonical entry point. The skill-
        # personalisation sync (PatternSkillSyncHook) runs inside the turn's
        # post_turn_hooks, keyed off the patterns it touched — both that set and
        # the confidence-decay sweep are derived from the turn's durable rows, so
        # nothing needs to be inspected on the processor after it returns.
        from configs.channels import PatternConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        MessageProcessor.process("", PatternConfig(cursor, latest))

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
        """Step 5 — refresh the user synopsis (short + long)."""
        from configs.channels import UserSummaryConfig, _should_synthesise  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        if not _should_synthesise():
            logger.info(f"{LOG_PREFIX} No new traits since last synthesis; skipping")
            return "no new traits/patterns; skipped"

        config = UserSummaryConfig()
        MessageProcessor.process("", config)
        return "ok"

    def _step_dmn(self) -> str:
        """Step 6 — background DMN reflection via DMNMessageProcessor."""
        synthesis = self._load_user_synthesis()
        if not synthesis:
            logger.info(f"{LOG_PREFIX} Skipping DMN — no user synthesis available")
            return "skipped: no user synthesis"

        from configs.channels import DmnConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        MessageProcessor.process("", DmnConfig())
        return "ok"

    def _step_capability_sync(self) -> str:
        """Step 7 — IMAP / CalDAV / CardDAV server sync."""
        from capabilities import load_capabilities

        synced = []
        for cap in load_capabilities().values():
            if cap.is_connected():
                cap.monitor()
                synced.append(cap.get_id())
        return f"synced: {', '.join(synced)}" if synced else "no connected capabilities"

    def _step_geo_patterns(self) -> str:
        """Step 8 — single-pass LLM geo-spatial pattern extractor."""
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

            # Same allowlist as the geo-pattern window (geo_pattern.py): only
            # user geo-activity channels advance the cursor, so a located row on
            # a muted channel can never fire the geo pass.
            from services.transcript_service import Transcript  # noqa: PLC0415
            latest = Transcript.latest_id(["user"], require_location=True, exclude_roles=("compaction",)) or 0
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

        # Fire the geo pass via the canonical entry point.
        from configs.channels import GeoConfig  # noqa: PLC0415
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        MessageProcessor.process("", GeoConfig(cursor, latest))

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
        """Check whether a user_summary row exists in data_graph."""
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
        """Hydrate last_fired_at from the durable dual-write store."""
        return _LAST_FIRED_TIMESTAMP.load()

    def _persist_last_fired(self, when: datetime) -> None:
        """Write last_fired_at to MemoryStore + data_graph."""
        _LAST_FIRED_TIMESTAMP.persist(when)


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


def subconscious_worker() -> None:
    """WorkerManager entry point. Tick loop with a stable cadence."""
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
