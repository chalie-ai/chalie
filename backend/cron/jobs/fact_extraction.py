"""Idle-gated cron job: drain the fact_extraction backlog.

Ported from ``services.subconscious_worker.SubconsciousWorker._step_fact_extraction``
and its two private helpers. Self-contained — every constant, helper, and import
the methods use is carried into this file so the original module can be deleted
without breaking the cron job.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, cast

from cron.base import IdleGatedJob
from models.episode import Episode

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

logger = logging.getLogger(__name__)

# ── Carried from services/subconscious_worker.py (VERBATIM) ──────────────────

LOG_PREFIX = "[SUBCONSCIOUS]"

# Fact-extraction step budget. The backlog of
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


class FactExtractionJob(IdleGatedJob):
    """Idle-gated cron job: drain episodes with NULL facts_extracted_at."""

    name = "fact_extraction"

    def _run(self) -> str:
        """Step 2 — route hard facts from new episodes into data_graph."""
        from configs.channels import FactExtractionConfig, parse_fact_ops  # noqa: PLC0415
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        backlog = Episode.fact_extraction_backlog(_FACT_EXTRACTION_CALL_BUDGET)
        if not backlog:
            return "no backlog"

        counters = {
            "episodes": 0, "add": 0, "update": 0, "delete": 0,
            "noop": 0, "unparseable": 0, "failed": 0,
        }

        for episode in backlog:
            try:
                self._extract_facts_for_episode(
                    episode, counters,
                    FactExtractionConfig, parse_fact_ops, MessageProcessor,
                )
            except Exception as exc:
                logger.warning(
                    f"{LOG_PREFIX} fact_extraction failed for episode "
                    f"{episode.id}: {exc}"
                )

        return (
            f"episodes={counters['episodes']} add={counters['add']} "
            f"update={counters['update']} delete={counters['delete']} "
            f"noop={counters['noop']} unparseable={counters['unparseable']} "
            f"failed={counters['failed']}"
        )

    def _extract_facts_for_episode(
        self,
        episode: Episode,
        counters: dict[str, int],
        config_cls: object,
        parse_ops: object,
        processor_cls: object,
    ) -> None:
        """Run the constrained-op pipeline for a single episode and stamp it."""
        from models.fact import FactRow  # noqa: PLC0415

        gist = episode.gist or ""
        neighbours: list[object] = (
            [{"kind": r.kind, "key": r.key, "value": r.value}
             for r in FactRow.search(gist, _FACT_NEIGHBOUR_LIMIT)]
            if gist else []
        )

        from collections.abc import Callable as _Callable  # noqa: PLC0415
        config = cast(_Callable[..., object], config_cls)(gist, neighbours)
        mp = cast(
            _Callable[[object], "MessageProcessor"], getattr(processor_cls, "process")
        )(config)
        response = mp.result()

        try:
            ops = cast(list[dict[str, object]], cast(_Callable[..., object], parse_ops)(response))
        except ValueError as exc:
            counters["unparseable"] += 1
            logger.warning(
                f"{LOG_PREFIX} fact_extraction unparseable output for episode "
                f"{episode.id} — NOOP: {exc}"
            )
            ops = []

        # Provenance is channel-tagged so a fact's origin (user vs dmn vs a
        # specific external agent) is recoverable from data_graph.source. dmn and
        # external-agent facts are wanted, so there is no channel gate here — the
        # backlog feeds every episode-producing channel.
        source = _fact_source_for(episode.channel)
        for op in ops:
            self._apply_fact_op(op, counters, source)

        Episode.set_facts_extracted_at(cast(str, episode.id))
        counters["episodes"] += 1

    def _apply_fact_op(self, op: dict[str, object], counters: dict[str, int], source: str) -> None:
        """Apply one validated constrained op to data_graph and count it.

        The pipeline writes user_specific only (fact_extraction stamps that kind
        on every op, DELETE included), so both branches route through the FACTS
        vertical: DELETE invalidates the key, ADD/UPDATE upsert it."""
        from configs.channels.fact_extraction import OP_DELETE  # noqa: PLC0415
        from models.fact import FactRow  # noqa: PLC0415
        from services.fact_service import FactService  # noqa: PLC0415

        verb = op["op"]
        try:
            if verb == OP_DELETE:
                FactRow.forget(cast(str, op["key"]))
                counters["delete"] += 1
                return
            env = FactService().store(
                cast(str, op["key"]), cast(str, op["value"]), source=source, canonicalize=False
            )
            counters[_FACT_STATUS_COUNTER.get(cast(str, env["status"]), "add")] += 1
        except Exception as exc:
            counters["failed"] += 1
            logger.warning(
                f"{LOG_PREFIX} fact_extraction op {verb} key='{op.get('key')}' "
                f"failed: {exc}"
            )
