"""One ``data_graph`` row of kind ``behavioral_pattern`` — the active-record
model for the user's repeating-behaviour lane.

Same table, columns and SQL semantics as :class:`models.data_graph.DataGraph`;
this model is the SOLE home of the *behavioural-pattern* slice of that table so
the generic ``DataGraph`` carries zero pattern-specific literals. Pure CRUD
(Rule-3 depth): holds no ``mp``, imports no service, never reaches upstream. It
stores its fields, projects itself (inherited ``to_dict``/``to_json``) and runs
its own table's SQL on the connection bound onto the :class:`Model` base.

The ``behavioral_pattern`` kind literal lives here (:attr:`KIND`) as the single
source of truth; every other layer imports it.
"""

from __future__ import annotations

import json
import math
from typing import ClassVar, Self, cast

from models.data_graph import DataGraphRow
from models.query import Query
from services.time_utils import utc_now
from utils.data_utils import parse_json_column


class BehavioralPattern(DataGraphRow):
    """One ``data_graph`` behavioural-pattern row: the shared shell from
    :class:`~models.data_graph.DataGraphRow` (table binding, columns, reinforce
    path, kind-bound reads and the FTS/vec write-sync) plus the pattern-specific
    upsert/reinforce write path, the confidence decay sweep and the value
    renderer."""

    #: The ``data_graph.kind`` this model owns — the one source of truth for the
    #: literal every other layer imports. Binds the inherited kind-bound reads
    #: (:meth:`~models.data_graph.DataGraphRow.live` / ``search``) to this lane.
    KIND: ClassVar[str] = "behavioral_pattern"

    # Write-path constants, ported verbatim from the legacy ``save_pattern`` SQL.
    _NEW_CONFIDENCE: ClassVar[float] = 7.0          # a first sighting's confidence
    _REINFORCE_STEP: ClassVar[float] = 7.0          # confidence added on a repeat
    _MAX_CONFIDENCE: ClassVar[float] = 10.0         # confidence ceiling
    _STRENGTH_BOOST: ClassVar[float] = 0.05         # evidence-diminishing boost numerator
    _DEFAULT_STRENGTH: ClassVar[float] = 0.5        # storage_strength fallback (schema default)
    _DEFAULT_EVIDENCE: ClassVar[int] = 1            # evidence_count fallback (schema default)
    _DECAY_STEP: ClassVar[float] = 0.005            # per-turn confidence decrement

    # ── lane reads — live scopes, late-binding (zero I/O until terminal) ──

    @classmethod
    def recent(cls) -> Query[Self]:
        """Live patterns, ``last_confirmed_at DESC`` — the user-summary channel's
        active-patterns read (ported from ``DataGraph.patterns``). Builds on the
        inherited kind-bound :meth:`~models.data_graph.DataGraphRow.live`."""
        return cls.live().order_by("last_confirmed_at DESC")

    @classmethod
    def gather_candidates(
        cls, query: str, query_embedding: list[float] | None, k: int
    ) -> dict[int, dict[str, object]]:
        """Vec-only candidate gather (ruling 1): this kind carries no FTS
        posting and no doc2query variants —
        ``services.search_expander_service.should_fts_index`` excludes
        ``behavioral_pattern`` from the entire async write-side enqueue, not
        just its FTS half (commit 5a343e7b), so only the key/value vec KNN
        lane (:meth:`~models.data_graph.DataGraphRow.vec_candidates`) ever has
        signal for this kind. Overrides the base's vec+variant+FTS composition."""
        return cls._hydrate_candidates(cls.vec_candidates(query_embedding, k))

    # ── write path (ported from abilities/save_pattern.py) ───────────────────

    @classmethod
    def upsert(cls, validated: dict[str, object], source: str) -> tuple[int, float, bool]:
        """Insert a new pattern or reinforce the existing one for
        ``validated['name']``. Returns ``(row_id, confidence, reinforced)``.

        The lookup mirrors the legacy ``_upsert_pattern`` SELECT (live rows for
        the exact key, newest first). ``source`` records which pass produced the
        row (its config channel)."""
        now = utc_now().isoformat()
        existing = cls.live().filter("key = ?", validated["name"]).order_by("id DESC").first()
        if existing is not None:
            row_id, confidence = existing._reinforce(validated, now, source)
            return row_id, confidence, True
        row_id, confidence = cls._insert(validated, now, source)
        return row_id, confidence, False

    @classmethod
    def _insert(cls, validated: dict[str, object], now: str, source: str) -> tuple[int, float]:
        """Insert a fresh pattern at :attr:`_NEW_CONFIDENCE`, leaving
        storage_strength/retrieval_weight/evidence_count to their schema
        defaults (ported from ``_insert_new_pattern``)."""
        row = cls(
            kind=cls.KIND,
            key=validated["name"],
            value=json.dumps(cls._build_value(
                validated, now, cls._NEW_CONFIDENCE, list(cast("list[object]", validated["evidence"])),
            )),
            first_seen_at=now,
            last_confirmed_at=now,
            source=source,
            active=1,
        ).save()
        return cast("int", row.id), cls._NEW_CONFIDENCE

    def _reinforce(self, validated: dict[str, object], now: str, source: str) -> tuple[int, float]:
        """Re-confirm this pattern on a repeated sighting (ported from
        ``_update_existing_pattern``): raise confidence toward the ceiling, apply
        the evidence-diminishing storage boost, reset retrieval_weight and merge
        the evidence ids."""
        prev = cast("dict[str, object]", parse_json_column(self.value))
        prev_conf = float(cast("float", prev.get("confidence") or 0.0))
        new_conf = min(self._MAX_CONFIDENCE, prev_conf + self._REINFORCE_STEP)
        merged_evidence = list(dict.fromkeys([
            *cast("list[object]", prev.get("evidence_transcript_ids") or []),
            *cast("list[object]", validated["evidence"]),
        ]))
        old_strength = float(self.storage_strength) if self.storage_strength is not None else self._DEFAULT_STRENGTH
        new_evidence = (int(self.evidence_count) if self.evidence_count is not None else self._DEFAULT_EVIDENCE) + 1
        self.value = json.dumps(self._build_value(validated, now, new_conf, merged_evidence))
        self.last_confirmed_at = now
        self.last_accessed_at = now
        self.source = source
        self.evidence_count = new_evidence
        self.storage_strength = min(1.0, old_strength + self._STRENGTH_BOOST / math.log2(new_evidence + 1))
        self.retrieval_weight = 1.0
        self.save()
        return cast("int", self.id), new_conf

    @classmethod
    def _build_value(
        cls, validated: dict[str, object], now: str, confidence: float, evidence_ids: list[object]
    ) -> dict[str, object]:
        """The stored ``value`` JSON blob — key order held byte-identical to the
        legacy writers so a round-trip through insert/reinforce is stable."""
        return {
            "name": validated["name"],
            "frequency": validated["frequency"],
            "time_anchor": validated["time_anchor"],
            "summary": validated["summary"],
            "confidence": confidence,
            "last_seen_at": now,
            "evidence_transcript_ids": evidence_ids,
        }

    # ── decay (ported verbatim from DataGraph.decay) ─────────────────────────

    @classmethod
    def decay(cls, touched_keys: set[str]) -> None:
        """The confidence sweep: decrement ``$.confidence`` by 0.005 on every
        live pattern row NOT touched this turn, then soft-delete (``active = 0``)
        any whose confidence has fallen to zero. ``touched_keys`` are the pattern
        names written this turn and so exempt from the decrement."""
        connection = cls._bound_connection()
        touched_filter = ""
        params: list[object] = []
        if touched_keys:
            placeholders = ",".join("?" * len(touched_keys))
            touched_filter = f"AND key NOT IN ({placeholders})"
            params = list(touched_keys)
        connection.execute(
            "UPDATE data_graph SET value = json_set(value, '$.confidence', "
            "MAX(0.0, CAST(json_extract(value, '$.confidence') AS REAL) - 0.005)) "
            f"WHERE kind = '{cls.KIND}' AND active = 1 AND deleted_at IS NULL "
            f"AND json_extract(value, '$.confidence') IS NOT NULL {touched_filter}",
            params,
        )
        connection.execute(
            f"UPDATE data_graph SET active = 0 WHERE kind = '{cls.KIND}' "
            "AND active = 1 AND CAST(json_extract(value, '$.confidence') AS REAL) <= 0.0"
        )

    # ── the shared value parser + renderer ───────────────────────────────────

    @classmethod
    def parse(cls, value: str | None) -> dict[str, object] | None:
        """One pattern row's JSON ``value`` parsed to a dict, or ``None`` on a
        missing/malformed blob — the ``json_valid(value) = 1`` guard the legacy
        raw SQL applied (ported from ``PromptService._parse_pattern``)."""
        if not value:
            return None
        try:
            content = json.loads(value)
        except (TypeError, ValueError):
            return None
        return content if isinstance(content, dict) else None

    @classmethod
    def render_line(cls, content: dict[str, object], *, include_last_seen: bool = False) -> str:
        """Render one parsed pattern as ``name (freq @ anchor): summary
        [confidence=N]``, optionally suffixed ``, last <YYYY-MM-DD>`` when
        ``include_last_seen`` — the single implementation replacing the near-
        duplicate ``PromptService._format_pattern_line`` and
        ``memory_retrieval._render_behavioral_pattern`` formatters."""
        anchor = cast("str", content.get("time_anchor") or "")
        anchor_part = f" @ {anchor}" if anchor else ""
        base = (
            f"{content.get('name', 'unknown')} ({content.get('frequency', '?')}{anchor_part}): "
            f"{content.get('summary', '')} [confidence={content.get('confidence', 0)}"
        )
        if include_last_seen:
            last_seen = cast("str", content.get("last_seen_at") or "")[:10] or "?"
            return f"{base}, last {last_seen}]"
        return f"{base}]"

    @classmethod
    def render(cls, value: str | None, *, include_last_seen: bool = False) -> str:
        """Render a raw JSON ``value`` string, passing it through verbatim on a
        parse failure — the memory-recall renderer (ported from
        ``memory_retrieval._render_behavioral_pattern``)."""
        content = cls.parse(value)
        if content is None:
            return value if isinstance(value, str) else ""
        return cls.render_line(content, include_last_seen=include_last_seen)
