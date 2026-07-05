"""One ``data_graph`` row — the active-record model for the structured
user-context lanes the prompt spine assembles (§4.1; §3.11 clusters A/B/C).

Pure CRUD (Rule-3 depth): holds no ``mp``, imports no service, never emits WS,
never reaches upstream. It only stores its fields, projects itself (inherited
``to_dict``/``to_json``), and runs its own table's SQL on the connection bound
onto the :class:`Model` base. This model is the SOLE home of ``data_graph`` SQL
(I6); the ``DataGraphService`` reads/writes exclusively through it.

Scope is the spine's structured user-context slice ONLY — the three lanes the
prompt builder needs at assembly time and the post-turn writes back into them:

* ``kind='system'``            — the user_summary rows (:meth:`system`).
* ``kind='user_specific'``     — user traits (:meth:`traits`).
* ``kind='behavioral_pattern'``— behavioural patterns (:meth:`patterns` for the
  user-summary channel's recency read, :meth:`top_patterns` for the
  pattern channel's confidence-ranked read).

The write side is :meth:`store` (exact-``(kind, key)`` upsert: insert / reinforce
/ temporal-supersede, ported from ``DataGraphService.store`` +
``_store_existing_row`` + ``_reinforce_row`` + ``_apply_temporal_supersession``)
and :meth:`decay` (the behavioural-pattern confidence sweep, ported from
``PatternDecayHook``). Episodic recall, FTS/vec KNN, the concept-LUT
canonicalization, graph edges, embedding writes and the off-spine decay cycle
are deliberately NOT here — they stay in their own layers.

Live-fact filters mirror today's code verbatim: the lane reads mirror
``DataGraphService.fetch`` (``active = 1 AND deleted_at IS NULL``; note ``fetch``
does NOT filter ``valid_to`` — only the out-of-scope ``recall`` does, so neither
does a lane here); the exact-key store lookup mirrors the service's
``_SELECT_ACTIVE_BY_KIND_KEY_SQL`` (``active = 1`` only, no ``deleted_at``).
"""

from __future__ import annotations

import math
from typing import ClassVar, Self

from models.model import Model
from models.query import Query
from services.time_utils import utc_now


class DataGraph(Model):
    """One ``data_graph`` row: field storage + CRUD + the structured
    user-context lane scopes and the upsert/reinforce/supersede write path."""

    __table__: ClassVar[str] = "data_graph"
    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "kind", "key", "value", "storage_strength", "retrieval_weight",
        "salience_score", "evidence_count", "first_seen_at", "last_confirmed_at",
        "last_accessed_at", "source", "deleted_at", "active", "search_queries",
        "valid_from", "valid_to",
    )

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    kind: str
    key: str
    value: str | None
    storage_strength: float
    retrieval_weight: float
    salience_score: float
    evidence_count: int
    first_seen_at: str
    last_confirmed_at: str
    last_accessed_at: str | None
    source: str | None
    deleted_at: str | None
    active: int
    search_queries: str | None
    valid_from: str | None
    valid_to: str | None

    # The kinds whose exact-key value change reconciles by temporal supersession
    # (``DataGraphService._KIND_POLICY`` contradiction 'cosine_supersede' /
    # 'lut_canonicalize'); with the embedding path out of scope both collapse to
    # the exact-key supersede fallback the service takes on a LUT/vec miss.
    # 'behavioral_pattern' (contradiction ``None``) instead inserts a fresh row.
    _SUPERSEDING_KINDS: ClassVar[frozenset[str]] = frozenset({"system", "user_specific"})
    # Retrieval-weight haircut on the demoted row (``_SUPERSEDE_RW_FACTOR``).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5
    # Evidence-diminishing reinforcement boost numerator (``_reinforce_row``).
    _REINFORCE_BOOST: ClassVar[float] = 0.05

    # ── lane reads (§3.11 A/B) — live scopes, late-binding (Critical 3) ──

    @classmethod
    def live(cls, kind: str) -> Query[Self]:
        """The live-fact scope for one kind — ``active = 1 AND deleted_at IS
        NULL``, mirroring ``DataGraphService.fetch``'s default filters. The
        one shared live predicate all three lanes build on (Essential 8)."""
        return cls.filter("kind = ?", kind).filter("active = 1").filter("deleted_at IS NULL")

    @classmethod
    def system(cls) -> Query[Self]:
        """Live ``system`` rows, ``retrieval_weight DESC`` — the user_summary
        lane (``UserConfig.get_user_definition`` / ``EAMPConfig.get_system_prompt``
        read ``key='user_summary'`` from this set)."""
        return cls.live("system").order_by("retrieval_weight DESC")

    @classmethod
    def traits(cls) -> Query[Self]:
        """Live ``user_specific`` traits, ``retrieval_weight DESC`` — the
        user-summary channel's facts section (``UserSummaryConfig`` section 1)."""
        return cls.live("user_specific").order_by("retrieval_weight DESC")

    @classmethod
    def patterns(cls) -> Query[Self]:
        """Live ``behavioral_pattern`` rows, ``last_confirmed_at DESC`` — the
        user-summary channel's active-patterns section (``UserSummaryConfig``
        section 2)."""
        return cls.live("behavioral_pattern").order_by("last_confirmed_at DESC")

    @classmethod
    def top_patterns(cls) -> Query[Self]:
        """Live ``behavioral_pattern`` rows with a valid JSON confidence,
        ranked ``confidence DESC`` — the pattern channel's existing-patterns
        block (``_pattern_existing_patterns_block``). The JSON guards mirror the
        ported SQL verbatim so a malformed row can never break the ORDER BY."""
        return (
            cls.live("behavioral_pattern")
            .filter("json_valid(value) = 1")
            .filter("json_extract(value, '$.confidence') IS NOT NULL")
            .order_by("CAST(json_extract(value, '$.confidence') AS REAL) DESC")
        )

    # ── write path (§3.11 C) ────────────────────────────────────────────

    @classmethod
    def active_by_key(cls, kind: str, key: str) -> Self | None:
        """The single active row for an exact ``(kind, key)`` — the store
        lookup, mirroring ``_SELECT_ACTIVE_BY_KIND_KEY_SQL`` (``active = 1``
        only; deliberately NOT ``deleted_at``-filtered, matching the service)."""
        return cls.filter("kind = ?", kind).filter("key = ?", key).filter("active = 1").first()

    @classmethod
    def store(cls, kind: str, key: str, value: str, source: str | None = None) -> Self:
        """Exact-``(kind, key)`` upsert, ported from ``DataGraphService.store``.

        No active row → insert new. Same value (case-folded, trimmed) →
        :meth:`reinforce`. Contradicting value → temporal supersede for a
        superseding kind (demote old, insert successor), else a fresh insert
        (``contradiction = None`` kinds). Returns the resulting live row."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(kind, key)
        if existing is None:
            return cls(
                kind=kind, key=key, value=value, source=source,
                first_seen_at=now, last_confirmed_at=now,
            ).save()
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce()
        if kind in cls._SUPERSEDING_KINDS:
            existing.demote()
            return cls(
                kind=kind, key=key, value=value, source=source,
                first_seen_at=now, last_confirmed_at=now, valid_from=now,
            ).save()
        return cls(
            kind=kind, key=key, value=value, source=source,
            first_seen_at=now, last_confirmed_at=now,
        ).save()

    def reinforce(self) -> Self:
        """Re-confirm this row on a repeated identical fact (``_reinforce_row``):
        bump ``evidence_count``, apply the evidence-diminishing storage boost,
        reset ``retrieval_weight`` to 1.0, and stamp the confirm/access times."""
        now = utc_now().isoformat()
        self.evidence_count += 1
        boost = self._REINFORCE_BOOST / math.log2(self.evidence_count + 1)
        self.storage_strength = min(1.0, self.storage_strength + boost)
        self.retrieval_weight = 1.0
        self.last_confirmed_at = now
        self.last_accessed_at = now
        return self.save()

    def demote(self) -> Self:
        """Close this row out on supersession (``_apply_temporal_supersession``):
        mark ``active = 0``, halve ``retrieval_weight``, and stamp ``valid_to``
        so it drops out of every live lane and enters the fast-decay window."""
        self.active = 0
        self.retrieval_weight *= self._SUPERSEDE_RW_FACTOR
        self.valid_to = utc_now().isoformat()
        return self.save()

    @classmethod
    def decay(cls, touched_keys: set[str]) -> None:
        """The behavioural-pattern confidence sweep, ported from
        ``PatternDecayHook``: decrement ``$.confidence`` by 0.005 on every live
        pattern row NOT touched this turn, then soft-delete (``active = 0``) any
        whose confidence has fallen to zero. ``touched_keys`` are the pattern
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
            "WHERE kind = 'behavioral_pattern' AND active = 1 AND deleted_at IS NULL "
            f"AND json_extract(value, '$.confidence') IS NOT NULL {touched_filter}",
            params,
        )
        connection.execute(
            "UPDATE data_graph SET active = 0 WHERE kind = 'behavioral_pattern' "
            "AND active = 1 AND CAST(json_extract(value, '$.confidence') AS REAL) <= 0.0"
        )
