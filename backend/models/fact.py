"""The ``user_specific`` (FACTS) vertical of ``data_graph`` — the exact-key
upsert/reinforce/temporal-supersede lane behind the memory tool, the worker
fact pipeline and the prompt's traits section. Subclasses the shared
:class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec write-sync,
reinforce, kind-scoped live/search); adds only the FACTS kind, its exact-key
store/forget/supersede and its power-law decay. Sole home of this lane's SQL (I6)."""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from models.query import Query
from services.time_utils import utc_now


class FactRow(DataGraphRow):
    KIND: ClassVar[str] = "user_specific"

    # Retrieval-weight haircut on the demoted (superseded) row (_SUPERSEDE_RW_FACTOR).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5
    # Power-law live-fact decay policy for user_specific (legacy _KIND_POLICY):
    # ttl 30d gates that decay runs at all; d_base/salience_floor set the curve.
    _DECAY_D_BASE: ClassVar[float] = 0.5
    _DECAY_SALIENCE_FLOOR: ClassVar[float] = 0.2
    _DECAY_RW_EPSILON: ClassVar[float] = 0.0001

    @classmethod
    def active_by_key(cls, key: str) -> Self | None:
        """The single active row for this kind's exact ``key`` — the store lookup
        (mirrors ``_SELECT_ACTIVE_BY_KIND_KEY_SQL``: ``active = 1`` only, NOT
        ``deleted_at``-filtered)."""
        return (cls.filter("kind", cls.KIND).filter("key", key)
                   .filter("active", 1).first())

    @classmethod
    def active_values(cls, key: str) -> list[str]:
        """Every currently-active value for this kind's exact ``key`` — the
        coexist key's live value set (mirrors ``_fetch_coexist_values``:
        ``active = 1 AND deleted_at IS NULL``)."""
        return [v for row in cls.live().filter("key", key).get()
                if (v := row.value) is not None]

    @classmethod
    def traits(cls) -> Query[Self]:
        """Live ``user_specific`` traits, ``retrieval_weight DESC`` — the
        user-summary channel's facts section and the prompt's traits block."""
        return cls.live().order_by("retrieval_weight DESC")

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key upsert. Returns ``(row, status, old_value)`` where status is
        ``created`` | ``reinforced`` | ``superseded`` (user_specific always
        temporally supersedes on a contradicting value)."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            row = cls(kind=cls.KIND, key=key, value=value, source=source,
                      first_seen_at=now, last_confirmed_at=now).save()
            return row, "created", None
        old_value = existing.value
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce(), "reinforced", old_value
        existing.demote()
        row = cls(kind=cls.KIND, key=key, value=value, source=source,
                  first_seen_at=now, last_confirmed_at=now, valid_from=now).save()
        return row, "superseded", old_value

    @classmethod
    def store_coexist(cls, key: str, value: str,
                      source: str | None = None) -> tuple[Self, str, list[str] | None]:
        """Coexist multi-value upsert at the canonical ``key``: each distinct
        value is its own live row. An exact-value dup already live (case-folded/
        trimmed) reinforces that row with NO new insert (``reinforced``,
        ``all_values`` None). A new value inserts a fresh row — ``appended`` when
        a prior live row exists under the key, else ``created`` — and returns
        every currently-active value for the key."""
        now = utc_now().isoformat()
        norm = (value or "").lower().strip()
        live_rows = cls.live().filter("key", key).get()
        for existing in live_rows:
            if (existing.value or "").lower().strip() == norm:
                return existing.reinforce(), "reinforced", None
        row = cls(kind=cls.KIND, key=key, value=value, source=source,
                  first_seen_at=now, last_confirmed_at=now).save()
        return row, ("appended" if live_rows else "created"), cls.active_values(key)

    @classmethod
    def supersede_value(cls, key: str, old_value: str, value: str,
                        source: str | None = None) -> tuple[Self, str, str] | None:
        """Value-addressed demote+insert on the live rows of ``key``.

        Finds a target row among the live set (``active = 1 AND deleted_at IS
        NULL``) of the given key via a three-rung ladder: (a) exact case-folded
        equality of the row's value with ``old_value``; (b) exactly one live row
        whose case-folded value contains the case-folded ``old_value`` as a
        substring — unique-substring; (c) absent or ambiguous → ``None`` so the
        caller may fall back.

        If the matched row's value is an exact (case-folded) match for the *new*
        ``value`` the row is reinforced (the same path :meth:`store_coexist` takes
        on an exact duplicate) and ``(row, "reinforced", matched_old_value)`` is
        returned. Otherwise the matched row is demoted via :meth:`demote` (no
        side-effect reimplementation) and the new value is inserted through
        :meth:`store_coexist` so dedup+insert stay single-sourced;
        ``(new_row, "superseded", matched_old_value)`` is returned."""
        norm_old = (old_value or "").lower().strip()
        norm_new = (value or "").lower().strip()
        live_rows = cls.live().filter("key", key).get()
        # (a) exact case-folded equality
        target: Self | None = None
        matched_old: str | None = None
        for row in live_rows:
            if (row.value or "").lower().strip() == norm_old:
                target = row
                matched_old = row.value
                break
        # (b) unique-substring
        if target is None:
            matches: list[Self] = []
            for row in live_rows:
                if norm_old and norm_old in (row.value or "").lower().strip():
                    matches.append(row)
            if len(matches) == 1:
                target = matches[0]
                matched_old = target.value
        if target is None:
            return None
        # new value equals matched row → reinforce
        if norm_new == (target.value or "").lower().strip():
            return target.reinforce(), "reinforced", matched_old
        # demote the old row, then insert new via store_coexist
        target.demote()
        new_row, _status, _ = cls.store_coexist(key, value, source=source)
        return new_row, "superseded", matched_old

    @classmethod
    def store_immutable(cls, key: str, value: str,
                        source: str | None = None) -> tuple[Self, str, str | None]:
        """Immutable single-value upsert at the canonical ``key``. No live row →
        insert (``created``). Same value (case-folded/trimmed) → reinforce
        (``reinforced``). A contradicting value is REJECTED: status ``conflict``,
        NO write (no insert, no demote); the untouched existing row and its value
        (as ``old_value``) are returned so the caller renders "kept X, rejected Y"."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            row = cls(kind=cls.KIND, key=key, value=value, source=source,
                      first_seen_at=now, last_confirmed_at=now).save()
            return row, "created", None
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce(), "reinforced", None
        return existing, "conflict", existing.value

    def demote(self) -> Self:
        """Close this row out on supersession: ``active = 0``, halve
        ``retrieval_weight``, stamp ``valid_to`` so it drops out of live lanes and
        enters the fast-decay window."""
        self.active = 0
        self.retrieval_weight *= self._SUPERSEDE_RW_FACTOR
        self.valid_to = utc_now().isoformat()
        return self.save()

    @classmethod
    def forget(cls, key: str, value: str | None = None) -> int:
        """Bi-temporally invalidate every live row for this kind's exact ``key``
        (optionally only the row whose value matches, case-folded/trimmed):
        ``active = 0``, ``valid_to = now``, ``retrieval_weight`` halved. Matches
        VERBATIM (no canonicalisation — the LLM chose the key from shown facts).
        Returns the number of rows closed. Ported from ``invalidate``."""
        now = utc_now().isoformat()
        closed = 0
        for row in cls.live().filter("key", key).get():
            if value is not None and (row.value or "").lower().strip() != value.lower().strip():
                continue
            row.active = 0
            row.valid_to = now
            row.retrieval_weight *= cls._SUPERSEDE_RW_FACTOR
            row.save()
            closed += 1
        return closed

    @classmethod
    def decay(cls) -> int:
        """Absolute power-law decay of live FACTS' retrieval_weight (idempotent):
        ``rw = max(salience_floor, max(1, age_days) ** -d_base)`` from
        ``last_confirmed_at``. Only rows last confirmed > 1h ago, and only when the
        new weight actually moves (> epsilon). Returns rows updated. The shared
        superseded-decay and 90-day hard-GC are the DecayEngine's job, not here."""
        from datetime import timedelta  # noqa: PLC0415
        now = utc_now()
        cutoff = (now - timedelta(hours=1)).isoformat()
        now_ts = now.timestamp()
        updated = 0
        for row in cls.live().filter("last_confirmed_at", cutoff, "<").get():
            new_rw = cls._decayed_rw(row.last_confirmed_at, now_ts)
            if new_rw is not None and abs(new_rw - row.retrieval_weight) > cls._DECAY_RW_EPSILON:
                row.retrieval_weight = new_rw
                row.save()
                updated += 1
        return updated

    @classmethod
    def _decayed_rw(cls, confirmed_at: str | None, now_ts: float) -> float | None:
        from services.time_utils import parse_utc  # noqa: PLC0415
        if not confirmed_at:
            return None
        try:
            confirmed_ts = parse_utc(confirmed_at).timestamp()
        except Exception:
            return None
        age_days = (now_ts - confirmed_ts) / 86400.0
        if age_days <= 0:
            return None
        return max(cls._DECAY_SALIENCE_FLOOR, float(max(1.0, age_days) ** (-cls._DECAY_D_BASE)))
