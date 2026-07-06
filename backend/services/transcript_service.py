"""Channel/turn-scoped transcript access — the ONE gateway to the ``transcript``
table for the MP spine (Rule-3 depth: coordinating service, holds ``mp``).

Every read/write below derives its channel and turn identity off ``self.mp``
(never a parameter — §2.4) and runs through ``models.transcript.Transcript``'s
active-record surface (Critical 3) — this class issues no SQL of its own.
FORK/MAIN read scoping and the settle boundary (§6.1) are preserved exactly:
settle0 is a turn's first ``role='assistant' AND settled=1`` row; a FORK view
(a reply into an already-settled turn) reads the WHOLE turn above the
compaction watermark, unfloored; a MAIN view (a fresh turn) reads every prior
turn above the watermark, each floored at its own settle0. A write that
changes the visible transcript (``append_assistant``) pokes every open
surface with ``TurnSignal(updated)`` via ``mp.push_websocket`` (Rule 7/9).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from models.transcript import Transcript
from models.turn_signal import TurnSignal

if TYPE_CHECKING:
    from configs.channels.geo_pattern import GeoConfig
    from configs.channels.pattern import PatternConfig
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

#: "No upper bound" sentinel for the FORK view's id ceiling (SQLite's max INTEGER).
_NO_CEILING = 9223372036854775807


class TranscriptService:
    """Transcript access for one turn — every method reads its channel/turn/
    row identity off ``self.mp`` rather than taking it as an argument."""

    def __init__(self, mp: MessageProcessor) -> None:
        # Turn-scoped: every read/write below derives channel/turn identity off
        # ``self.mp`` (§2.4) — this service is never constructed off-turn.
        self.mp = mp

    # ── reads ────────────────────────────────────────────────────────────────

    def read(self) -> list[Transcript]:
        """The current turn's history view, above the compaction watermark
        (§6.1): FORK (``self.mp._forked`` — a reply into an already-settled
        thread) returns the whole turn, unfloored, between the watermark and
        this turn's anchoring row; MAIN (a fresh turn) returns every prior
        turn above the watermark, each floored at its own settle0. Keyed off
        the same ``mp._forked`` flag ``CompactionService`` uses to pick its
        watermark axis, so the two never disagree mid-turn (a live
        ``settle0`` lookup can flip under a MAIN turn that settles more than
        one row before its terminal step; the fixed flag can't). Feeds
        ``PromptService``'s history assembly. ``[]`` when the config
        suppresses history."""
        if self.mp.config.suppress_history:
            return []
        channel = self.mp.config.read_channel or self.mp.channel
        watermark = self.mp.compaction_service.watermark()
        if self.mp._forked:
            ceiling = self.mp.uid if self.mp.uid is not None else _NO_CEILING
            return (
                Transcript.filter("channel = ?", channel)
                .filter("turn_id = ?", self.mp.turn_id)
                .filter("id > ? AND id < ?", watermark, ceiling)
                .order_by("id ASC")
                .get()
            )
        by_turn: dict[int, list[Transcript]] = {}
        for row in (
            Transcript.filter("channel = ?", channel)
            .filter("turn_id > ?", watermark)
            .order_by("id ASC")
            .get()
        ):
            by_turn.setdefault(cast("int", row.to_dict()["turn_id"]), []).append(row)
        rows: list[Transcript] = []
        for tid, turn_rows in by_turn.items():
            floor = Transcript.settle0(channel, tid)
            if floor is not None:
                rows.extend(r for r in turn_rows if cast("int", r.id) <= floor)
        return rows

    def turn_rows(self) -> list[Transcript]:
        """Every row of this turn, oldest-first, unfloored — the FORK/act-
        trail view of the turn currently in flight."""
        return (
            Transcript.filter("channel = ?", self.mp.channel)
            .filter("turn_id = ?", self.mp.turn_id)
            .order_by("id ASC")
            .get()
        )

    def window(self) -> list[Transcript]:
        """The pattern channel's id-bounded window over the ``user`` channel's
        content rows (ported from ``Transcript.window(["user"], after_id,
        before_id, require_content=True)`` — the pre-rewrite pattern-recognition
        read). Bounds come off this turn's ``PatternConfig`` (``_window_start``
        exclusive, ``_window_end`` inclusive); empty-content rows are excluded so
        the model only ever sees real utterances. Zero-param (§2.4): the id
        bounds are config fields, reachable off ``self.mp.config``."""
        config = cast("PatternConfig", self.mp.config)
        return (
            Transcript.filter("channel = ?", "user")
            .filter("id > ? AND id <= ?", config._window_start, config._window_end)
            .filter("content IS NOT NULL AND content != ''")
            .order_by("id ASC")
            .get()
        )

    def location_window(self) -> list[Transcript]:
        """The geo-pattern channel's id-bounded window over the ``user``
        channel's location-tagged content rows (ported from
        ``Transcript.window(["user"], after_id, before_id, require_location=True,
        require_content=True)`` — the pre-rewrite geo read). Bounds come off this
        turn's ``GeoConfig`` (``_window_start`` exclusive, ``_window_end``
        inclusive), same semantics as :meth:`window`; only rows carrying a
        latitude/longitude and real content survive. Zero-param (§2.4)."""
        config = cast("GeoConfig", self.mp.config)
        return (
            Transcript.filter("channel = ?", "user")
            .filter("id > ? AND id <= ?", config._window_start, config._window_end)
            .filter("location_lat IS NOT NULL AND location_lon IS NOT NULL")
            .filter("content IS NOT NULL AND content != ''")
            .order_by("id ASC")
            .get()
        )

    def deliberation_score(self) -> float:
        """This turn's persisted deliberation score (§6.12) off its anchoring
        input row, or ``0.0`` before that row exists / has a score."""
        row = self._anchor_row()
        return float(row.deliberation_score) if row and row.deliberation_score is not None else 0.0

    # ── turn identity (§6.8, resolved once inside begin()) ─────────────────────

    def allocate_turn(self) -> int:
        """Resolve this turn's per-channel turn_id off ``self.mp.turn_id`` (the
        caller's request): a real id (``>= 0``) is used verbatim (a reply forks
        it); the unset sentinel (``-1``) allocates the channel's next
        ``MAX(turn_id)+1``. Atomicity is the single-writer ``begin()``
        transaction (``BEGIN IMMEDIATE``), not a Python lock — two concurrent
        fresh turns on one channel can never read the same max and collide."""
        requested = self.mp.turn_id
        return Transcript.next_turn_id(requested if requested >= 0 else None, self.mp.channel)

    def turn_exists(self) -> bool:
        """Whether this turn's ``(channel, turn_id)`` already has any transcript
        row — the fork guard (§6.8): a reply can only fork a turn that exists,
        else ``begin()`` rejects the request."""
        return (
            Transcript.filter("channel = ?", self.mp.channel)
            .filter("turn_id = ?", self.mp.turn_id)
            .exists()
        )

    # ── writes ───────────────────────────────────────────────────────────────

    def append_input(self, content: str) -> int:
        """Write this turn's anchoring input row (unsettled) and return its id."""
        return self._append(content, role=self.mp.config.role, settled=0)

    def append_assistant(self, content: str) -> int:
        """Write one assistant row for this turn's step (settled) and poke
        every open surface to refetch the turn block — the visible-transcript
        write every chain step and the final synthesis land through. The
        broadcast gate lives in ``mp.push_websocket`` (silent/background configs
        are dropped there), so this is a single direct emit."""
        row_id = self._append(content, role="assistant", settled=1)
        self.mp.push_websocket(TurnSignal.updated(self.mp))
        return row_id

    def set_deliberation_score(self, score: float) -> None:
        """Persist ``score`` on this turn's anchoring input row — the value
        that drives thinking-level selection (§6.12). A no-op before that row
        exists (``skip_input_row`` channels)."""
        row = self._anchor_row()
        if row is not None:
            row.set_deliberation_score(score)

    def settle(self) -> int | None:
        """This turn's settle0 — the id of its first settled assistant row,
        or ``None`` while the turn is still in flight (§6.1)."""
        return Transcript.settle0(self.mp.channel, self.mp.turn_id)

    def unsettle(self) -> None:
        """Demote this turn's settle0 row back to unsettled — the cross-table
        half of a settling tool-call re-opening the turn (§6.9), driven by
        ``ToolCallService.start``/``record`` via ``self.mp.transcript_service``."""
        settle_id = self.settle()
        if settle_id is None:
            return
        row = Transcript.filter("id = ?", settle_id).first()
        if row is not None:
            row.unsettle()

    def link_doc(self, doc_id: str) -> None:
        """Link an uploaded document to this turn's anchoring transcript row
        so a page refresh can re-render the attachment from its stored id.

        NOT YET PERSISTABLE: the ``transcript_docs`` join table (composite
        ``(transcript_id, doc_id)`` key, no ``id`` column) has no active-record
        model on this spine (a gap against the B1 model inventory — this
        service may not add one; see STOP-LINE) and this file may not issue
        raw SQL (I6). Logs loudly rather than silently dropping the link so
        the gap stays visible until a model closes it."""
        logger.warning(
            "[TranscriptService] link_doc(doc_id=%s) for transcript %s NOT persisted — "
            "transcript_docs has no model on this spine yet",
            doc_id, self.mp.uid,
        )

    def gc(self, cited_ids: frozenset[int]) -> int:
        """Delete this turn's rows already folded below the compaction
        watermark, on the turn's own axis (§6.1) — the same ``mp._forked``
        flag ``read()``/``CompactionService`` key on: MAIN measures the
        watermark against ``turn_id`` (the whole turn folds at once, only
        once absorbed); FORK measures it against transcript ``id`` (rows
        fold one at a time as the thread scrolls past it). ``cited_ids``
        (episode citations — not ``mp``-reachable, §3.11) and settle0 itself
        are never collected. Deletes run in one atomic block; returns the
        count deleted. A no-op on a turn that hasn't settled yet (its
        boundary is undefined). Tool-call rows anchored to a deleted row are
        NOT cascaded here — ``ToolCallService`` exposes no bulk-delete entry
        yet; orphaned ``tool_calls`` rows are swept by the separate retention
        pass."""
        settle_id = self.settle()
        if settle_id is None:
            return 0
        watermark = self.mp.compaction_service.watermark()
        ids = [cast("int", r.id) for r in self.turn_rows()]
        if self.mp._forked:
            dead = [rid for rid in ids if rid not in cited_ids and rid != settle_id and rid <= watermark]
        elif self.mp.turn_id > watermark:
            return 0
        else:
            dead = [rid for rid in ids if rid not in cited_ids and rid != settle_id and rid > settle_id]
        if not dead:
            return 0
        with self.mp.db.transaction():
            for rid in dead:
                Transcript(id=rid).delete()
        return len(dead)

    # ── private helpers ──────────────────────────────────────────────────────

    def _anchor_row(self) -> Transcript | None:
        """This turn's anchoring input row, or ``None`` before it exists."""
        if self.mp.uid is None:
            return None
        return Transcript.filter("id = ?", self.mp.uid).first()

    def _append(self, content: str, *, role: str, settled: int) -> int:
        """Write one transcript row for this turn and return its id."""
        loc = self._location()
        row = Transcript(
            channel=self.mp.channel, role=role, content=content,
            turn_id=self.mp.turn_id, settled=settled, xml_migrated=1,
            deliberation_score=0.0,
            location_lat=loc.get("lat"), location_lon=loc.get("lon"),
            location_name=loc.get("name"),
        ).save()
        return cast("int", row.id)

    def _location(self) -> dict[str, object]:
        """Live location for a new row, gated to channels whose source profile
        permits backfill (user-activity channels) — muted/background channels
        store NULL so their rows never corrupt the geo signal. ``{}`` when
        gated off or when the lookup fails."""
        from services.source_profiles import profile_for  # noqa: PLC0415
        if not profile_for(self.mp.channel).location_backfill:
            return {}
        from services.locale_service import get_location  # noqa: PLC0415
        try:
            return get_location()
        except Exception as exc:  # noqa: BLE001 — a geo hiccup must not lose the row
            logger.warning(
                "[TranscriptService] location backfill failed for %s: %s",
                self.mp.channel, exc,
            )
            return {}
