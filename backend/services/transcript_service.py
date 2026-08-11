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

from configs.enums.channels import Channel
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
        suppresses history. The FORK view excludes ``role='memory'`` rows
        (memory-step inputs — turn plumbing, not conversation); the MAIN view
        needs no filter because a memory-step turn never settles an assistant
        row, so the per-turn settle0 floor already drops it. A FORK always
        reads ``self.mp.channel`` — a fork IS the thread, its view is its own
        turn's rows; the split-channel read (``read_channel``, e.g.
        DiscoveryConfig) applies to the MAIN cross-turn view only. Turn ids
        are per-channel, so resolving ``read_channel`` on a FORK would cross
        namespaces and return another channel's unrelated turn."""
        if self.mp.config.suppress_history:
            return []
        watermark = self.mp.compaction_service.watermark()
        if self.mp._forked:
            ceiling = self.mp.uid if self.mp.uid is not None else _NO_CEILING
            return (
                Transcript.filter("channel", self.mp.channel)
                .filter("turn_id", self.mp.turn_id)
                .filter("id", watermark, ">")
                .filter("id", ceiling, "<")
                .filter("role", "memory", "!=")
                .order_by("id ASC")
                .get()
            )
        channel = self.mp.config.read_channel or self.mp.channel
        by_turn: dict[int, list[Transcript]] = {}
        for row in (
            Transcript.filter("channel", channel)
            .filter("turn_id", watermark, ">")
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
            Transcript.filter("channel", self.mp.channel)
            .filter("turn_id", self.mp.turn_id)
            .order_by("id ASC")
            .get()
        )

    def exchange_assistant_rows(self) -> list[Transcript]:
        """The CURRENT exchange's assistant rows — the interim-step prose that
        anchors tool calls in this exchange, feeding the act-trail interleave.
        Filtered to this MP's channel and turn, role ``assistant``, ordered by
        id ASC; when ``self.mp.uid`` is set (a real input row exists) only rows
        written after it survive — the uid is the exchange floor, mirroring
        ``ToolCall.by_exchange``'s ``transcript_id >= uid`` bound. When
        ``self.mp.uid`` is None (channels without an input row) there is no
        floor, so every assistant row of the turn is returned."""
        q = (
            Transcript.filter("channel", self.mp.channel)
            .filter("turn_id", self.mp.turn_id)
            .filter("role", "assistant")
            .order_by("id ASC")
        )
        if self.mp.uid is not None:
            q = q.filter("id", self.mp.uid, ">")
        return q.get()

    def window(self) -> list[Transcript]:
        """The pattern channel's id-bounded window over the ``user`` channel's
        content rows (ported from ``Transcript.window(["user"], after_id,
        before_id, require_content=True)`` — the pre-rewrite pattern-recognition
        read). Bounds come off this turn's ``PatternConfig`` (``_window_start``
        exclusive, ``_window_end`` inclusive); empty-content rows are excluded so
        the model only ever sees real utterances. Memory-step input rows live
        on the user channel but are plumbing, not utterances, so they are
        excluded. Zero-param (§2.4): the id bounds are config fields, reachable
        off ``self.mp.config``."""
        config = cast("PatternConfig", self.mp.config)
        return (
            Transcript.filter("channel", Channel.USER.value)
            .filter("id", config._window_start, ">")
            .filter("id", config._window_end, "<=")
            .filter("role", "memory", "!=")
            .filter("content", None, "IS NOT")
            .filter("content", "", "!=")
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
        latitude/longitude and real content survive. Memory-step input rows live
        on the user channel but are plumbing, not utterances, so they are
        excluded. Zero-param (§2.4)."""
        config = cast("GeoConfig", self.mp.config)
        return (
            Transcript.filter("channel", Channel.USER.value)
            .filter("id", config._window_start, ">")
            .filter("id", config._window_end, "<=")
            .filter("location_lat", None, "IS NOT")
            .filter("location_lon", None, "IS NOT")
            .filter("role", "memory", "!=")
            .filter("content", None, "IS NOT")
            .filter("content", "", "!=")
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
            Transcript.filter("channel", self.mp.channel)
            .filter("turn_id", self.mp.turn_id)
            .exists()
        )

    # ── writes ───────────────────────────────────────────────────────────────

    def append_input(self, content: str, *, thinking_level: str | None = None) -> int:
        """Write this turn's anchoring input row (unsettled) and return its id.
        ``thinking_level`` is persisted only when one of {auto, medium, high};
        otherwise NULL is stored."""
        valid = thinking_level if thinking_level in {"auto", "medium", "high"} else None
        return self._append(content, role=self.mp.config.role, settled=0, thinking_level=valid)

    def append_assistant(self, content: str) -> int:
        """Write one assistant row for this turn's step (settled) and poke
        every open surface to refetch the turn block — the visible-transcript
        write every chain step and the final synthesis land through. The
        broadcast gate lives in ``mp.push_websocket`` (silent/background configs
        are dropped there), so this is a single direct emit."""
        row_id = self._append(content, role="assistant", settled=1)
        self.mp.push_websocket(TurnSignal.updated(self.mp))
        return row_id

    def append_handover(self, content: str) -> int:
        """Write the pre-compaction hand-over row (role="compaction", settled=0).

        ``role="compaction"`` is deliberate: it matches no ``role='user'`` query —
        ``Transcript.last_user_message_at()`` gates idle cognition and DMN
        reflection with ``SELECT MAX(created_at) FROM transcript WHERE role = 'user'``
        (no channel filter), and a hand-over written under an input role would
        fool both.
        """
        return self._append(content, role="compaction", settled=0)

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
        row = Transcript.filter("id", settle_id).first()
        if row is not None:
            row.unsettle()

    # ── private helpers ──────────────────────────────────────────────────────

    def _anchor_row(self) -> Transcript | None:
        """This turn's anchoring input row, or ``None`` before it exists."""
        if self.mp.uid is None:
            return None
        return Transcript.filter("id", self.mp.uid).first()

    def _append(self, content: str, *, role: str, settled: int, thinking_level: str | None = None) -> int:
        """Write one transcript row for this turn and return its id."""
        loc = self._location()
        row = Transcript(
            channel=self.mp.channel, role=role, content=content,
            turn_id=self.mp.turn_id, settled=settled, xml_migrated=1,
            deliberation_score=0.0,
            location_lat=loc.get("lat"), location_lon=loc.get("lon"),
            location_name=loc.get("name"),
            thinking_level=thinking_level,
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
