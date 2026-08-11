"""Coordinating service for the two-axis compaction watermark (§4.2, §6.1).

Holds ``self.mp`` and reaches ``models.compaction.Compaction`` for every
read/write; no raw SQL, no WS emit (a checkpoint is prose folded into the next
prompt, never a wire frame). The MAIN/FORK axis is never a parameter — it is
the view the CURRENT turn is on, read straight off ``self.mp`` (``mp._forked``
selects the axis; ``mp.turn_id`` is the FORK scope key; ``mp.config`` supplies
the channel, split read/write aware) exactly as ``chat_history_compactor``
already keys its own read/write pair.

ESCALATION (open, not resolved by this file): ``mp._forked`` is not a member
of the §4.3 public attribute contract table — that surface has no MAIN/FORK
axis selector yet because it isn't derivable from any attribute the table
does list (``turn_id`` resolves to a concrete int on both axes once
``begin()`` allocates it, per §6.8). ``CompactionService`` cannot add the
missing member itself — the contract is owned by F1 (``controllers/message_processor.py``,
not yet built). Before Phase F wires this service in, F1 must add a public
``is_forked: bool`` (or equivalent) to §4.3 and this file must be re-pointed
at it in the same change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from models.compaction import Compaction

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor


class CompactionService:
    """get/write/watermark for the current turn's compaction view, plus the
    checkpoint-envelope read that used to be ``MessageProcessor._wrap_with_checkpoint``."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def _channel(self) -> str:
        """The channel this view's checkpoint is keyed on. MAIN: the READ
        channel when the config splits read/write (e.g. ``DiscoveryConfig``) —
        its cross-turn view is the read channel's spine, so that is the
        watermark that must advance. FORK: always the write channel — a fork
        is its own thread, and ``TranscriptService.read()`` scopes its FORK
        view the same way, so the id floor and the rows it cuts stay on one
        channel. Matches ``chat_history_compactor``'s own keying."""
        if self.mp._forked:
            return self.mp.config.channel
        return self.mp.config.read_channel or self.mp.config.channel

    def _for_turn_id(self) -> int | None:
        """The watermark axis selector: the current turn_id on a FORK reply,
        ``None`` on the MAIN spine. Reads ``mp._forked`` — see the module
        docstring's ESCALATION note: not yet a sanctioned §4.3 member."""
        return cast("int | None", self.mp.turn_id) if self.mp._forked else None

    def get(self) -> Compaction | None:
        """Latest checkpoint for the current view, or ``None``. MAIN
        (``for_turn_id`` None) reads ``Compaction.latest_main``; FORK reads
        ``Compaction.latest_fork``."""
        channel = self._channel()
        for_turn_id = self._for_turn_id()
        if for_turn_id is None:
            return Compaction.latest_main(channel)
        return Compaction.latest_fork(channel, for_turn_id)

    def watermark(self) -> int:
        """``compacted_up_to`` for the current view, or ``0`` when no
        checkpoint has ever been written — the read floor callers cut history
        on (everything at/below it is already carried as the checkpoint's
        prose)."""
        compaction = self.get()
        return compaction.compacted_up_to if compaction else 0

    def write(self, compacted_up_to: int, content: str) -> Compaction:
        """Append a checkpoint for the current view and return it.
        ``compacted_up_to``/``content`` are the compactor's own output — the
        max id/turn_id folded and the model's verbatim summary — genuinely
        external, never mp-reachable."""
        return Compaction.write(self._channel(), self._for_turn_id(), compacted_up_to, content)

    def checkpoint(self, user_body: str) -> str:
        """Wrap ``user_body`` in the ``### Checkpoint`` envelope when a
        checkpoint exists for the current view, else return it unchanged.
        Ports ``MessageProcessor._wrap_with_checkpoint``: the envelope pairs
        the checkpoint's reconstruction of everything at/below the watermark
        with the live tail above it that the history read returns."""
        compaction = self.get()
        content = (compaction.content or "").strip() if compaction else ""
        if not content:
            return user_body
        return (
            "### Checkpoint - What you were previously discussing / doing\n"
            f"{content}\n"
            "\n"
            "---\n"
            "### Current State - What's happening in the current turn\n"
            f"{user_body}"
        )
