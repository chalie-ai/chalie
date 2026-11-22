"""Coordinating service for the two-axis compaction watermark (§4.2, §6.1).

Holds ``self.mp`` and reaches ``models.compaction.Compaction`` for every
read/write; no raw SQL, no WS emit (a checkpoint is prose folded into the next
prompt, never a wire frame). The MAIN/FORK axis is never a parameter — it is
the view the CURRENT turn is on, read straight off ``self.mp``
(``mp.fork_view`` selects the axis; the FORK scope is the current thread's
opening transcript row id; ``mp.config`` supplies the channel, split read/write
aware) exactly as ``chat_history_compactor`` and ``TranscriptService.read`` key
theirs. The watermark is a transcript row id on both axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from models.compaction import Compaction
from models.transcript import Transcript

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
        if self.mp.fork_view:
            return self.mp.config.channel
        return self.mp.config.read_channel or self.mp.config.channel

    def _opening_id(self) -> int | None:
        """The FORK scope: the current thread's opening transcript row id,
        ``None`` when the thread has no rows yet."""
        return Transcript.first_id(self.mp.channel, self.mp.turn_id)

    def get(self) -> Compaction | None:
        """Latest checkpoint for the current view, or ``None``. MAIN reads
        ``Compaction.latest_main``; FORK reads ``Compaction.latest_fork`` for
        the current thread, and a thread with no rows has none."""
        channel = self._channel()
        if not self.mp.fork_view:
            return Compaction.latest_main(channel)
        opening_id = self._opening_id()
        return Compaction.latest_fork(channel, opening_id) if opening_id is not None else None

    def watermark(self) -> int:
        """``compacted_up_to_id`` for the current view, or ``0`` when no
        checkpoint has ever been written — the row-id floor callers cut
        history on (everything at/below it is already carried as the
        checkpoint's prose)."""
        compaction = self.get()
        return (compaction.compacted_up_to_id or 0) if compaction else 0

    def write(self, compacted_up_to_id: int, content: str) -> Compaction:
        """Append a checkpoint for the current view and return it.
        ``compacted_up_to_id``/``content`` are the compactor's own output — the
        max transcript row id folded and the model's verbatim summary —
        genuinely external, never mp-reachable. A FORK thread with no rows has
        nothing to scope the checkpoint to, which is a caller bug: it raises.

        The MAIN read of the turn's own channel skips the turn in flight, so
        none of its rows were folded: the cut-off stays below its first row
        above the current watermark, even when another thread wrote rows after
        it."""
        if not self.mp.fork_view:
            channel = self._channel()
            if channel == self.mp.channel:
                live = Transcript.first_id(channel, self.mp.turn_id, above=self.watermark())
                if live is not None:
                    compacted_up_to_id = min(compacted_up_to_id, live - 1)
            return Compaction.write(channel, None, compacted_up_to_id, content)
        opening_id = self._opening_id()
        if opening_id is None:
            raise ValueError(f"no opening row for {self.mp.channel} turn {self.mp.turn_id}; nothing to compact")
        return Compaction.write(self._channel(), opening_id, compacted_up_to_id, content)

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
