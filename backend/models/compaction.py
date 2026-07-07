"""Active-record row for the ``compactions`` table — the two-axis watermark.

A compaction is a row in its own table, never a transcript row, so firing one
never moves a ``turn_id`` and the turn boundary survives a mid-turn collapse.

The watermark is two-axis, selected by ``for_turn_id`` (the conversation view):

* **MAIN** view → ``for_turn_id IS NULL``; ``compacted_up_to`` is a **turn_id**.
  The spine read drops turns whose ``turn_id <= compacted_up_to``.
* **FORK** view → ``for_turn_id = T``; ``compacted_up_to`` is a **transcript.id**.
  The thread read drops rows of turn ``T`` whose ``id <= compacted_up_to``.

The two axes never collide (§6.1): a MAIN compaction can't hide a fork reply and
a FORK compaction can't hide the spine. This model owns the ``compactions`` SQL
only; ``CompactionService`` owns the checkpoint envelope (``_wrap_with_checkpoint``).
"""

from __future__ import annotations

from models.model import Model


class Compaction(Model):
    """One ``compactions`` row: a checkpoint on a MAIN or FORK watermark axis."""

    __columns__ = ("id", "channel", "for_turn_id", "compacted_up_to", "content", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "compactions"

    channel: str
    for_turn_id: int | None
    compacted_up_to: int
    content: str
    created_at: str

    @classmethod
    def latest_main(cls, channel: str) -> Compaction | None:
        """Newest MAIN-view checkpoint for ``channel`` (``for_turn_id IS NULL``),
        or ``None``. ``compacted_up_to`` is a turn_id. Ports ``get_compaction``'s
        ``channel = ? AND for_turn_id IS NULL ORDER BY id DESC LIMIT 1``."""
        return (
            cls.filter("channel", channel)
            .filter("for_turn_id", None, "IS")
            .order_by("id DESC")
            .first()
        )

    @classmethod
    def latest_fork(cls, channel: str, for_turn_id: int) -> Compaction | None:
        """Newest FORK-view checkpoint for ``(channel, for_turn_id)``, or ``None``.
        ``compacted_up_to`` is a transcript.id. Ports ``get_compaction``'s
        ``channel = ? AND for_turn_id = ? ORDER BY id DESC LIMIT 1``."""
        return (
            cls.filter("channel", channel)
            .filter("for_turn_id", for_turn_id)
            .order_by("id DESC")
            .first()
        )

    @classmethod
    def write(cls, channel: str, for_turn_id: int | None, compacted_up_to: int, content: str) -> Compaction:
        """Append a checkpoint for one ``(channel, for_turn_id)`` scope and return
        it. ``for_turn_id`` None = MAIN (``compacted_up_to`` a turn_id), int = FORK
        (a transcript.id). ``created_at`` is left unset so the SQL ``datetime('now')``
        default fires on INSERT. Ports ``write_compaction``'s single INSERT."""
        return cls(
            channel=channel,
            for_turn_id=for_turn_id,
            compacted_up_to=compacted_up_to,
            content=content,
        ).save()
