"""Active-record row for the ``compactions`` table — the two-axis watermark.

A compaction is a row in its own table, never a transcript row, so firing one
never moves a ``turn_id`` and the turn boundary survives a mid-turn collapse.

The watermark is two-axis, selected by ``opening_id`` (the conversation view);
``compacted_up_to_id`` is a **transcript.id** on both:

* **MAIN** view → ``opening_id IS NULL``. The spine read drops the channel's
  rows whose ``id <= compacted_up_to_id``.
* **FORK** view → ``opening_id`` = the thread's opening transcript row id, so a
  thread's checkpoint is found by its record id. The thread read drops that
  thread's rows whose ``id <= compacted_up_to_id``.

A row with no ``compacted_up_to_id`` is one the upgrade could not match to a
transcript row (its thread has no rows), and it is never read.
The two axes never collide (§6.1): a MAIN compaction can't hide a fork reply and
a FORK compaction can't hide the spine. This model owns the ``compactions`` SQL
only; ``CompactionService`` owns the checkpoint envelope (``_wrap_with_checkpoint``).
"""

from __future__ import annotations

from models.model import Model


class Compaction(Model):
    """One ``compactions`` row: a checkpoint on a MAIN or FORK watermark axis."""

    __columns__ = ("id", "channel", "opening_id", "compacted_up_to_id", "content", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "compactions"

    channel: str
    opening_id: int | None
    compacted_up_to_id: int | None
    content: str
    created_at: str

    @classmethod
    def latest_main(cls, channel: str) -> Compaction | None:
        """Newest MAIN-view checkpoint for ``channel`` (``opening_id IS NULL``),
        or ``None``."""
        return (
            cls.filter("channel", channel)
            .filter("opening_id", None, "IS")
            .filter("compacted_up_to_id", None, "IS NOT")
            .order_by("id DESC")
            .first()
        )

    @classmethod
    def latest_fork(cls, channel: str, opening_id: int) -> Compaction | None:
        """Newest FORK-view checkpoint for the thread whose opening row is
        ``opening_id``, or ``None``."""
        return (
            cls.filter("channel", channel)
            .filter("opening_id", opening_id)
            .filter("compacted_up_to_id", None, "IS NOT")
            .order_by("id DESC")
            .first()
        )

    @classmethod
    def write(cls, channel: str, opening_id: int | None, compacted_up_to_id: int, content: str) -> Compaction:
        """Append a checkpoint for one ``(channel, opening_id)`` scope and return
        it. ``opening_id`` None = MAIN, int = FORK. ``created_at`` is left unset
        so the SQL ``datetime('now')`` default fires on INSERT."""
        return cls(
            channel=channel,
            opening_id=opening_id,
            compacted_up_to_id=compacted_up_to_id,
            content=content,
        ).save()
