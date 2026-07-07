"""ThreadGist — one ``thread_gist`` row: a terse topical label per thread.

Active-record row-model (Rule 5 / §4.1). The table's primary key is the
composite ``(channel, turn_id)`` — there is **no** ``id`` column — so the base
``Model.save()``/``delete()`` (id-centric: INSERT excludes ``id`` + reads
``lastrowid``; UPDATE/DELETE ``WHERE id = ?``) do not apply here. This model
carries its own :meth:`upsert` (``INSERT … ON CONFLICT(channel, turn_id) DO
UPDATE``) and :meth:`bulk_get` read, ported verbatim from the dissolved
``services/thread_gist_service.py``. ``__columns__`` omits ``id``, so the base's
phantom ``self.id`` never leaks into ``to_dict``/``to_json`` (``_fields`` keeps
only ``__dict__ ∩ __columns__``). Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

from typing import ClassVar, cast

from models.model import Model
from services.time_utils import utc_now


class ThreadGist(Model):
    """One per-thread topical label, keyed by the composite ``(channel, turn_id)``."""

    __columns__: ClassVar[tuple[str, ...]] = ("channel", "turn_id", "gist", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "thread_gist"

    channel: str
    turn_id: int
    gist: str
    created_at: str

    def upsert(self) -> None:
        """Persist (or replace) this thread's label, idempotent per turn.

        Ported from ``ThreadGistService.upsert``: on a ``(channel, turn_id)``
        conflict only ``gist`` is overwritten, preserving the original
        ``created_at``. ``created_at`` is stamped from :func:`utc_now` on insert."""
        self._bound_connection().execute(
            "INSERT INTO thread_gist (channel, turn_id, gist, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(channel, turn_id) DO UPDATE SET gist = excluded.gist",
            (self.channel, self.turn_id, self.gist, utc_now().isoformat()),
        )

    @classmethod
    def bulk_get(cls, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """Labels for a batch of turn_ids — the thread feed's collapsed labels.

        Ported from ``ThreadGistService.bulk_get``. Returns ``{turn_id: gist}``
        for every turn_id in ``turn_ids`` that has a label on ``channel``."""
        if not turn_ids:
            return {}
        rows = cls.filter("channel", channel).filter_in("turn_id", turn_ids).select("turn_id", "gist")
        return {int(cast("int", row["turn_id"])): cast("str", row["gist"]) for row in rows}
