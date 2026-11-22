"""ThreadGist — one ``thread_gist`` row: a terse topical label per thread.

Active-record row-model (Rule 5 / §4.1). A label is found by its thread's
opening transcript row id (``opening_id``, unique). The table has **no** ``id``
column, so the base ``Model.save()``/``delete()`` (id-centric: INSERT excludes
``id`` + reads ``lastrowid``; UPDATE/DELETE ``WHERE id = ?``) do not apply here.
This model carries its own :meth:`upsert` (``INSERT … ON CONFLICT(opening_id)
DO UPDATE``) and :meth:`bulk_get` read. ``__columns__`` omits ``id``, so the
base's phantom ``self.id`` never leaks into ``to_dict``/``to_json``
(``_fields`` keeps only ``__dict__ ∩ __columns__``). Holds no mp, calls no
service (Rule-3 depth).
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model
from services.time_utils import utc_now


class ThreadGist(Model):
    """One per-thread topical label, found by its thread's opening row id."""

    __columns__: ClassVar[tuple[str, ...]] = ("opening_id", "gist", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "thread_gist"

    opening_id: int
    gist: str
    created_at: str

    def upsert(self) -> None:
        """Persist (or replace) this thread's label, idempotent per thread.

        On an ``opening_id`` conflict only ``gist`` is overwritten, preserving
        the original ``created_at``. ``created_at`` is stamped from
        :func:`utc_now` on insert."""
        self._bound_connection().execute(
            "INSERT INTO thread_gist (opening_id, gist, created_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(opening_id) DO UPDATE SET gist = excluded.gist",
            (self.opening_id, self.gist, utc_now().isoformat()),
        )

    @classmethod
    def bulk_get(cls, channel: str, turn_ids: list[int]) -> dict[int, str]:
        """Labels for a batch of turn_ids — the thread feed's collapsed labels.

        Returns ``{turn_id: gist}`` for every turn_id in ``turn_ids`` on
        ``channel`` whose opening row carries a label."""
        if not turn_ids:
            return {}
        placeholders = ", ".join("?" for _ in turn_ids)
        rows = cls._bound_connection().execute(
            "SELECT t.turn_id, tg.gist FROM thread_gist tg JOIN transcript t ON t.id = tg.opening_id "
            f"WHERE t.channel = ? AND t.turn_id IN ({placeholders})",
            (channel, *turn_ids),
        ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}
