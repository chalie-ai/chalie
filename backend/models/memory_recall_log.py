"""One ``memory_recall_log`` row — recall telemetry for one episode-recall call.

Active-record row-model (Rule 5 / §4.1). Pure CRUD (Rule-3 depth): holds no
``mp``, imports no service, never emits WS, never reaches upstream. It only
stores its fields, projects itself, and runs its own table's SQL on the
connection bound onto the :class:`Model` base.

``id`` is the DDL's own ``INTEGER PRIMARY KEY AUTOINCREMENT``, so the base's
``save``/``get``/``delete`` id-centric verbs apply unmodified — no id-generation
override is needed here. Both ``id`` and ``created_at`` carry SQL defaults in
the schema, so callers must leave them unset and let the base ``save()`` omit
them so the DB defaults fire on INSERT.

This model is the SOLE home of ``memory_recall_log`` SQL;
:class:`~services.memory_service.MemoryService` writes exclusively through it.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class MemoryRecallLog(Model):
    """One ``memory_recall_log`` row: field storage + CRUD."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "created_at",
        "turn_uid", "transcript_id", "channel", "caller",
        "query", "query_embedding_hash",
        "episode_count", "floor_cut_count", "final_rrf_count", "top_distances",
    )

    @classmethod
    def get_table(cls) -> str:
        return "memory_recall_log"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    turn_uid: str
    transcript_id: int | None
    channel: str | None
    caller: str
    query: str
    query_embedding_hash: str
    episode_count: int
    floor_cut_count: int
    final_rrf_count: int
    top_distances: str
