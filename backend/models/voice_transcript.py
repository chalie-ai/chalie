"""VoiceTranscript — one ``voice_transcript`` row: the terminal outcome of the
pre-synthesis pipeline for a settled assistant transcript row.

Active-record row-model (Rule 5 / §4.1). The table's primary key is
``transcript_id`` — there is **no** ``id`` column — so the base
``Model.save()``/``delete()`` (id-centric) do not apply; ``__columns__`` omits
``id`` and this model carries its own conflict-safe :meth:`record` insert.
``file_path`` is the stored WAV's path relative to the voice root — the read
side resolves it via ``FileMapperService.get_voice_path`` (the
``transcript_files`` precedent: relative in the row, joined at read) — and
``None`` persists "synthesis failed after all attempts" (the red speaker
button survives reloads). Row absent = never attempted / still in flight.
Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

import json
from typing import ClassVar

from models.model import Model


class VoiceTranscript(Model):
    """One pipeline outcome per settled transcript row, keyed by ``transcript_id``."""

    __columns__: ClassVar[tuple[str, ...]] = ("transcript_id", "file_path")

    @classmethod
    def get_table(cls) -> str:
        return "voice_transcript"

    transcript_id: int
    file_path: str | None

    def record(self) -> None:
        """Persist this pipeline outcome, first terminal state wins.

        ``transcript_id`` is the PK, so a duplicate insert (two pipeline runs
        racing on the same row) is a no-op via ``ON CONFLICT … DO NOTHING``
        rather than an IntegrityError. Never commits — the bound connection
        autocommits and ``Database.transaction()`` groups multi-write atomic
        blocks (I6)."""
        self._bound_connection().execute(
            "INSERT INTO voice_transcript (transcript_id, file_path) "
            "VALUES (?, ?) "
            "ON CONFLICT(transcript_id) DO NOTHING",
            (self.transcript_id, self.file_path),
        )

    @classmethod
    def for_transcripts(cls, transcript_ids: list[int]) -> "list[VoiceTranscript]":
        """Every ``voice_transcript`` row for the given transcript ids. Binds
        the whole id list as one JSON-array parameter via ``json_each`` (see
        :meth:`~models.tool_call.ToolCall.delete_by_transcripts`) because a
        GC sweep's candidate set can exceed SQLite's bound-variable limit.
        Empty input is a clean no-op — no query runs."""
        if not transcript_ids:
            return []
        rows = cls._bound_connection().execute(
            "SELECT * FROM voice_transcript "
            "WHERE transcript_id IN (SELECT value FROM json_each(?))",
            (json.dumps(transcript_ids),),
        ).fetchall()
        return [cls.hydrate(row) for row in rows]

    @classmethod
    def delete_by_transcripts(cls, transcript_ids: list[int]) -> int:
        """Hard-delete every row for the given transcript ids — the voice leg
        of a transcript delete. The GC sweep runs with ``PRAGMA foreign_keys``
        OFF, so the ``ON DELETE CASCADE`` on this table never fires there and
        this explicit delete is the path that actually clears the rows (file
        removal is :meth:`~services.voice_transcript_service.
        VoiceTranscriptService.delete_for_transcripts`'s job, which calls
        this). Same ``json_each`` binding as :meth:`for_transcripts`; empty
        input is a clean no-op. Returns rows deleted."""
        if not transcript_ids:
            return 0
        cursor = cls._bound_connection().execute(
            "DELETE FROM voice_transcript "
            "WHERE transcript_id IN (SELECT value FROM json_each(?))",
            (json.dumps(transcript_ids),),
        )
        return cursor.rowcount or 0
