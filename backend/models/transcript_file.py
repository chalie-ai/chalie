"""TranscriptFile — one ``transcript_files`` row: a link between a turn's
anchoring transcript row and an uploaded file.

Active-record row-model (Rule 5 / §4.1). The table's primary key is the
composite ``(transcript_id, path)`` — there is **no** ``id`` column — so the
base ``Model.save()``/``delete()`` (id-centric: INSERT excludes ``id`` + reads
``lastrowid``; UPDATE/DELETE ``WHERE id = ?``) do not apply here.
``__columns__`` omits ``id`` (so the base's phantom
``self.id`` never leaks into ``to_dict``/``to_json`` — ``_fields`` keeps only
``__dict__ ∩ __columns__``), and this model carries its own conflict-safe
:meth:`link` insert. Holds no mp, calls no service (Rule-3 depth).

The read side (``TurnSerializerService``'s attachment hydration) reads this
table directly — no join — and resolves the absolute path by joining the
relative ``path`` onto the documents root.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class TranscriptFile(Model):
    """One transcript↔file link, keyed by the composite ``(transcript_id, path)``."""

    __columns__: ClassVar[tuple[str, ...]] = ("transcript_id", "path")

    @classmethod
    def get_table(cls) -> str:
        return "transcript_files"

    transcript_id: int
    path: str

    def link(self) -> None:
        """Persist this transcript↔file link, idempotent per pair.

        The composite ``(transcript_id, path)`` PK dedups, so re-linking the
        same file to the same turn (a retry, or the same file attached twice)
        is a no-op via ``ON CONFLICT … DO NOTHING`` rather than an
        IntegrityError. Never commits — the bound connection autocommits and
        ``Database.transaction()`` groups multi-write atomic blocks (I6)."""
        self._bound_connection().execute(
            "INSERT INTO transcript_files (transcript_id, path) "
            "VALUES (?, ?) "
            "ON CONFLICT(transcript_id, path) DO NOTHING",
            (self.transcript_id, self.path),
        )
