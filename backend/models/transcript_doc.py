"""TranscriptDoc — one ``transcript_docs`` row: a link between a turn's
anchoring transcript row and an uploaded document.

Active-record row-model (Rule 5 / §4.1). The table's primary key is the
composite ``(transcript_id, doc_id)`` — there is **no** ``id`` column — so the
base ``Model.save()``/``delete()`` (id-centric: INSERT excludes ``id`` + reads
``lastrowid``; UPDATE/DELETE ``WHERE id = ?``) do not apply here. Mirrors
``ThreadGist``: ``__columns__`` omits ``id`` (so the base's phantom ``self.id``
never leaks into ``to_dict``/``to_json`` — ``_fields`` keeps only
``__dict__ ∩ __columns__``), and this model carries its own conflict-safe
:meth:`link` insert. Holds no mp, calls no service (Rule-3 depth).

The read side (``api.threads._fetch_attachments_for_transcripts``) JOINs this
table to re-render a message's attachments after a refresh; both foreign keys
CASCADE, so a link never dangles once either end is deleted.
"""

from __future__ import annotations

from typing import ClassVar

from models.model import Model


class TranscriptDoc(Model):
    """One transcript↔document link, keyed by the composite ``(transcript_id, doc_id)``."""

    __columns__: ClassVar[tuple[str, ...]] = ("transcript_id", "doc_id")

    @classmethod
    def get_table(cls) -> str:
        return "transcript_docs"

    transcript_id: int
    doc_id: str

    def link(self) -> None:
        """Persist this transcript↔document link, idempotent per pair.

        The composite ``(transcript_id, doc_id)`` PK dedups, so re-linking the
        same document to the same turn (a retry, or the same file attached
        twice) is a no-op via ``ON CONFLICT … DO NOTHING`` rather than an
        IntegrityError. Never commits — the bound connection autocommits and
        ``Database.transaction()`` groups multi-write atomic blocks (I6)."""
        self._bound_connection().execute(
            "INSERT INTO transcript_docs (transcript_id, doc_id) "
            "VALUES (?, ?) "
            "ON CONFLICT(transcript_id, doc_id) DO NOTHING",
            (self.transcript_id, self.doc_id),
        )
