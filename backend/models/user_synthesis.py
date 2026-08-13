"""Active-record row for the ``user_synthesis`` table — a versioned
append-only store for the user's running profile summary.

Every save writes a new immutable row; the ``version`` column is computed in a
single atomic SQL statement so two concurrent saves never collide on the same
version number. The latest synthesis is the row with the highest ``id`` — the
natural insert order — so ``latest()`` is a plain ``ORDER BY id DESC LIMIT 1``.
"""

from __future__ import annotations

from models.model import Model


class UserSynthesisRow(Model):
    """One ``user_synthesis`` row: an immutable versioned snapshot of the user profile."""

    __columns__ = ("id", "version", "content", "created_at")

    @classmethod
    def get_table(cls) -> str:
        return "user_synthesis"

    version: int
    content: str
    created_at: str

    @classmethod
    def latest(cls) -> "UserSynthesisRow | None":
        """Newest synthesis row (highest ``id``), or ``None`` if the table
        is empty. Ports the ``ORDER BY id DESC LIMIT 1`` read every caller uses
        to fetch the current user summary."""
        return cls.order_by("id DESC").first()

    @classmethod
    def latest_content(cls) -> str:
        """Content of the newest row, or ``""`` if there is no row yet.
        Convenience shortcut — callers that only need the prose don't have to
        hydrate a row first."""
        row = cls.latest()
        return row.content if row is not None else ""

    @classmethod
    def write(cls, content: str) -> None:
        """Append a new versioned snapshot. The ``version`` is computed in a
        single atomic SQL statement — ``INSERT ... SELECT COALESCE(MAX(version),0)+1``
        — so two concurrent saves never collide on the same version number
        (no read-then-write in Python). ``created_at`` is left unset so the SQL
        ``datetime('now')`` default fires on INSERT."""
        cls._bound_connection().execute(
            "INSERT INTO user_synthesis (version, content) "
            "SELECT COALESCE(MAX(version), 0) + 1, ? FROM user_synthesis",
            (content,),
        )
