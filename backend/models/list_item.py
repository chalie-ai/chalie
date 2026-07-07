"""ListItem — one ``list_items`` row: one entry inside a :class:`~models.list.List`.

Active-record row-model (Rule 5 / §4.1). Same TEXT-hex PK generation as
:class:`~models.list.List` (``secrets.token_hex(4)``, mirroring
``Episode.save``'s override). This model is the SOLE home of ``list_items``
SQL; ``ListService`` reads and writes exclusively through it. Holds no mp,
calls no service (Rule-3 depth).
"""

from __future__ import annotations

import secrets
from typing import ClassVar, Self

from models.model import Model
from models.query import Query


class ListItem(Model):
    """One ``list_items`` row: field storage + CRUD + the position/dedupe
    reads and the content-addressed targeted writes the list ability needs."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "list_id", "content", "checked", "position", "metadata",
        "added_at", "updated_at", "removed_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "list_items"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    list_id: str
    content: str
    checked: int
    position: int
    metadata: str
    added_at: str
    updated_at: str
    removed_at: str | None

    def save(self) -> Self:
        """INSERT a new item (TEXT-hex PK generated here), or UPDATE the
        existing row — mirrors ``List.save``'s override."""
        if self.id is not None:
            return super().save()
        self.id = secrets.token_hex(4)
        connection = self._bound_connection()
        columns = self._fields()
        values = [getattr(self, name) for name in columns]
        column_list = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO {self.get_table()} ({column_list}) VALUES ({placeholders})",
            values,
        )
        return self

    # ── Live scope + id reads ───────────────────────────────────────────────

    @classmethod
    def active(cls, list_id: str) -> Query[Self]:
        """The live-item scope for one list, ordered ``position ASC, added_at
        ASC`` — the one shared predicate every active-items read builds on.
        The multi-column ORDER BY is a literal fragment interpolated verbatim
        by the builder (like ``Episode.fact_extraction_backlog``'s
        ``created_at ASC, id ASC``) — not a user-controllable value, so this
        is safe despite the builder's identifier check only covering
        ``filter``/``filter_in`` keys."""
        return (
            cls.filter("list_id", list_id)
            .filter("removed_at", None, "IS")
            .order_by("position ASC, added_at ASC")
        )

    @classmethod
    def by_id_in_list(cls, item_id: str, list_id: str) -> Self | None:
        """The single active item for an exact id within one list, or ``None``."""
        return (
            cls.filter("id", item_id)
            .filter("list_id", list_id)
            .filter("removed_at", None, "IS")
            .first()
        )

    @classmethod
    def max_position(cls, list_id: str) -> int:
        """Highest active ``position`` in a list, or -1 when empty — the
        append cursor. Raw SQL: the ``MAX()`` aggregate can't be expressed by
        the structured builder."""
        row = cls._bound_connection().execute(
            "SELECT COALESCE(MAX(position), -1) FROM list_items "
            "WHERE list_id = ? AND removed_at IS NULL",
            (list_id,),
        ).fetchone()
        return int(row[0]) if row is not None else -1

    @classmethod
    def active_normalized_contents(cls, list_id: str) -> set[str]:
        """Normalized (``LOWER(TRIM(...))``) content of every active item in a
        list — the dedupe comparison set. Raw SQL: the ``LOWER``/``TRIM``
        expression can't be expressed by the identifier-only filter builder."""
        rows = cls._bound_connection().execute(
            "SELECT LOWER(TRIM(content)) FROM list_items "
            "WHERE list_id = ? AND removed_at IS NULL",
            (list_id,),
        ).fetchall()
        return {str(row[0]) for row in rows}

    @classmethod
    def find_removed_by_content(cls, list_id: str, normalized_content: str) -> str | None:
        """The most-recently-removed item id in a list whose normalized
        content matches, or ``None`` — the revive-on-re-add lookup. Raw SQL:
        same ``LOWER``/``TRIM`` expression constraint as
        :meth:`active_normalized_contents`."""
        row = cls._bound_connection().execute(
            "SELECT id FROM list_items "
            "WHERE list_id = ? AND LOWER(TRIM(content)) = ? AND removed_at IS NOT NULL "
            "ORDER BY removed_at DESC LIMIT 1",
            (list_id, normalized_content),
        ).fetchone()
        return str(row[0]) if row is not None else None

    # ── Targeted writes ──────────────────────────────────────────────────────

    @classmethod
    def revive(cls, item_id: str, position: int) -> None:
        """Bring a previously-removed item back onto its list at a fresh
        ``position``, unchecked."""
        cls._bound_connection().execute(
            "UPDATE list_items SET removed_at = NULL, checked = 0, "
            "position = ?, updated_at = datetime('now') WHERE id = ?",
            (position, item_id),
        )

    @classmethod
    def clear_active(cls, list_id: str) -> int:
        """Soft-remove every active item in a list — the ``clear`` action.
        Returns the number of rows updated."""
        cursor = cls._bound_connection().execute(
            "UPDATE list_items SET removed_at = datetime('now'), updated_at = datetime('now') "
            "WHERE list_id = ? AND removed_at IS NULL",
            (list_id,),
        )
        return cursor.rowcount or 0

    @classmethod
    def remove_by_content(cls, list_id: str, normalized_content: str) -> int:
        """Soft-remove every active item in a list whose normalized content
        matches. Returns the number of rows updated."""
        cursor = cls._bound_connection().execute(
            "UPDATE list_items SET removed_at = datetime('now'), updated_at = datetime('now') "
            "WHERE list_id = ? AND LOWER(TRIM(content)) = ? AND removed_at IS NULL",
            (list_id, normalized_content),
        )
        return cursor.rowcount or 0

    @classmethod
    def set_checked_by_content(cls, list_id: str, normalized_content: str, checked: bool) -> int:
        """Set the checked state of every active item in a list whose
        normalized content matches. Returns the number of rows updated."""
        cursor = cls._bound_connection().execute(
            "UPDATE list_items SET checked = ?, updated_at = datetime('now') "
            "WHERE list_id = ? AND LOWER(TRIM(content)) = ? AND removed_at IS NULL",
            (1 if checked else 0, list_id, normalized_content),
        )
        return cursor.rowcount or 0

    @classmethod
    def update_fields(cls, item_id: str, list_id: str, fields: dict[str, object]) -> int:
        """Targeted UPDATE of arbitrary active columns plus ``updated_at`` for
        one item, scoped to its list. Returns the number of rows updated."""
        set_clauses = [f"{key} = ?" for key in fields]
        set_clauses.append("updated_at = datetime('now')")
        values: list[object] = list(fields.values())
        values.extend([item_id, list_id])
        cursor = cls._bound_connection().execute(
            f"UPDATE list_items SET {', '.join(set_clauses)} "
            "WHERE id = ? AND list_id = ? AND removed_at IS NULL",
            values,
        )
        return cursor.rowcount or 0

    @classmethod
    def soft_remove(cls, item_id: str, list_id: str) -> int:
        """Soft-remove one item by id, scoped to its list. Returns the number
        of rows updated (0 or 1)."""
        cursor = cls._bound_connection().execute(
            "UPDATE list_items SET removed_at = datetime('now'), updated_at = datetime('now') "
            "WHERE id = ? AND list_id = ? AND removed_at IS NULL",
            (item_id, list_id),
        )
        return cursor.rowcount or 0
