"""List — one ``lists`` row: a named, id-addressed checklist/list container.

Active-record row-model (Rule 5 / §4.1). The table's primary key is a TEXT id
(8-char hex, generated the same way ``list_service`` always has — via
``secrets.token_hex(4)``), so :meth:`save` overrides the base's
autoincrement-``lastrowid`` INSERT the same way ``Episode.save`` does, minus
the paired FTS insert (lists carry no FTS index). This model is the SOLE home
of ``lists`` + ``lists_vec`` SQL; ``ListService`` reads and writes exclusively
through it (and through :class:`~models.list_item.ListItem` for items). Holds
no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

import secrets
import sqlite3
from typing import ClassVar, Self

from models.model import Model
from models.query import Query
from services._vec_upsert import vec0_upsert


class List(Model):
    """One ``lists`` row: field storage + CRUD + the name-collision lookup,
    counted-summary read, targeted updates, and the ``lists_vec`` embedding write."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "name", "list_type", "metadata", "created_at", "updated_at", "deleted_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "lists"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    name: str
    list_type: str
    metadata: str
    created_at: str
    updated_at: str
    deleted_at: str | None

    def save(self) -> Self:
        """INSERT a new list (TEXT-hex PK generated here), or UPDATE the
        existing row — mirrors ``Episode.save``'s id-generation override."""
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

    # ── Live scope + id/name reads ─────────────────────────────────────────

    @classmethod
    def active(cls) -> Query[Self]:
        """The live-list scope — ``deleted_at IS NULL``. The one shared
        predicate every non-deleted read builds on."""
        return cls.filter("deleted_at", None, "IS")

    @classmethod
    def by_id(cls, list_id: str) -> Self | None:
        """The single active list for an exact id, or ``None``."""
        return cls.active().filter("id", list_id).first()

    @classmethod
    def find_by_name(cls, name: str) -> Self | None:
        """The single active list whose name matches case-insensitively, or
        ``None`` — the create/rename collision check. Raw SQL: a
        case-insensitive ``LOWER()`` comparison can't be expressed by the
        structured filter builder (its column check only allows a bare
        identifier, not a SQL function over it)."""
        row = cls._bound_connection().execute(
            "SELECT * FROM lists WHERE LOWER(name) = LOWER(?) AND deleted_at IS NULL LIMIT 1",
            (name,),
        ).fetchone()
        return cls.hydrate(row) if row is not None else None

    @classmethod
    def with_counts(cls) -> list[sqlite3.Row]:
        """Every active list plus its live/checked item counts, most-recently-
        updated first. Raw SQL: a LEFT JOIN + conditional aggregate the
        structured builder can't express."""
        cursor = cls._bound_connection().execute(
            "SELECT "
            "    l.id, l.name, l.list_type, l.created_at, l.updated_at, "
            "    SUM(CASE WHEN li.removed_at IS NULL AND li.id IS NOT NULL THEN 1 ELSE 0 END) AS item_count, "
            "    SUM(CASE WHEN li.removed_at IS NULL AND li.checked THEN 1 ELSE 0 END) AS checked_count "
            "FROM lists l "
            "LEFT JOIN list_items li ON li.list_id = l.id "
            "WHERE l.deleted_at IS NULL "
            "GROUP BY l.id, l.name, l.list_type, l.created_at, l.updated_at "
            "ORDER BY l.updated_at DESC"
        )
        return cursor.fetchall()

    # ── Targeted writes ──────────────────────────────────────────────────────

    @classmethod
    def update_fields(cls, list_id: str, fields: dict[str, object]) -> int:
        """Targeted UPDATE of arbitrary active columns plus ``updated_at``.
        Returns the number of rows updated (0 if the list is missing/deleted)."""
        set_clauses = [f"{key} = ?" for key in fields]
        set_clauses.append("updated_at = datetime('now')")
        values: list[object] = list(fields.values())
        values.append(list_id)
        cursor = cls._bound_connection().execute(
            f"UPDATE lists SET {', '.join(set_clauses)} WHERE id = ? AND deleted_at IS NULL",
            values,
        )
        return cursor.rowcount or 0

    @classmethod
    def soft_delete(cls, list_id: str) -> int:
        """Mark a list deleted (idempotent — already-deleted rows unchanged).
        Returns the number of rows updated."""
        cursor = cls._bound_connection().execute(
            "UPDATE lists SET deleted_at = datetime('now'), updated_at = datetime('now') "
            "WHERE id = ? AND deleted_at IS NULL",
            (list_id,),
        )
        return cursor.rowcount or 0

    @classmethod
    def touch(cls, list_id: str) -> None:
        """Stamp ``updated_at`` unconditionally — bumps a list's position in
        the most-recently-touched ordering after an item write."""
        cls._bound_connection().execute(
            "UPDATE lists SET updated_at = datetime('now') WHERE id = ?",
            (list_id,),
        )

    @classmethod
    def set_embedding(cls, list_id: str, blob: bytes) -> None:
        """Upsert a list's embedding blob into ``lists_vec`` keyed by rowid. A
        missing list is a silent no-op."""
        connection = cls._bound_connection()
        row = connection.execute(
            "SELECT rowid FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        if row:
            vec0_upsert(connection, "lists_vec", row[0], blob)
