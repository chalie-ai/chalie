"""WatchedFolder — one ``watched_folders`` row.

Active-record row-model (Rule 5 / §4.1). ``id`` is a TEXT id (8-char hex,
generated via ``secrets.token_hex(4)`` in :meth:`save`), so :meth:`save`
overrides the base's autoincrement-``lastrowid`` INSERT. This model is the
SOLE home of ``watched_folders`` SQL; ``FolderWatcherService`` reads and
writes exclusively through it. Holds no mp, calls no service (Rule-3 depth).
"""

from __future__ import annotations

import json
import secrets
from typing import ClassVar, Self

from models.model import Model


class WatchedFolder(Model):
    """One ``watched_folders`` row: field storage + CRUD + scan-state updates."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "folder_path", "label", "source_type", "enabled",
        "file_patterns", "ignore_patterns", "recursive",
        "scan_interval", "last_scan_at", "last_scan_files",
        "last_scan_error", "source_config", "created_at", "updated_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "watched_folders"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    id: int | str | None
    folder_path: str
    label: str | None
    source_type: str
    enabled: int
    file_patterns: str
    ignore_patterns: str
    recursive: int
    scan_interval: int
    last_scan_at: str | None
    last_scan_files: int | None
    last_scan_error: str | None
    source_config: str
    created_at: str
    updated_at: str

    def save(self) -> Self:
        """INSERT a new watched folder (TEXT-hex PK generated here), or
        UPDATE the existing row — mirrors ``List.save``'s id-generation
        override for TEXT-hex PKs."""
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

    # ── Reads ────────────────────────────────────────────────────────────────

    @classmethod
    def all_ordered(cls) -> list[Self]:
        """Every watched folder, newest first."""
        return cls.order_by("created_at").get()

    @classmethod
    def enabled_ordered(cls) -> list[Self]:
        """Only enabled watched folders, newest first."""
        return cls.filter("enabled", 1).order_by("created_at").get()

    def to_dict(self) -> dict[str, object]:
        """Project to a dict with JSON fields parsed (file_patterns,
        ignore_patterns, source_config)."""
        d = super().to_dict()
        for field in ('file_patterns', 'ignore_patterns', 'source_config'):
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])  # type: ignore[arg-type]
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    # ── Writes ──────────────────────────────────────────────────────────────

    @classmethod
    def update_fields(cls, folder_id: str, fields: dict[str, object]) -> int:
        """Targeted UPDATE of arbitrary columns plus ``updated_at``.
        Returns the number of rows updated (0 if the folder is missing).
        Mirrors ``List.update_fields``'s dynamic-field pattern."""
        set_clauses = [f"{key} = ?" for key in fields]
        set_clauses.append("updated_at = datetime('now')")
        values: list[object] = list(fields.values())
        values.append(folder_id)
        cursor = cls._bound_connection().execute(
            f"UPDATE watched_folders SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        return cursor.rowcount or 0

    @classmethod
    def update_scan_stats(cls, folder_id: str, file_count: int) -> None:
        """Stamp scan completion: last_scan_at, last_scan_files, clear error,
        updated_at. Raw SQL: ``datetime('now')`` can't be bound as a ?
        parameter."""
        cls._bound_connection().execute(
            "UPDATE watched_folders "
            "SET last_scan_at = datetime('now'), last_scan_files = ?, "
            "last_scan_error = NULL, updated_at = datetime('now') "
            "WHERE id = ?",
            (file_count, folder_id),
        )

    @classmethod
    def update_scan_error(cls, folder_id: str, error: str) -> None:
        """Record a scan error message and stamp updated_at. Raw SQL:
        ``datetime('now')`` can't be bound as a ? parameter."""
        cls._bound_connection().execute(
            "UPDATE watched_folders "
            "SET last_scan_error = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (error, folder_id),
        )
