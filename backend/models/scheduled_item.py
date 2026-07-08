"""ScheduledItem — one ``scheduled_items`` row: a prompt-only, dumb-cron
reminder/recurring-prompt.

Active-record row-model (Rule 5 / §4.1). ``id`` is the DDL's own
``INTEGER PRIMARY KEY AUTOINCREMENT`` (also the thread's ``turn_id`` on the
``'schedule'`` channel, §13.1 — a cancelled schedule's id is never reissued),
so the base's ``save``/``get``/``delete`` id-centric verbs apply unmodified;
no id-generation override is needed here (contrast
:meth:`~models.list.List.save`'s TEXT-hex PK). This model is the SOLE home of
``scheduled_items`` + ``scheduled_items_vec`` SQL; :mod:`abilities.schedule`
reads and writes exclusively through it. Holds no mp, calls no service
(Rule-3 depth).

There is no lifecycle to filter on — no status/hidden/item_type/group_id/
due_at columns exist (see the ability's module docstring): every row is a
live schedule until it is hard-deleted, and a poller elsewhere matches
``enabled``/``start_at``/``cron_*`` against the current wall-clock minute."""

from __future__ import annotations

import sqlite3
from typing import ClassVar

from models.model import Model
from services.database import Database


class ScheduledItem(Model):
    """One ``scheduled_items`` row: field storage + CRUD, the ordered/limited
    list read, the fuzzy message lookup, the vec0 nearest-neighbour search,
    and the paired vec0 embedding delete a hard cancel must ride with."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "message", "start_at",
        "cron_dom", "cron_hour", "cron_minute",
        "enabled", "channel", "created_by_session", "created_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "scheduled_items"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    message: str
    start_at: str
    cron_dom: int | None
    cron_hour: int | None
    cron_minute: int | None
    enabled: int
    channel: str | None
    created_by_session: str | None
    created_at: str

    # ── Reads ────────────────────────────────────────────────────────────────

    @classmethod
    def by_start_at(
        cls, columns: tuple[str, ...], *, limit: int | None = None
    ) -> list[dict[str, object]]:
        """Every row (optionally capped at ``limit``) as plain per-row dicts
        restricted to ``columns``, soonest ``start_at`` first — the single
        read path for the ``list`` action. Builder-only: ORDER BY + LIMIT are
        both expressible by the structured filter."""
        query = cls.order_by("start_at")
        if limit is not None:
            query = query.limit(limit)
        return query.select(*columns)

    @classmethod
    def search_by_message(cls, pattern: str) -> list[dict[str, object]]:
        """Fuzzy ``LIKE`` lookup by content, oldest first — the cancel/enable/
        disable target resolver when ``item_id`` is unknown. Builder-only:
        LIKE + ORDER BY are both expressible by the structured filter."""
        return (
            cls.filter("message", pattern, "LIKE")
            .order_by("created_at")
            .select("id", "message")
        )

    @classmethod
    def due_at(cls, now_iso: str) -> list[dict[str, object]]:
        """Enabled, already-started rows the poller tests against the current
        minute — ``id``/``message`` plus the ``cron_*`` fields ``matches()``
        needs. A LOCKLESS builder read (never ``Database.transaction()``'s
        ``BEGIN IMMEDIATE``): the poll must not take the write lock, or a
        contended minute could stall past ``busy_timeout`` and be skipped —
        and a fixed-time schedule due in that minute is then missed until its
        next natural occurrence. ``_bound_connection()`` is the autocommit
        ``Database.conn()`` handle, so a bare ``SELECT`` reads without locking."""
        return (
            cls.filter("enabled", 1)
            .filter("start_at", now_iso, "<=")
            .select("id", "message", "cron_dom", "cron_hour", "cron_minute")
        )

    @classmethod
    def vector_search(cls, embedding_blob: bytes, limit: int) -> list[sqlite3.Row]:
        """Nearest-neighbour lookup over ``scheduled_items_vec`` joined back
        to the row it embeds, closest first. Raw SQL: a vec0 ``MATCH`` join
        can't be expressed by the structured filter builder — rowid == id
        under ``INTEGER PRIMARY KEY``."""
        cursor = cls._bound_connection().execute(
            """
            SELECT s.id, s.message, s.start_at,
                   s.cron_dom, s.cron_hour, s.cron_minute, v.distance
            FROM scheduled_items_vec v
            JOIN scheduled_items s ON s.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (embedding_blob, limit),
        )
        return cursor.fetchall()

    # ── Writes ──────────────────────────────────────────────────────────────

    @classmethod
    def write_embedding(cls, item_id: int, embedding_blob: bytes) -> None:
        """Upsert this item's vec0 embedding, skipping a row that has since
        been cancelled (rowid == id under ``INTEGER PRIMARY KEY``) so a stale
        embedding is never written for a gone schedule. Best-effort companion
        to :meth:`delete_embedding`. Raw SQL grouped under its own
        ``Database.transaction()``: the vec0 shadow table has no model, and the
        existence guard + upsert commit as one."""
        with Database.transaction() as conn:
            if conn.execute(
                "SELECT 1 FROM scheduled_items WHERE id = ?", (item_id,)
            ).fetchone() is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO scheduled_items_vec (rowid, embedding) VALUES (?, ?)",
                (item_id, embedding_blob),
            )

    @classmethod
    def delete_embedding(cls, item_id: int) -> None:
        """Drop this item's row from ``scheduled_items_vec``. Must ride in
        the same transaction as a hard delete of the main row (rowid == id
        under ``INTEGER PRIMARY KEY``) so a cancelled schedule never orphans
        its embedding — the caller groups both under
        ``Database.transaction()``. Raw SQL: the vec0 shadow table has no
        model of its own."""
        cls._bound_connection().execute(
            "DELETE FROM scheduled_items_vec WHERE rowid = ?", (item_id,)
        )
