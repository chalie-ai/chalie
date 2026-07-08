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
from typing import ClassVar, cast

from models.model import Model
from models.thread_gist import ThreadGist
from services.database import Database


class ScheduledItem(Model):
    """One ``scheduled_items`` row: field storage + CRUD, the ordered/limited
    list read, the fuzzy message lookup, the vec0 nearest-neighbour search,
    the paired vec0 embedding delete a hard cancel must ride with, and the 1-1
    thread gist a schedule is born with."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "message", "start_at",
        "cron_dom", "cron_hour", "cron_minute",
        "enabled", "channel", "created_by_session", "created_at",
    )

    # The channel a schedule's thread lives on: its integer ``id`` IS the
    # ``turn_id`` on this channel (§13.1), so its :class:`ThreadGist` is keyed
    # ``(SCHEDULE_CHANNEL, id)``. Distinct from the ``channel`` column, which
    # records the channel a schedule was *created from*.
    SCHEDULE_CHANNEL: ClassVar[str] = "schedule"

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
    def recent(
        cls, *, limit: int | None = None, offset: int = 0
    ) -> list[ScheduledItem]:
        """Live schedules newest-first (``created_at`` DESC) as hydrated
        instances — the REST list read (paged via ``limit``/``offset``) and the
        REST turns read (unpaged, ``limit=None``). Builder-only: ORDER BY +
        LIMIT/OFFSET are all expressible by the structured filter. ``offset`` is
        applied only alongside a ``limit`` (SQLite OFFSET requires LIMIT)."""
        query = cls.order_by("created_at DESC")
        if limit is not None:
            query = query.limit(limit).offset(offset)
        return query.get()

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

    def get_gist(self) -> ThreadGist | None:
        """This schedule's thread label — the :class:`ThreadGist` keyed by
        ``(SCHEDULE_CHANNEL, id)``. Seeded from the prompt at :meth:`create`
        time, so it is present from birth; no join, a single-row read on
        ThreadGist's own table (``turn_id == id`` on the schedule channel)."""
        return ThreadGist.for_turn(self.SCHEDULE_CHANNEL, cast(int, self.id))

    # ── Writes ──────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, **fields: object) -> ScheduledItem:
        """Insert a new schedule AND seed its 1-1 thread gist from the prompt,
        atomically — the sole birth path for a schedule (both the REST create
        and the ability's ``create``/``update`` route through here).

        The gist is the schedule's own ``message``, written synchronously so a
        schedule thread carries a label the instant it exists — no waiting for
        the fork-time async LLM ingest (``MessageProcessor._maybe_fire_gist``,
        which only fires when *no* gist exists, so this seed naturally
        suppresses it). Grouped under one ``Database.transaction()`` so a gist
        failure rolls the row back rather than leaving a labelless schedule."""
        item = cls(**fields)
        with Database.transaction():
            item.save()  # INSERT; id autoincrements, captured onto item.id
            item.seed_gist()
        return item

    def seed_gist(self) -> None:
        """Write this schedule's thread gist from its own ``message`` (upsert,
        idempotent per turn). Called at :meth:`create` time inside its
        transaction; ``turn_id == id`` on the schedule channel."""
        ThreadGist(
            channel=self.SCHEDULE_CHANNEL, turn_id=cast(int, self.id), gist=self.message
        ).upsert()

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
