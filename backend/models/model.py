"""Active-record base for every model on the MP spine.

Holds no ``mp``, never imports a service, never reaches upstream (Rule-3
depth: a model is pure CRUD). Provides instance field storage from kwargs,
``to_dict``/``to_json`` projection, ``save``/``delete`` persistence, and the
late-binding query entry point reached on the model itself
(``Model.filter(...).limit(...).get()`` — Critical 3). There is no separate
``db.query(Model)`` path; this is the one query entry.

A zero-arg connection getter is bound once, at boot, by
``services.database.Database`` via :meth:`Model.bind`, and shared by every
subclass off the base class attribute. Every access calls back through the
getter so each thread re-derives its own connection (mirroring
``Database``'s thread-local caching) — subclasses never open, own, or
freeze a connection themselves.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from typing import ClassVar, Self, TypeVar

from contracts.has_table import HasTable
from models.query import Query
from models.serializable import Serializable

T = TypeVar("T", bound="Model")


class Model(Serializable, HasTable):
    """Base active-record row: field storage + CRUD + late-binding query.

    Subclasses :class:`~models.serializable.Serializable` for the one shared
    wire-encoding step (``to_json`` over a per-subclass ``to_dict``) so a
    persisted-row frame and a transient frame on the same socket can never
    encode a value two different ways (Essential 8); this class supplies only
    the column-filtered ``to_dict`` projection.

    Also explicitly subclasses :class:`~contracts.has_table.HasTable` — its
    abstract ``get_table()`` is left unimplemented here on purpose, so a
    subclass that forgets to override it stays abstract too (enforcement
    detail lives on the Protocol itself)."""

    # The real table columns, in schema order. A persisted subclass declares
    # this so persistence + projection are driven by the schema, not by whatever
    # attributes happen to sit on the instance: a service may attach transient
    # overlay fields (e.g. the WS envelope `type`/`turn_id` a `ToolCall` carries
    # for broadcast — §6.2) that are NOT columns, and those must never reach an
    # INSERT/UPDATE. Empty on the base; every row-model overrides it.
    __columns__: ClassVar[tuple[str, ...]] = ()
    _connection_getter: ClassVar[Callable[[], sqlite3.Connection] | None] = None

    def __init__(self, **fields: object) -> None:
        # int for INTEGER-autoincrement PKs (read back off cursor.lastrowid in
        # save()); str for TEXT-UUID PKs (episodes), whose model overrides
        # save() to generate the id itself.
        self.id: int | str | None = None
        for name, value in fields.items():
            setattr(self, name, value)

    @classmethod
    def bind(cls, connection_getter: Callable[[], sqlite3.Connection]) -> None:
        """Bind a zero-arg connection getter once, at boot, onto the base
        class — every subclass re-derives the calling thread's own
        connection through it on every access (Rule-3 depth), rather than
        freezing one thread's ``sqlite3.Connection`` onto the class."""
        Model._connection_getter = connection_getter

    @classmethod
    def _bound_connection(cls) -> sqlite3.Connection:
        """Call back into the getter bound by :meth:`bind` to derive the
        calling thread's own connection, or raise loudly if unbound."""
        if Model._connection_getter is None:
            raise RuntimeError(
                f"{cls.__name__}: no connection bound — Database.bind() must run at boot"
            )
        return Model._connection_getter()

    @classmethod
    def hydrate(cls: type[T], row: sqlite3.Row) -> T:
        """Build one instance from a fetched row."""
        return cls(**dict(row))

    @classmethod
    def filter(cls: type[T], key: str, value: object, operator: str = "=") -> Query[T]:
        """Start a late-binding query with one structured predicate
        (``key <operator> ?``); no DB I/O yet."""
        return Query(cls).filter(key, value, operator)

    @classmethod
    def where(cls: type[T], key: str, value: object, operator: str = "=") -> Query[T]:
        """Alias of :meth:`filter`."""
        return Query(cls).where(key, value, operator)

    @classmethod
    def filter_in(cls: type[T], key: str, values: Sequence[object]) -> Query[T]:
        """Start a late-binding query with a ``key IN (...)`` predicate; no DB
        I/O yet. An empty ``values`` matches nothing (see
        :meth:`~models.query.Query.filter_in`)."""
        return Query(cls).filter_in(key, values)

    @classmethod
    def _select_where_in_json(cls, column: str, values: Sequence[int], order_by: str | None = None) -> list[Self]:
        """Every row whose ``column`` is in ``values``, bound as one JSON-array
        parameter via ``json_each`` (large IN-lists exceed SQLite's bound-variable
        limit). Empty ``values`` is a clean no-op — no query runs."""
        if not values:
            return []
        safe_column = Query._validate_identifier(column)
        order = f" ORDER BY {Query._validate_identifier(order_by)}" if order_by else ""
        rows = cls._bound_connection().execute(
            f"SELECT * FROM {cls.get_table()} WHERE {safe_column} IN (SELECT value FROM json_each(?)){order}",
            (json.dumps(list(values)),),
        ).fetchall()
        return [cls.hydrate(row) for row in rows]

    @classmethod
    def _delete_where_in_json(cls, column: str, values: Sequence[int]) -> int:
        """Hard-delete every row whose ``column`` is in ``values``, bound as one
        JSON-array parameter via ``json_each``. Empty ``values`` is a clean
        no-op — no query runs. Returns rows deleted."""
        if not values:
            return 0
        safe_column = Query._validate_identifier(column)
        cursor = cls._bound_connection().execute(
            f"DELETE FROM {cls.get_table()} WHERE {safe_column} IN (SELECT value FROM json_each(?))",
            (json.dumps(list(values)),),
        )
        return cursor.rowcount or 0

    @classmethod
    def order_by(cls: type[T], terms: str) -> Query[T]:
        """Start a late-binding query ordered by ``terms`` — comma-separated
        ``column [ASC|DESC]`` entries (see :meth:`~models.query.Query.order_by`)."""
        return Query(cls).order_by(terms)

    @classmethod
    def all(cls: type[T]) -> Query[T]:
        """Start a late-binding query with no predicate."""
        return Query(cls)

    @classmethod
    def count(cls) -> int:
        """Bare ``SELECT COUNT(*)`` over the whole table, no predicate."""
        return Query(cls).count()

    @classmethod
    def get(cls: type[T], pk: object) -> T | None:
        """Fetch one row by primary key (the ``id`` column); ``None`` if no
        row matches. The single-row "fetchone by id" verb."""
        return Query(cls).filter("id", pk).first()

    @classmethod
    def first(cls: type[T]) -> T | None:
        """Whole-table convenience: the first row, or ``None`` if the table
        is empty. Delegates to :meth:`~models.query.Query.first`."""
        return Query(cls).first()

    @classmethod
    def exists(cls) -> bool:
        """Whole-table convenience: whether the table holds at least one
        row. Delegates to :meth:`~models.query.Query.exists`."""
        return Query(cls).exists()

    def _fields(self) -> list[str]:
        """The instance's currently-set values that are real table columns
        (``id`` first), derived live off ``__dict__`` ∩ ``__columns__``.

        Live off ``__dict__`` so a real-column attribute set after construction
        (``row.state = X; row.save()``) is picked up rather than frozen at
        ``__init__``; intersected with ``__columns__`` so a transient overlay
        attribute (a WS envelope field that is not a column, §6.2) never leaks
        into ``to_dict``/``save``. Columns left unset are omitted so their SQL
        defaults fire on INSERT."""
        return [
            name
            for name in self.__dict__
            if not name.startswith("_") and name in self.__columns__
        ]

    def to_dict(self) -> dict[str, object]:
        """Project every stored field (including ``id``) to a plain dict —
        column-filtered off ``__columns__`` (:class:`Serializable` renders it
        to JSON)."""
        return {name: getattr(self, name) for name in self._fields()}

    def save(self) -> Self:
        """INSERT a new row, or UPDATE the existing one, on the bound
        connection. Never commits: the connection autocommits each statement,
        and ``Database.transaction()`` groups multi-write atomic blocks
        (I6 — ``Database`` owns transaction/commit)."""
        connection = self._bound_connection()
        table = self.get_table()
        columns = [name for name in self._fields() if name != "id"]
        values = [getattr(self, name) for name in columns]
        if self.id is None:
            column_list = ", ".join(columns)
            placeholders = ", ".join("?" for _ in columns)
            cursor = connection.execute(
                f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})",
                values,
            )
            self.id = cursor.lastrowid
        else:
            assignments = ", ".join(f"{name} = ?" for name in columns)
            connection.execute(
                f"UPDATE {table} SET {assignments} WHERE id = ?",
                [*values, self.id],
            )
        return self

    def delete(self) -> None:
        """DELETE this row on the bound connection, if it has been persisted.
        Never commits (see :meth:`save`)."""
        if self.id is None:
            return
        connection = self._bound_connection()
        connection.execute(f"DELETE FROM {self.get_table()} WHERE id = ?", (self.id,))
        self.id = None
