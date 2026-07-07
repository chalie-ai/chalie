"""Late-binding query builder for active-record models (Critical 3).

``Query`` is bound to one model class and one connection at construction.
``filter``/``where``/``filter_in``/``order_by``/``limit``/``offset`` only
mutate the builder's internal state and return ``self`` — no database I/O
happens. Only ``get``/``first``/``count``/``exists``/``pluck`` are terminal:
they build the parametrized SQL, execute it on the bound connection, and
(except ``pluck``) hydrate rows via the owning model class.

``filter``/``where``/``filter_in`` never accept a raw predicate string: the
value is always bound as a ``?`` parameter, the operator is checked against a
fixed whitelist, and the column name is checked against a safe-identifier
pattern — so a caller can never smuggle SQL through either argument. A
predicate this builder genuinely cannot express (a join, an OR across
columns, an UPSERT, FTS/vec0 MATCH, DDL) stays hand-rolled SQL on the owning
model, same as before.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

if TYPE_CHECKING:
    from models.model import Model

T = TypeVar("T", bound="Model")

#: Safe SQL identifier — a bare column name, no dots/quotes/whitespace/SQL
#: keywords smuggled in. Keys are checked against this before ever reaching
#: an f-string, since (unlike the value) a column name can't be bound as a
#: ``?`` parameter.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Query(Generic[T]):
    """Late-binding SQL builder bound to one model class and connection."""

    #: The only comparison operators a structured predicate may use — every
    #: other token (including anything that could inject extra SQL) is
    #: rejected loudly by :meth:`filter`.
    _ALLOWED_OPERATORS: ClassVar[frozenset[str]] = frozenset(
        {"=", "!=", "<", "<=", ">", ">=", "LIKE", "IS", "IS NOT"}
    )

    #: A predicate that is always false and binds no parameters — the
    #: :meth:`filter_in` answer to an empty ``values`` sequence, so
    #: "IN ()" (invalid SQL) never gets generated.
    _NEVER_MATCHES: ClassVar[str] = "1 = 0"

    def __init__(self, model_class: type[T]) -> None:
        table = model_class.get_table()
        if not table:
            raise RuntimeError(f"{model_class.__name__} must implement get_table() before querying")
        self._model_class = model_class
        self._predicates: list[str] = []
        self._values: list[object] = []
        self._order_by_column: str | None = None
        self._limit_count: int | None = None
        self._offset_count: int | None = None

    @classmethod
    def _validate_identifier(cls, key: str) -> str:
        """Loudly reject anything that isn't a bare column name — a key can't
        be bound as a ``?`` parameter, so it must be proven safe before it
        ever reaches an f-string."""
        if not _IDENTIFIER_RE.match(key):
            raise ValueError(f"unsafe column identifier: {key!r}")
        return key

    @classmethod
    def _validate_operator(cls, operator: str) -> str:
        """Loudly reject any operator outside the fixed whitelist."""
        if operator not in cls._ALLOWED_OPERATORS:
            raise ValueError(
                f"unsupported filter operator: {operator!r} "
                f"(allowed: {sorted(cls._ALLOWED_OPERATORS)})"
            )
        return operator

    def filter(self, key: str, value: object, operator: str = "=") -> "Query[T]":
        """Append one structured predicate (``key <operator> ?``), ANDed with
        any already present. ``value`` is always bound as a parameter —
        including ``None`` for ``IS``/``IS NOT`` (SQLite treats a bound
        ``NULL`` parameter as equivalent to the ``NULL`` literal for those
        two operators)."""
        column = self._validate_identifier(key)
        op = self._validate_operator(operator)
        self._predicates.append(f"{column} {op} ?")
        self._values.append(value)
        return self

    def where(self, key: str, value: object, operator: str = "=") -> "Query[T]":
        """Alias of :meth:`filter` for readability at the call site."""
        return self.filter(key, value, operator)

    def filter_in(self, key: str, values: Sequence[object]) -> "Query[T]":
        """Append a ``key IN (?, ?, ...)`` predicate, ANDed with any already
        present. An empty ``values`` makes the WHOLE query match nothing
        (``get`` → ``[]``, ``count`` → 0, ``exists`` → False) rather than
        emitting invalid ``IN ()`` SQL or raising."""
        column = self._validate_identifier(key)
        if not values:
            self._predicates.append(self._NEVER_MATCHES)
            return self
        placeholders = ", ".join("?" for _ in values)
        self._predicates.append(f"{column} IN ({placeholders})")
        self._values.extend(values)
        return self

    def order_by(self, column: str) -> "Query[T]":
        """Set the ORDER BY clause."""
        self._order_by_column = column
        return self

    def limit(self, count: int) -> "Query[T]":
        """Set the LIMIT clause."""
        self._limit_count = count
        return self

    def offset(self, count: int) -> "Query[T]":
        """Set the OFFSET clause."""
        self._offset_count = count
        return self

    def _where_clause(self) -> str:
        """Render the WHERE clause, or an empty string when no predicate was set."""
        if not self._predicates:
            return ""
        return " WHERE " + " AND ".join(self._predicates)

    def _select_sql(self, columns: str = "*") -> tuple[str, list[object]]:
        """Render the SELECT statement and its ordered bind-parameter list.
        ``LIMIT``/``OFFSET`` are bound as ``?`` params (never interpolated),
        appended after the predicate values in execution order."""
        sql = f"SELECT {columns} FROM {self._model_class.get_table()}{self._where_clause()}"
        values = list(self._values)
        if self._order_by_column:
            sql += f" ORDER BY {self._order_by_column}"
        if self._limit_count is not None:
            sql += " LIMIT ?"
            values.append(self._limit_count)
        if self._offset_count is not None:
            sql += " OFFSET ?"
            values.append(self._offset_count)
        return sql, values

    def get(self) -> list[T]:
        """Execute the built SELECT and hydrate every row into a model instance.

        The connection is resolved here, lazily: chaining
        ``filter``/``order_by``/``limit`` performs zero I/O (Critical 3)."""
        sql, values = self._select_sql()
        cursor = self._model_class._bound_connection().execute(sql, values)
        return [self._model_class.hydrate(row) for row in cursor.fetchall()]

    def first(self) -> T | None:
        """Execute the built SELECT bounded to one row; ``None`` if no match.

        A pure terminal read (Essential 7) — it does not permanently mutate
        this builder's ``LIMIT`` state, so the same builder can still be
        reused afterward for its true, unbounded result set.
        """
        saved_limit_count = self._limit_count
        try:
            self._limit_count = 1
            rows = self.get()
        finally:
            self._limit_count = saved_limit_count
        return rows[0] if rows else None

    def count(self) -> int:
        """Execute a COUNT(*) over the current predicate (no LIMIT/OFFSET).
        Resolves the connection lazily, like :meth:`get`."""
        sql = f"SELECT COUNT(*) FROM {self._model_class.get_table()}{self._where_clause()}"
        cursor = self._model_class._bound_connection().execute(sql, self._values)
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def exists(self) -> bool:
        """Whether the current predicate matches at least one row."""
        return self.count() > 0

    def pluck(self, column: str) -> list[object]:
        """Single-column projection terminal: the current predicate's
        ``column`` values as a plain list, with no model hydration
        (LIMIT/OFFSET/ORDER BY still apply). Resolves the connection lazily,
        like :meth:`get`."""
        safe_column = self._validate_identifier(column)
        sql, values = self._select_sql(columns=safe_column)
        cursor = self._model_class._bound_connection().execute(sql, values)
        return [row[0] for row in cursor.fetchall()]

    def select(self, *columns: str) -> list[dict[str, object]]:
        """Multi-column projection terminal: the current predicate's rows as
        plain per-row dicts restricted to ``columns``, with no model
        hydration (LIMIT/OFFSET/ORDER BY still apply). Every column is
        identifier-validated, same as :meth:`pluck`. Resolves the connection
        lazily, like :meth:`get`."""
        safe_columns = [self._validate_identifier(column) for column in columns]
        sql, values = self._select_sql(columns=", ".join(safe_columns))
        cursor = self._model_class._bound_connection().execute(sql, values)
        return [dict(row) for row in cursor.fetchall()]

    def update(self, **fields: object) -> int:
        """Execute ``UPDATE {table} SET col = ?, ... WHERE <predicate>``
        against the current predicate and return the affected row count.

        Refuses loudly when no predicate has been set — a predicate-less
        bulk UPDATE must never happen through this path, even via
        ``Model.all()``. Never commits: the connection autocommits each
        statement, and ``Database.transaction()`` groups multi-write atomic
        blocks (I6 — ``Database`` owns transaction/commit), same as
        :meth:`~models.model.Model.save`."""
        if not self._predicates:
            raise RuntimeError(
                f"{self._model_class.__name__}: refusing an UPDATE with no predicate — "
                "narrow with filter()/where() first"
            )
        if not fields:
            raise ValueError("update() requires at least one field")
        assignments = ", ".join(
            f"{self._validate_identifier(column)} = ?" for column in fields
        )
        sql = f"UPDATE {self._model_class.get_table()} SET {assignments}{self._where_clause()}"
        values = [*fields.values(), *self._values]
        cursor = self._model_class._bound_connection().execute(sql, values)
        return cursor.rowcount

    def delete(self) -> int:
        """Execute ``DELETE FROM {table} WHERE <predicate>`` against the
        current predicate and return the affected row count.

        Refuses loudly when no predicate has been set, same as :meth:`update`.
        Never commits (see :meth:`update`)."""
        if not self._predicates:
            raise RuntimeError(
                f"{self._model_class.__name__}: refusing a DELETE with no predicate — "
                "narrow with filter()/where() first"
            )
        sql = f"DELETE FROM {self._model_class.get_table()}{self._where_clause()}"
        cursor = self._model_class._bound_connection().execute(sql, self._values)
        return cursor.rowcount
