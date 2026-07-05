"""Late-binding query builder for active-record models (Critical 3).

``Query`` is bound to one model class and one connection at construction.
``filter``/``where``/``order_by``/``limit``/``offset`` only mutate the
builder's internal state and return ``self`` — no database I/O happens.
Only ``get``/``first``/``count``/``exists`` are terminal: they build the
parametrized SQL, execute it on the bound connection, and hydrate rows via
the owning model class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from models.model import Model

T = TypeVar("T", bound="Model")


class Query(Generic[T]):
    """Late-binding SQL builder bound to one model class and connection."""

    def __init__(self, model_class: type[T]) -> None:
        if not model_class.__table__:
            raise RuntimeError(f"{model_class.__name__} must set __table__ before querying")
        self._model_class = model_class
        self._predicates: list[str] = []
        self._values: list[object] = []
        self._order_by_column: str | None = None
        self._limit_count: int | None = None
        self._offset_count: int | None = None

    def filter(self, predicate: str, *values: object) -> "Query[T]":
        """Append a parametrized predicate, ANDed with any already present."""
        self._predicates.append(predicate)
        self._values.extend(values)
        return self

    def where(self, predicate: str, *values: object) -> "Query[T]":
        """Alias of :meth:`filter` for readability at the call site."""
        return self.filter(predicate, *values)

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

    def _select_sql(self) -> tuple[str, list[object]]:
        """Render the SELECT statement and its ordered bind-parameter list.
        ``LIMIT``/``OFFSET`` are bound as ``?`` params (never interpolated),
        appended after the predicate values in execution order."""
        sql = f"SELECT * FROM {self._model_class.__table__}{self._where_clause()}"
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
        sql = f"SELECT COUNT(*) FROM {self._model_class.__table__}{self._where_clause()}"
        cursor = self._model_class._bound_connection().execute(sql, self._values)
        row = cursor.fetchone()
        return int(row[0]) if row is not None else 0

    def exists(self) -> bool:
        """Whether the current predicate matches at least one row."""
        return self.count() > 0
