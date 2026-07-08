"""Regression guard for ``SystemRow.newest_active_by_key`` — the hardened
system-cursor read the subconscious worker uses for its ``pattern_match_cursor``
and ``geo_pattern_cursor`` watermarks.

Unlike ``active_by_key`` (the store lookup: ``active``-only, unordered), this
read adds ``deleted_at IS NULL`` and a deterministic ``ORDER BY id DESC`` so a
historical or concurrent UPSERT collision that left more than one active row for
a key resolves to the NEWEST row rather than SQLite's implementation-defined
order. That defensive ordering is the whole reason the method exists, so these
tests seed the collision directly and pin the winner.
"""

import sqlite3

import pytest

from models.system import SystemRow

pytestmark = pytest.mark.integration


def _seed(db: sqlite3.Connection, key: str, value: str, *, active: int = 1, deleted: bool = False) -> int:
    """Insert one ``system``-kind data_graph row and return its id."""
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active, deleted_at) "
        "VALUES ('system', ?, ?, ?, ?)",
        (key, value, active, "2025-01-01T00:00:00" if deleted else None),
    )
    return int(db.execute("SELECT last_insert_rowid()").fetchone()[0])


def test_newest_active_row_wins(db: sqlite3.Connection) -> None:
    """Two active rows for one key resolve to the newest (highest id) — the
    duplicate-active-collision defense the raw SQL's ORDER BY id DESC provided."""
    _seed(db, "pattern_match_cursor", "100")
    _seed(db, "pattern_match_cursor", "250")
    db.commit()

    row = SystemRow.newest_active_by_key("pattern_match_cursor")

    assert row is not None
    assert row.value == "250"


def test_deleted_and_inactive_rows_excluded(db: sqlite3.Connection) -> None:
    """A soft-deleted row (``deleted_at`` set) and an inactive row are both
    outside the live predicate, so a key holding only those reads as None."""
    _seed(db, "geo_pattern_cursor", "5", deleted=True)
    _seed(db, "geo_pattern_cursor", "9", active=0)
    db.commit()

    assert SystemRow.newest_active_by_key("geo_pattern_cursor") is None


def test_missing_key_returns_none(db: sqlite3.Connection) -> None:
    assert SystemRow.newest_active_by_key("never_written") is None
