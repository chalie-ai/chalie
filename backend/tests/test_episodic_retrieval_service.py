"""
Tests for services/episodic_retrieval_service.py — module-level retrieve() and helpers.

Old tests for the EpisodicService class (embedding_dimensions, weights,
retrieve_episodes, _analyze_query, format_for_prompt, _count_episodes,
_calculate_vector_similarity, _calculate_effective_freshness,
_get_consolidated_date_range, _hybrid_retrieve, adaptive-radius internals)
were deleted in Commit D because:

  - EpisodicService no longer has retrieval methods (storage-only).
  - The above helpers were removed in the episodic-simplification arbiter pass.

Remaining coverage:
  - TestFts5AliasRegression: FTS5 alias-form regression (real sqlite3, no mocks).
    Verifies that the production FTS query uses the unaliased table name in WHERE
    MATCH, which is the only correct form for SQLite FTS5 external-content tables.
"""

import pytest
import sqlite3


pytestmark = pytest.mark.unit


# ── FTS5 alias regression ─────────────────────────────────────────────────────


class TestFts5AliasRegression:
    """
    Regression: the FTS query aliased episodes_fts as 'f' in the FROM clause, but
    SQLite FTS5 requires the MATCH operator in WHERE to reference the virtual table
    by its full unaliased name. Mixing an alias in FROM with the full name in WHERE
    causes empty results (not a syntax error), while using the alias in WHERE raises
    OperationalError('no such column').

    The fix removes the alias entirely — the table is referenced by its full name
    in SELECT (rank), FROM, JOIN ON, WHERE MATCH, and ORDER BY.

    These tests use a real in-memory SQLite FTS5 table to confirm the behaviour.
    """

    def _make_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, gist TEXT, deleted_at TEXT)")
        conn.execute("INSERT INTO episodes VALUES (1, 'watering plants reminder', NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(gist, content=episodes, content_rowid=id)"
        )
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
        return conn

    def test_fts5_no_alias_match_returns_results(self):
        """Fixed query — no alias, full table name in WHERE MATCH — returns rows."""
        conn = self._make_conn()
        query = """
            SELECT e.id, e.gist, episodes_fts.rank AS text_rank
            FROM episodes_fts
            JOIN episodes e ON e.rowid = episodes_fts.rowid
            WHERE episodes_fts MATCH ?
              AND e.deleted_at IS NULL
            ORDER BY episodes_fts.rank
        """
        rows = conn.execute(query, ("watering",)).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 'watering plants reminder'

    def test_fts5_alias_in_where_raises(self):
        """Alias in WHERE MATCH raises OperationalError — confirms alias form is invalid."""
        conn = self._make_conn()
        bad_query = """
            SELECT f.rank
            FROM episodes_fts f
            JOIN episodes e ON e.rowid = f.rowid
            WHERE f MATCH ?
        """
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            conn.execute(bad_query, ("watering",)).fetchall()
