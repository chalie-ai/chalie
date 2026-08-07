"""Episode FTS indexing through SearchExpanderService — real stack, no mocks.

Episodes route their full-text posting through the same declaration-driven async
engine ``data_graph`` uses. These tests drive the REAL pipeline synchronously
against the bound test DB (in production the ``search_expander_worker`` daemon
does this continuously; under pytest no worker runs, so a test drains it
explicitly). They prove:

* ``Episode.save`` no longer writes ``episodes_fts`` inline — the posting is
  deferred to the engine (``indexed_at`` stays NULL until drained).
* Draining indexes the row: ``indexed_at`` becomes non-NULL and the gist is
  FTS-searchable.
* The invariant holds — a second drain does not double-post (self-heal skips
  already-indexed rows; a forced re-process deletes before re-inserting).
* ``hard_delete`` tears down both FTS columns and the vec row.
* ``data_graph`` indexing is unchanged by the generalization (regression guard).
"""

import re
import sqlite3

import pytest

from models.episode import Episode
from models.fact import FactRow
from services.embedding_utils import pack_embedding
from services.episodic_service import EpisodicService

pytestmark = pytest.mark.unit


def _drain_search_index() -> None:
    """Drive the real async search-expander pipeline synchronously against the
    bound test DB — the exact production code path, no mocks."""
    from services.search_expander_service import SearchExpanderService
    svc = SearchExpanderService()
    svc._self_heal()
    item = svc._dequeue()
    while item is not None:
        svc._process(item)
        item = svc._dequeue()


def _vec_dim(conn: sqlite3.Connection, table: str) -> int:
    """The live vec0 column dimension, read off the schema so a synthetic
    embedding always matches the width the table was declared with."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    match = re.search(r"float\[(\d+)\]", row[0]) if row else None
    return int(match.group(1)) if match else 768


def _store_episode(gist: str = "a lively chat about migrating databases",
                   channel: str = "programming", salience: int = 5) -> str:
    """Store one episode through the real service and return its id."""
    return EpisodicService().store_episode(
        {"gist": gist, "salience": salience, "channel": channel}
    )


def _fts_rowids(db: sqlite3.Connection, term: str) -> list[int]:
    """rowids matching ``term`` in episodes_fts (MATCH spans every FTS column)."""
    return [
        r[0] for r in db.execute(
            "SELECT rowid FROM episodes_fts WHERE episodes_fts MATCH ?", (f'"{term}"',)
        ).fetchall()
    ]


class TestSaveDefersFtsToEngine:

    def test_save_leaves_indexed_at_null_and_no_inline_posting(
        self, db: sqlite3.Connection
    ) -> None:
        """A freshly stored episode is not yet indexed: indexed_at is NULL
        and no episodes_fts posting exists until the engine drains it."""
        episode_id = _store_episode(gist="quantum widgets and flux capacitors")

        row = db.execute(
            "SELECT indexed_at FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        assert row[0] is None

        assert _fts_rowids(db, "flux") == []
        assert Episode.fts_search("flux", None, 10) == []


class TestDrainIndexesEpisode:

    def test_drain_indexes_gist_and_stamps_indexed_at(
        self, db: sqlite3.Connection
    ) -> None:
        """Draining the engine writes the FTS posting and marks the row indexed
        (indexed_at non-NULL) so self-heal won't re-enqueue it."""
        episode_id = _store_episode(gist="the aardvark reviewed the schema")

        _drain_search_index()

        row = db.execute(
            "SELECT rowid, indexed_at FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        rowid, indexed_at = row[0], row[1]
        assert indexed_at is not None

        assert rowid in _fts_rowids(db, "aardvark")
        hits = Episode.fts_search("aardvark", None, 10)
        assert [e.id for e in hits] == [episode_id]


class TestNoDoublePosting:

    def test_second_drain_is_idempotent(self, db: sqlite3.Connection) -> None:
        """A second drain skips the already-indexed row (indexed_at non-NULL),
        so the gist keeps exactly one posting — no duplicate."""
        episode_id = _store_episode(gist="the pangolin optimized a query")
        _drain_search_index()
        _drain_search_index()

        rowid = db.execute(
            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()[0]
        assert _fts_rowids(db, "pangolin") == [rowid]

    def test_forced_reprocess_deletes_before_reinserting(
        self, db: sqlite3.Connection
    ) -> None:
        """Re-enqueuing an already-indexed row (prior indexed_at non-NULL)
        exercises the delete-before-insert guard: the posting is replaced, never
        duplicated."""
        from services.search_expander_service import SearchExpanderService

        episode_id = _store_episode(gist="the ocelot refactored a module")
        _drain_search_index()
        rowid = db.execute(
            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()[0]

        svc = SearchExpanderService()
        svc._enqueue("episodes", rowid)
        item = svc._dequeue()
        while item is not None:
            svc._process(item)
            item = svc._dequeue()

        assert _fts_rowids(db, "ocelot") == [rowid]


class TestHardDeleteTeardown:

    def test_hard_delete_removes_fts_and_vec(self, db: sqlite3.Connection) -> None:
        """hard_delete removes the episode, its FTS posting (both columns), and
        its vec row — no orphaned index entries."""
        episode_id = _store_episode(gist="the marmot benchmarked the cache")
        _drain_search_index()
        rowid = db.execute(
            "SELECT rowid FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()[0]

        # Seed a vec row at the live dimension so teardown has something to remove.
        blob = pack_embedding([0.1] * _vec_dim(db, "episodes_vec"))
        db.execute(
            "INSERT OR REPLACE INTO episodes_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        db.commit()
        assert _fts_rowids(db, "marmot") == [rowid]

        Episode.hard_delete(episode_id)

        assert _fts_rowids(db, "marmot") == []
        assert db.execute(
            "SELECT 1 FROM episodes_vec WHERE rowid = ?", (rowid,)
        ).fetchone() is None
        assert db.execute(
            "SELECT 1 FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone() is None


class TestDataGraphUnchanged:

    def test_data_graph_still_indexes_through_generalized_engine(
        self, db: sqlite3.Connection
    ) -> None:
        """Regression: a searchable data_graph row still earns its FTS posting
        after the engine was generalized to be table-driven."""
        FactRow.store("favorite_language", "the wombat prefers rust")

        _drain_search_index()

        rows = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH ?", ('"wombat"',)
        ).fetchall()
        assert len(rows) == 1
        assert db.execute(
            "SELECT indexed_at FROM data_graph WHERE id = ?", (rows[0][0],)
        ).fetchone()[0] is not None
