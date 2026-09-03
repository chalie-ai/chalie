"""Feature tests — migration 015 (purge orphaned ``kind='document'`` rows).

Recreates the on-disk footprint the deleted document vertical left behind on
a provisioned database: ``data_graph`` fragment rows plus their FTS5 postings,
key/value vec shadows, and referencing edges. The migration must remove every
piece of that footprint — index postings and vec rows included, or their
rowids later alias onto new ``data_graph`` rows — while leaving every other
kind untouched. Real migration module, real SQLite + sqlite-vec, zero mocks.
"""

import re
import sqlite3

import pytest

from migrations import migration_015_purge_document_data_graph_rows as mig
from migrations.runner import run_all
from services.embedding_utils import pack_embedding
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit


def _db_path() -> str:
    return str(FileMapperService.get_db_path())


def _vec_dim(conn: sqlite3.Connection, table: str) -> int:
    """The live vec0 column dimension, read off the schema so a synthetic
    embedding always matches the width the table was declared with."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    match = re.search(r"float\[(\d+)\]", row[0]) if row else None
    return int(match.group(1)) if match else 768


def _blob(dim: int, hot: int) -> bytes:
    vector = [0.0] * dim
    vector[hot] = 1.0
    blob = pack_embedding(vector)
    assert blob is not None
    return blob


def _index_row(db: sqlite3.Connection, rowid: int, key: str, value: str, kind: str) -> None:
    """Write one row's full search-index footprint exactly as the production
    write-sync engine did: the external-content FTS posting plus one shadow
    row in each vec lane (keyed by the base rowid)."""
    db.execute(
        "INSERT INTO data_graph_fts (rowid, key, value, kind) "
        "VALUES (?, ?, ?, ?)",
        (rowid, key, value, kind),
    )
    db.execute(
        "INSERT INTO data_graph_key_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, _blob(_vec_dim(db, "data_graph_key_vec"), rowid % 700)),
    )
    db.execute(
        "INSERT INTO data_graph_value_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, _blob(_vec_dim(db, "data_graph_value_vec"), rowid % 700)),
    )


def _insert(db: sqlite3.Connection, kind: str, key: str, value: str, source: str | None) -> int:
    cur = db.execute(
        "INSERT INTO data_graph (kind, key, value, source) "
        "VALUES (?, ?, ?, ?)",
        (kind, key, value, source),
    )
    assert cur.lastrowid is not None
    _index_row(db, cur.lastrowid, key, value, kind)
    return cur.lastrowid


def _seed(db: sqlite3.Connection, fragments: int = 3) -> tuple[list[int], int, int]:
    """Seed ``fragments`` document rows plus two survivor rows of another
    kind, all fully indexed, with edges in every direction that matters:
    document→survivor, survivor→document, and survivor→survivor (the one
    edge that must outlive the purge). Returns (doc_ids, survivor_a, survivor_b)."""
    doc_ids = [
        _insert(db, "document", f"doc:abc:{i:03d}", f"docterm fragment {i}", "document:abc")
        for i in range(fragments)
    ]
    survivor_a = _insert(db, "user_specific", "favorite_color", "survivorterm blue", None)
    survivor_b = _insert(db, "user_specific", "home_town", "survivorterm valletta", None)

    edges = [
        (doc_ids[0], survivor_a),
        (survivor_a, doc_ids[-1]),
        (survivor_a, survivor_b),
    ]
    for from_id, to_id in edges:
        db.execute(
            "INSERT INTO data_graph_edges (from_id, to_id) VALUES (?, ?)",
            (from_id, to_id),
        )
    db.commit()
    return doc_ids, survivor_a, survivor_b


class TestNeeded:
    def test_true_when_document_rows_exist(self, db: sqlite3.Connection) -> None:
        _seed(db)
        assert mig.needed(db) is True

    def test_false_on_fresh_db(self, db: sqlite3.Connection) -> None:
        assert mig.needed(db) is False


class TestApply:
    def test_document_rows_die_survivors_live(self, db: sqlite3.Connection) -> None:
        _, survivor_a, survivor_b = _seed(db)
        mig.apply(_db_path())

        assert db.execute(
            "SELECT COUNT(*) FROM data_graph WHERE kind = 'document'"
        ).fetchone()[0] == 0
        survivors = {
            row[0] for row in db.execute("SELECT id FROM data_graph").fetchall()
        }
        assert survivors == {survivor_a, survivor_b}

    def test_fts_postings_are_purged_with_the_rows(self, db: sqlite3.Connection) -> None:
        """The external-content index must forget the dead rows — a posting
        left behind resurfaces as a phantom hit joined onto whatever row
        later reuses the rowid."""
        _seed(db)
        mig.apply(_db_path())

        dead = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH 'docterm'"
        ).fetchall()
        assert dead == []
        alive = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH 'survivorterm'"
        ).fetchall()
        assert len(alive) == 2
        # Index ↔ content-table consistency, verified by FTS5 itself.
        db.execute("INSERT INTO data_graph_fts(data_graph_fts) VALUES('integrity-check')")

    def test_vec_shadows_are_purged_with_the_rows(self, db: sqlite3.Connection) -> None:
        """Both vec lanes must drop the dead rowids — orphaned vec rows alias
        onto future data_graph rows and poison KNN results."""
        doc_ids, survivor_a, survivor_b = _seed(db)
        mig.apply(_db_path())

        placeholders = ",".join("?" * len(doc_ids))
        for lane in ("data_graph_key_vec", "data_graph_value_vec"):
            orphans = db.execute(
                f"SELECT rowid FROM {lane} WHERE rowid IN ({placeholders})", doc_ids
            ).fetchall()
            assert orphans == [], f"{lane} kept vec rows for purged rowids"
            kept = {row[0] for row in db.execute(f"SELECT rowid FROM {lane}").fetchall()}
            assert kept == {survivor_a, survivor_b}

    def test_edges_touching_document_rows_die_first(self, db: sqlite3.Connection) -> None:
        """Edges into AND out of document rows must go; the survivor↔survivor
        edge stays. (The edge delete must run before the row delete — after
        it, the kind='document' subqueries match nothing.)"""
        _, survivor_a, survivor_b = _seed(db)
        mig.apply(_db_path())

        remaining = [tuple(row) for row in db.execute("SELECT from_id, to_id FROM data_graph_edges")]
        assert remaining == [(survivor_a, survivor_b)]


class TestIdempotence:
    def test_apply_twice_is_stable(self, db: sqlite3.Connection) -> None:
        _seed(db)
        mig.apply(_db_path())
        first = db.execute("SELECT id, kind, key FROM data_graph ORDER BY id").fetchall()

        assert mig.needed(db) is False  # second pass sees nothing to do
        mig.apply(_db_path())
        assert db.execute("SELECT id, kind, key FROM data_graph ORDER BY id").fetchall() == first


class TestRunnerIntegration:
    def test_runner_applies_and_ledgers_the_step(self, db: sqlite3.Connection) -> None:
        _seed(db)
        run_all(_db_path())

        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'data-graph-document-purge-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "applied"
        assert db.execute(
            "SELECT COUNT(*) FROM data_graph WHERE kind = 'document'"
        ).fetchone()[0] == 0

    def test_runner_noops_on_fresh_db(self, db: sqlite3.Connection) -> None:
        run_all(_db_path())
        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'data-graph-document-purge-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "noop"
