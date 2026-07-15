"""Re-embedding must REPLACE the vec0 row for every ``*_vec`` writer.

vec0 virtual tables reject ``INSERT OR REPLACE`` (no conflict resolution —
it raises ``UNIQUE constraint failed on <table> primary key``), so a naive
upsert leaves the previous vector in place while the source text moves on,
and retrieval drifts from what the row now says. Each test writes twice
through the real production path and asserts the second blob is the one the
table serves — exactly one row, holding the new vector."""

import re
import sqlite3

import pytest

from models.document_meta import DocumentMetaData
from models.list import List
from models.scheduled_item import ScheduledItem
from services.embedding_utils import pack_embedding
from services.episodic_service import EpisodicService

pytestmark = pytest.mark.unit


def _vec_dim(conn: sqlite3.Connection, table: str) -> int:
    """The live vec0 column dimension, read off the schema so a synthetic
    embedding always matches the width the table was declared with."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    match = re.search(r"float\[(\d+)\]", row[0]) if row else None
    return int(match.group(1)) if match else 768


def _emb(dim: int, hot: int) -> list[float]:
    """A ``dim``-length one-hot vector; distinct ``hot`` values never collide."""
    vector = [0.0] * dim
    vector[hot] = 1.0
    return vector


def _blob(dim: int, hot: int) -> bytes:
    """The packed one-hot vector, non-None for the type-checker."""
    blob = pack_embedding(_emb(dim, hot))
    assert blob is not None
    return blob


def test_update_episode_reembed_replaces_vector(db: sqlite3.Connection) -> None:
    """The encoder's update-snapshot path (``update_id``) regenerates the
    gist and its embedding — ``episodes_vec`` must serve the new vector."""
    dim = _vec_dim(db, 'episodes_vec')
    svc = EpisodicService()
    episode_id = svc.store_episode(
        {'gist': 'original gist', 'salience': 5, 'channel': 'programming'},
        embedding=_emb(dim, 0),
    )

    svc.update_episode(episode_id, {'gist': 'rewritten gist'}, embedding=_emb(dim, 1))

    rows = db.execute(
        "SELECT v.embedding FROM episodes_vec v "
        "JOIN episodes e ON e.rowid = v.rowid WHERE e.id = ?",
        (episode_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == _blob(dim, 1)


def test_list_set_embedding_replaces_vector(db: sqlite3.Connection) -> None:
    """A list rename re-embeds the new name — ``lists_vec`` must follow."""
    db.execute("INSERT INTO lists (id, name) VALUES ('l1', 'Groceries')")
    db.commit()

    dim = _vec_dim(db, 'lists_vec')
    List.set_embedding('l1', _blob(dim, 0))
    List.set_embedding('l1', _blob(dim, 1))

    rows = db.execute(
        "SELECT v.embedding FROM lists_vec v "
        "JOIN lists l ON l.rowid = v.rowid WHERE l.id = 'l1'",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == _blob(dim, 1)


def test_scheduled_item_write_embedding_replaces_vector(db: sqlite3.Connection) -> None:
    """Re-embedding a schedule's message must replace its vec row."""
    cursor = db.execute(
        "INSERT INTO scheduled_items (message, start_at) "
        "VALUES ('morning check-in reminder', '2026-01-01T00:00:00')"
    )
    db.commit()
    item_id = cursor.lastrowid
    assert item_id is not None

    dim = _vec_dim(db, 'scheduled_items_vec')
    ScheduledItem.write_embedding(item_id, _blob(dim, 0))
    ScheduledItem.write_embedding(item_id, _blob(dim, 1))

    rows = db.execute(
        "SELECT embedding FROM scheduled_items_vec WHERE rowid = ?", (item_id,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == _blob(dim, 1)


def test_document_set_embedding_replaces_vector(db: sqlite3.Connection) -> None:
    """Documents already used delete-then-insert; converging on the shared
    idiom must preserve that replace behaviour."""
    db.execute(
        "INSERT INTO documents (id, original_name, mime_type, file_path) "
        "VALUES ('d1', 'notes.txt', 'text/plain', '/data/notes.txt')"
    )
    db.commit()

    dim = _vec_dim(db, 'documents_vec')
    DocumentMetaData.set_embedding('d1', _blob(dim, 0))
    DocumentMetaData.set_embedding('d1', _blob(dim, 1))

    rows = db.execute(
        "SELECT v.embedding FROM documents_vec v "
        "JOIN documents d ON d.rowid = v.rowid WHERE d.id = 'd1'",
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == _blob(dim, 1)
