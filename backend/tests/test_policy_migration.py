"""Boot migration: create policy, copy policy_rules 1:1, apply static seed."""
import contextlib
import sqlite3

import pytest

from run import _migrate_policy_table

pytestmark = pytest.mark.unit


class _FakeDB:
    def __init__(self, conn): self._conn = conn
    def connection(self):
        @contextlib.contextmanager
        def _ctx(): yield self._conn
        return _ctx()


@pytest.fixture()
def legacy_db():
    """A pre-redesign DB: has policy_rules with one custom row, no policy table."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE policy_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
            context TEXT NOT NULL, state TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(action_id, context)
        );
        INSERT INTO policy_rules (action_id, context, state, updated_at)
        VALUES ('email.manage', 'chat', 'allow', '2026-01-01T00:00:00+00:00');
    """)
    yield conn
    conn.close()


def test_migration_creates_copies_and_seeds(legacy_db):
    _migrate_policy_table(_FakeDB(legacy_db))
    # policy table exists
    assert legacy_db.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='policy'").fetchone()[0] == 1
    # custom row copied 1:1 and NOT overwritten by the seed (INSERT OR IGNORE)
    assert legacy_db.execute(
        "SELECT setting FROM policy WHERE channel='chat' AND permission='email.manage'"
    ).fetchone()[0] == "allow"
    # static seed applied (a visible default + an internal row)
    assert legacy_db.execute(
        "SELECT setting FROM policy WHERE channel='chat' AND permission='email.search'"
    ).fetchone()[0] == "allow"
    assert legacy_db.execute(
        "SELECT setting FROM policy WHERE channel='subconscious' AND permission='memory.recall'"
    ).fetchone()[0] == "internal"
