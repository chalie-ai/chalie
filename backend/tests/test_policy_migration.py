"""Boot policy migration + fresh-install seed-ordering regression.

Two paths:
  * Upgrade — a legacy ``policy_rules`` table is copied 1:1 into the flat
    ``policy`` table before convergence drops policy_rules.
  * Fresh   — no policy_rules, so the copy is a no-op that must NOT create the
    ``policy`` table early.  If it did, convergence would treat the database as
    non-fresh and skip schema.sql's INSERT OR IGNORE seed pass, leaving the
    ``api_key`` setting unmarked-sensitive — and thus stored in cleartext.
"""
import contextlib
import sqlite3
from pathlib import Path

import pytest

from run import _migrate_legacy_policy_rules
from services.database_service import DatabaseService
from services.policy_manager import PolicyManager
from services.schema_convergence_service import SchemaConvergenceService

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


def test_upgrade_copies_legacy_rules_one_to_one(legacy_db):
    """Upgrade path: policy_rules copied verbatim into the flat policy table."""
    _migrate_legacy_policy_rules(_FakeDB(legacy_db))
    # policy table created to hold the copy
    assert legacy_db.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='policy'").fetchone()[0] == 1
    # custom row copied 1:1 (context→channel, action_id→permission, state→setting)
    assert legacy_db.execute(
        "SELECT setting FROM policy WHERE channel='chat' AND permission='email.manage'"
    ).fetchone()[0] == "allow"


def test_fresh_install_is_noop_and_leaves_db_empty():
    """Fresh path: no policy_rules → no-op → the policy table is NOT created early,
    so convergence still sees a genuinely empty (fresh) database."""
    conn = sqlite3.connect(":memory:")
    try:
        _migrate_legacy_policy_rules(_FakeDB(conn))
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_fresh_install_seeds_api_key_as_sensitive(tmp_path: Path):
    """REGRESSION: a fresh boot (copy no-op → converge → seed) must run schema.sql's
    seed pass so ``api_key`` is marked sensitive — otherwise the REST API key is
    persisted in cleartext.  Mirrors _init_database's boot order against a real
    temp DB with real convergence — zero mocks."""
    db = DatabaseService(str(tmp_path / "fresh.db"))

    _migrate_legacy_policy_rules(db)                                   # no-op on fresh
    SchemaConvergenceService(db, embedding_dimensions=256).converge()  # creates + seeds
    PolicyManager(db).apply_seed()                                     # policy defaults

    with db.connection() as conn:
        api_key = conn.execute(
            "SELECT is_sensitive FROM settings WHERE key='api_key'"
        ).fetchone()
        policy_rows = conn.execute("SELECT count(*) FROM policy").fetchone()[0]

    assert api_key is not None, "api_key seed row missing — schema.sql seed pass was skipped"
    assert api_key[0] == 1, "api_key not marked sensitive — REST key would be stored cleartext"
    assert policy_rows > 0, "policy defaults not seeded"
