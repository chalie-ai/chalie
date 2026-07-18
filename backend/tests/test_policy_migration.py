"""Boot policy migration + fresh-install seed-ordering regression.

Two paths:
  * Upgrade — a legacy ``policy_rules`` table is copied 1:1 into the flat
    ``policy`` table before convergence drops policy_rules.
  * Fresh   — no policy_rules, so the copy is a no-op that must NOT create the
    ``policy`` table early.  If it did, convergence would treat the database as
    non-fresh and skip schema.sql's INSERT OR IGNORE seed pass, leaving the
    ``api_key`` setting unmarked-sensitive — and thus stored in cleartext.
"""
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from run import _migrate_legacy_policy_rules
from services.database import Database
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager
from services.schema_convergence_service import SchemaConvergenceService

pytestmark = pytest.mark.unit


@pytest.fixture()
def legacy_db_path(tmp_path: Path) -> Path:
    """A pre-redesign DB file: has policy_rules with one custom row, no policy table."""
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE policy_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
            context TEXT NOT NULL, state TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(action_id, context)
        );
        INSERT INTO policy_rules (action_id, context, state, updated_at)
        VALUES ('bash.modify_file', 'chat', 'allow', '2026-01-01T00:00:00+00:00');
    """)
    conn.commit()
    conn.close()
    return db_path


def test_upgrade_copies_legacy_rules_one_to_one(legacy_db_path: Path) -> None:
    # _migrate_legacy_policy_rules resolves its write path through the Database
    # gateway (FileMapperService.get_db_path) — redirect it to the legacy db file
    # so the copy runs against that file, not the real chalie.db.
    with patch.object(FileMapperService, "get_db_path", return_value=legacy_db_path):
        Database.close()
        _migrate_legacy_policy_rules()
        conn = Database.conn()
        # policy table created to hold the copy
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='policy'").fetchone()[0] == 1
        # custom row copied 1:1 (context→channel, action_id→permission, state→setting)
        assert conn.execute(
            "SELECT setting FROM policy WHERE channel='chat' AND permission='bash.modify_file'"
        ).fetchone()[0] == "allow"
        Database.close()


def test_fresh_install_is_noop_and_leaves_db_empty(tmp_path: Path) -> None:
    """Fresh path: no policy_rules → no-op → the policy table is NOT created early,
    so convergence still sees a genuinely empty (fresh) database."""
    fresh_noop_path = tmp_path / "fresh_noop.db"
    with patch.object(FileMapperService, "get_db_path", return_value=fresh_noop_path):
        Database.close()
        _migrate_legacy_policy_rules()
        conn = Database.conn()
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0] == 0
        Database.close()


def test_fresh_install_seeds_api_key_as_sensitive(tmp_path: Path) -> None:
    """REGRESSION: a fresh boot (copy no-op → converge → seed) must run schema.sql's
    seed pass so ``api_key`` is marked sensitive — otherwise the REST API key is
    persisted in cleartext.  Mirrors _init_database's boot order against a real
    temp DB with real convergence — zero mocks."""
    fresh_path = tmp_path / "fresh.db"
    # converge()/apply_seed()/_migrate_legacy_policy_rules all resolve their write
    # path through the Database gateway (FileMapperService.get_db_path); point it
    # at fresh.db for the whole boot sequence so every step lands in the same file.
    with patch.object(FileMapperService, "get_db_path", return_value=fresh_path):
        Database.close()  # drop any thread connection bound to another path
        _migrate_legacy_policy_rules()                                 # no-op on fresh
        SchemaConvergenceService().converge()  # creates + seeds
        PolicyManager().apply_seed()                                   # policy defaults
        api_key = Database.conn().execute(
            "SELECT is_sensitive FROM settings WHERE key='api_key'"
        ).fetchone()
        policy_rows = Database.conn().execute("SELECT count(*) FROM policy").fetchone()[0]
        Database.close()

    assert api_key is not None, "api_key seed row missing — schema.sql seed pass was skipped"
    assert api_key[0] == 1, "api_key not marked sensitive — REST key would be stored cleartext"
    assert policy_rows > 0, "policy defaults not seeded"
