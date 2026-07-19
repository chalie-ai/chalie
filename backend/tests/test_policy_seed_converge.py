"""Boot-time convergence of the ``policy`` table to ``policy_defaults.json``.

``PolicyManager.apply_seed`` runs on every boot and makes the seed file the
single source of truth: it INSERT-OR-IGNOREs every seeded row, then reaps every
row whose permission is neither in the seed nor an MCP tool permission.

That reap is what makes retiring an ability a seed-file edit instead of a
one-shot cleanup migration, so these tests pin the four behaviours the reap
must get right together — kill a retired ability, kill a gate-bypassing
INTERNAL row, SPARE lazily-created MCP rows, and PRESERVE a user's setting on a
still-seeded permission.  Real convergence, real seed file, real SQLite; the
only patch is the DB path (mirrors test_policy_migration.py).
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from services.database import Database
from services.file_mapper_service import FileMapperService
from services.policy_manager import PolicyManager
from services.schema_convergence_service import SchemaConvergenceService

pytestmark = pytest.mark.unit


@pytest.fixture()
def seeded_db(tmp_path: Path):
    """A booted DB: converged schema + policy defaults applied, exactly as
    _init_database does it."""
    db_path = tmp_path / "converge.db"
    with patch.object(FileMapperService, "get_db_path", return_value=db_path):
        Database.close()
        SchemaConvergenceService().converge()
        PolicyManager().apply_seed()
        yield db_path
        Database.close()


def test_reap_kills_retired_and_internal_rows_but_spares_mcp_and_user_settings(
    seeded_db: Path,
) -> None:
    """The whole contract in one boot: a second apply_seed pass converges the
    table to the seed without collateral damage."""
    with patch.object(FileMapperService, "get_db_path", return_value=seeded_db):
        conn = Database.conn()
        # Rows a real instance accumulates between boots:
        conn.executescript("""
            -- retired abilities (what migrations 016/017 used to delete by hand)
            INSERT INTO policy (channel, permission, setting)
                 VALUES ('chat', 'code_eval', 'allow'),
                        ('chat', 'file_permissions.read', 'allow'),
            -- a gate-bypassing INTERNAL tool that once carried a row
                        ('chat', 'read_file', 'ask'),
            -- an MCP tool row, created lazily by the gate, absent from the seed
                        ('chat', '_mcp_github_create_issue', 'allow'),
            -- a name that SQLite's LIKE '_mcp_%' would wrongly match ('_' is a
            -- single-char wildcard) but which is NOT an MCP tool
                        ('chat', 'xmcpy_not_an_mcp_tool', 'allow');
            -- a user's own override on a permission that IS still seeded
            UPDATE policy SET setting = 'deny'
             WHERE channel = 'chat' AND permission = 'bash.read';
        """)
        conn.commit()

        PolicyManager().apply_seed()  # the boot pass under test

        def setting(permission: str) -> str | None:
            row = conn.execute(
                "SELECT setting FROM policy WHERE channel='chat' AND permission=?",
                (permission,),
            ).fetchone()
            return row[0] if row else None

        # Reaped — no migration required to remove them.
        assert setting("code_eval") is None, "retired ability row survived the reap"
        assert setting("file_permissions.read") is None, "retired sub-permission survived"
        assert setting("read_file") is None, "INTERNAL-tool row survived the reap"

        # Spared — MCP rows are legitimately absent from the seed.
        assert setting("_mcp_github_create_issue") == "allow", "MCP row was reaped"

        # Reaped — proves the reap does not use LIKE '_mcp_%', whose '_' wildcard
        # would have spared this non-MCP row.
        assert setting("xmcpy_not_an_mcp_tool") is None, "LIKE wildcard leaked a non-MCP row"

        # Preserved — the reap keys on permission membership, never on setting,
        # so a user's override on a seeded permission is untouched.
        assert setting("bash.read") == "deny", "user override on a seeded row was reset"


def test_fresh_install_seeds_api_key_as_sensitive(tmp_path: Path) -> None:
    """REGRESSION: a fresh boot (converge → seed) must run schema.sql's seed pass
    so ``api_key`` is marked sensitive — otherwise the REST API key is persisted
    in cleartext.  Convergence skips that pass on a database that already looks
    non-empty, so nothing may create the ``policy`` table ahead of it.  Mirrors
    _init_database's boot order against a real temp DB, zero mocks."""
    fresh_path = tmp_path / "fresh.db"
    with patch.object(FileMapperService, "get_db_path", return_value=fresh_path):
        Database.close()  # drop any thread connection bound to another path
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


def test_apply_seed_is_idempotent_across_boots(seeded_db: Path) -> None:
    """Re-booting must not churn: the seed is exhaustive (every permission in
    every channel), so a second pass inserts nothing and reaps nothing."""
    with patch.object(FileMapperService, "get_db_path", return_value=seeded_db):
        conn = Database.conn()
        before = conn.execute("SELECT count(*) FROM policy").fetchone()[0]

        inserted = PolicyManager().apply_seed()

        after = conn.execute("SELECT count(*) FROM policy").fetchone()[0]
        assert inserted == 0, "already-seeded rows were re-inserted"
        assert after == before, "row count changed on a no-op boot"

        # Every seeded row is present exactly once.
        with open(FileMapperService.get_policy_defaults_path()) as f:
            seed = json.load(f)
        assert after == len(seed), f"expected the {len(seed)} seeded rows, found {after}"
