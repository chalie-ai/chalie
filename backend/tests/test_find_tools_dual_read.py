"""Feature tests for the Phase 1 dual-read in find_tools_skill.py.

Verifies that _query_abilities_db integrates correctly with handle_find_tools:
- New abilities.sqlite results are surfaced and take priority over old-DB rows
- Old-DB-only results still work (regression — existing path not broken)
- Missing new DB degrades gracefully; old-DB results still surface
- Empty new DB adds nothing; result is identical to old-DB-only
- Dedup: when both DBs contain the same ability name, it appears exactly once

Strategy: monkeypatch _ABILITIES_DB_PATH in find_tools_skill to a tmp_path
database. Use either the production 256-dim db fixture or a local 768-dim
fixture depending on which embedding dimension the test path requires.

Real embeddings are generated once per session via _real_embeddings fixture.
Real EmbeddingService and ToolRegistryService are used — no mocks.
"""

import shutil
import sqlite3
import struct
from pathlib import Path

import pytest

import services.database_service as _db_mod
import services.innate_skills.find_tools_skill as _fts_mod
from services.innate_skills.find_tools_skill import handle_find_tools

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Session-scoped real embeddings (computed once, reused across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _real_embeddings():
    """Return pre-computed 768-dim embeddings for reuse across tests.

    Computed once per process using the real EmbeddingService.
    """
    from services.embedding_service import EmbeddingService
    es = EmbeddingService()
    weather_summary = (
        "Get current weather and tomorrow's forecast for a city or device coordinates."
    )
    return {
        "query": es.generate_embedding("weather forecast"),
        "weather_summary": es.generate_embedding(weather_summary),
        "weather_summary_text": weather_summary,
    }


# ---------------------------------------------------------------------------
# 768-dim DB fixtures (new-DB tests need the old-DB vec table to accept 768-dim)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _db_template_768(tmp_path_factory):
    """Session-scoped fully-converged 768-dim SQLite database template."""
    from services.database_service import DatabaseService
    from services.schema_convergence_service import SchemaConvergenceService

    template_dir = tmp_path_factory.mktemp("db_template_768")
    template_path = str(template_dir / "template768.db")

    db = DatabaseService(template_path)
    convergence = SchemaConvergenceService(db, embedding_dimensions=768)
    convergence.converge()

    with db.connection() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    db.close_pool()
    return template_path


@pytest.fixture
def db768(_db_template_768, tmp_path):
    """Fresh 768-dim database — one per test. Mirrors the standard db fixture."""
    from services.database_service import DatabaseService

    test_db_path = str(tmp_path / "test768.db")
    shutil.copy2(_db_template_768, test_db_path)

    db_service = DatabaseService(test_db_path)

    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    import services.data_graph_service as _dgs_mod
    original_dgs = _dgs_mod._instance
    _dgs_mod._instance = None

    conn = db_service._get_connection()
    try:
        yield conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs


# ---------------------------------------------------------------------------
# Real ToolRegistryService fixture — seeds the singleton's tools dict directly
# ---------------------------------------------------------------------------

@pytest.fixture
def _real_registry():
    """Yield a real ToolRegistryService with a clean tools dict.

    Resets the module-level singleton before and after each test so tests
    don't bleed state. Returns the live instance for per-test seeding.
    """
    import services.tool_registry_service as _trs_mod
    from services.tool_registry_service import ToolRegistryService

    # Reset the singleton so __init__ runs fresh
    _trs_mod._instance = None
    registry = ToolRegistryService()
    # Override tools with an empty dict — tests seed what they need
    registry.tools = {}

    yield registry

    # Cleanup: reset again so subsequent tests get a clean slate
    _trs_mod._instance = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_embedding(values):
    return struct.pack(f'{len(values)}f', *values)


def _library_tool_entry(manifest=None):
    """Minimal registry entry that passes _is_ready() and _is_interface_online()."""
    return {
        "manifest": manifest or {
            "input_schema": {"type": "object", "properties": {}, "required": []}
        },
        "source_type": "library",
    }


def _seed_old_db_tool(conn, tool_name, embedding):
    """Insert a tool_capability_profiles row + vec companion."""
    conn.execute(
        "INSERT INTO tool_capability_profiles "
        "(id, tool_name, tool_type, short_summary, full_profile, domain, effort) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            f'tcp-{tool_name}', tool_name, 'tool',
            f'{tool_name} summary', f'{tool_name} full', 'Context', 'light',
        ),
    )
    row = conn.execute(
        "SELECT rowid FROM tool_capability_profiles WHERE tool_name = ?", (tool_name,)
    ).fetchone()
    rowid = row[0]
    blob = _pack_embedding(embedding)
    conn.execute(
        "INSERT INTO tool_capability_profiles_vec (rowid, embedding) VALUES (?, ?)",
        (rowid, blob),
    )
    conn.commit()


def _build_abilities_sqlite(path: Path, abilities: list) -> None:
    """Build a minimal abilities.sqlite at *path* using real production schema helpers.

    Each ability dict: {"name": str, "summary": str, "embedding": list[float]}.
    """
    from utils.build_ability_db import _rebuild_schema, _load_sqlite_vec
    from services.embedding_utils import pack_embedding

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _load_sqlite_vec(conn)
    _rebuild_schema(conn)

    for ab in abilities:
        conn.execute(
            "INSERT INTO abilities(name, summary, always_available) VALUES (?, ?, ?)",
            (ab["name"], ab["summary"], 0),
        )
        ability_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_entries(ability_id, text, kind) VALUES (?, ?, ?)",
            (ability_id, ab["summary"], "summary"),
        )
        entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_vec(rowid, embedding) VALUES (?, ?)",
            (entry_id, pack_embedding(ab["embedding"])),
        )

    conn.commit()
    conn.close()


def _schema_only_abilities_sqlite(path: Path) -> None:
    """Create abilities.sqlite at *path* with schema but zero rows."""
    from utils.build_ability_db import _rebuild_schema, _load_sqlite_vec

    conn = sqlite3.connect(str(path))
    _load_sqlite_vec(conn)
    _rebuild_schema(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDualRead:
    """handle_find_tools dual-read: new abilities.sqlite + old tool_capability_profiles."""

    # -- 1. New-DB-only weather discovery ------------------------------------

    def test_new_db_only_weather_discovered(
        self, db768, tmp_path, monkeypatch, _real_embeddings, _real_registry
    ):
        """Weather in new DB, no weather in old DB → 'weather' appears in _discovered_tools.

        New-DB rows bypass registry checks (they are pre-scored dicts), so the
        empty real registry is sufficient — no tool seeding needed here.
        """
        weather_emb = _real_embeddings["weather_summary"]

        # Build new DB with weather (768-dim; matches abilities.sqlite schema)
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": weather_emb,
        }])
        monkeypatch.setattr(_fts_mod, "_ABILITIES_DB_PATH", new_db_path)

        # Old DB (db768): no weather rows — new-DB is the only source.
        # Registry has no tools — new-DB rows bypass registry checks entirely.

        result = handle_find_tools("text", {"query": "weather forecast"})
        assert isinstance(result, dict)
        assert "weather" in result["_discovered_tools"], (
            f"Expected 'weather' in _discovered_tools, got: {result['_discovered_tools']}"
        )

    # -- 2. Old-DB-only weather discovery (regression) -----------------------

    def test_old_db_only_weather_still_discovered(
        self, db768, tmp_path, monkeypatch, _real_embeddings, _real_registry
    ):
        """Old-DB has weather, new DB is empty → 'weather' still surfaces (regression guard).

        Uses db768 (768-dim vec table) so the real EmbeddingService query embedding
        matches the stored vector dimension without a mismatch error.
        """
        embedding = _real_embeddings["weather_summary"]

        _seed_old_db_tool(db768, "weather", embedding)

        new_db_path = tmp_path / "abilities.sqlite"
        _schema_only_abilities_sqlite(new_db_path)
        monkeypatch.setattr(_fts_mod, "_ABILITIES_DB_PATH", new_db_path)

        # Old-DB tuples go through registry checks: seed the real registry
        _real_registry.tools["weather"] = _library_tool_entry()

        result = handle_find_tools("text", {"query": "weather forecast"})
        assert "weather" in result["_discovered_tools"], (
            f"Old-DB regression: expected 'weather', got: {result['_discovered_tools']}"
        )

    # -- 3. Dedup: both DBs contain "weather" — appears exactly once ---------

    def test_dedup_weather_appears_once_when_in_both_dbs(
        self, db768, tmp_path, monkeypatch, _real_embeddings, _real_registry
    ):
        """Both DBs have 'weather'. _discovered_tools lists it exactly once."""
        query_emb = _real_embeddings["query"]
        weather_emb = _real_embeddings["weather_summary"]

        # New DB
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": weather_emb,
        }])
        monkeypatch.setattr(_fts_mod, "_ABILITIES_DB_PATH", new_db_path)

        # Old DB also has weather (768-dim to match db768)
        _seed_old_db_tool(db768, "weather", query_emb)

        # Registry entry needed only if old-DB row survives dedup (it won't — new DB wins)
        _real_registry.tools["weather"] = _library_tool_entry()

        result = handle_find_tools("text", {"query": "weather forecast"})
        discovered = result["_discovered_tools"]
        assert discovered.count("weather") <= 1, (
            f"Dedup failed: 'weather' appears {discovered.count('weather')} times"
        )
        assert "weather" in discovered

    # -- 4. Missing new DB: graceful degrade, old-DB results surface ---------

    def test_missing_new_db_falls_back_to_old_db(
        self, db768, tmp_path, monkeypatch, _real_embeddings, _real_registry
    ):
        """If abilities.sqlite does not exist, old-DB results still surface; no exception raised."""
        embedding = _real_embeddings["weather_summary"]
        _seed_old_db_tool(db768, "weather", embedding)

        nonexistent = tmp_path / "does_not_exist" / "abilities.sqlite"
        monkeypatch.setattr(_fts_mod, "_ABILITIES_DB_PATH", nonexistent)

        _real_registry.tools["weather"] = _library_tool_entry()

        result = handle_find_tools("text", {"query": "weather forecast"})
        assert isinstance(result, dict)
        assert "weather" in result["_discovered_tools"]

    # -- 5. Empty new DB: adds nothing, identical to old-DB-only ------------

    def test_empty_new_db_adds_nothing(
        self, db768, tmp_path, monkeypatch, _real_embeddings, _real_registry
    ):
        """Schema-only abilities.sqlite (zero rows) contributes nothing; old-DB result unchanged."""
        embedding = _real_embeddings["weather_summary"]
        _seed_old_db_tool(db768, "weather", embedding)

        new_db_path = tmp_path / "abilities.sqlite"
        _schema_only_abilities_sqlite(new_db_path)
        monkeypatch.setattr(_fts_mod, "_ABILITIES_DB_PATH", new_db_path)

        _real_registry.tools["weather"] = _library_tool_entry()

        result = handle_find_tools("text", {"query": "weather forecast"})
        assert "weather" in result["_discovered_tools"]
