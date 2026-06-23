"""Feature tests for find_tools discovery via abilities.sqlite only.

Strategy: patch _DB_PATH on FindToolsAbility to a tmp_path database populated
with real embeddings built via the real _create_schema (trigram FTS5) and the
real EmbeddingService. ``FindToolsAbility.run()`` discovers against the GLOBAL
roster of ``DISCOVERABLE=True`` abilities (``AbilityRegistry.discoverable_names()``)
— there is no per-config allowlist; tests bind a stub processor only because
run() needs a config.
"""

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from abilities.find_tools import FindToolsAbility
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub processor — a *flat* MessageProcessor instance (no subclass: §7a / P1).
# Discovery is GLOBAL now: find_tools.run() reads the candidate roster from
# ``AbilityRegistry.discoverable_names()`` (every DISCOVERABLE=True ability),
# not from any per-config list. The stub only needs a real config carrying
# find_tools — there is nothing per-config to inject.
# ---------------------------------------------------------------------------


def _make_stub_processor() -> MessageProcessor:
    proc = object.__new__(MessageProcessor)
    proc.config = cast("ProcessorConfig", make_stub_config())
    proc._active_tools = []
    return proc


# ---------------------------------------------------------------------------
# Session-scoped real embeddings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _real_embeddings() -> dict[str, object]:
    from services.embedding_service import EmbeddingService
    es = EmbeddingService()
    weather_summary = "Get current weather and tomorrow's forecast for a city or device coordinates."
    return {
        "query_weather": es.generate_embedding("weather forecast"),
        "weather_summary": es.generate_embedding(weather_summary),
        "weather_summary_text": weather_summary,
        "code_summary": es.generate_embedding("Execute Python code in a restricted sandbox."),
        "code_summary_text": "Execute Python code in a restricted sandbox.",
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_abilities_sqlite(path: Path, abilities: list[dict[str, object]]) -> None:
    """Build a minimal abilities.sqlite at path using the real schema (trigram FTS).

    Each ability dict: {"name": str, "summary": str, "embedding": list[float]}.
    Mirrors what utils.build_ability_db._insert_ability does: inserts both the
    summary entry (with embedding) AND a kind='name' FTS-only entry per ability
    so the keyword path (+docs → chalie_docs) works on trigram substrings.
    """
    from utils.build_ability_db import _create_schema, _load_sqlite_vec
    from services.embedding_utils import pack_embedding

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _load_sqlite_vec(conn)
    _create_schema(conn)  # creates FTS with tokenize='trigram'

    for ab in abilities:
        conn.execute(
            "INSERT INTO abilities(name, summary) VALUES (?, ?)",
            (ab["name"], ab["summary"]),
        )
        ability_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Summary entry: indexed for both vector and keyword search.
        conn.execute(
            "INSERT INTO ability_search_entries(ability_id, text, kind) VALUES (?, ?, ?)",
            (ability_id, ab["summary"], "summary"),
        )
        entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_vec(rowid, embedding) VALUES (?, ?)",
            (entry_id, pack_embedding(ab["embedding"])),
        )
        conn.execute(
            "INSERT INTO ability_search_fts(rowid, text) VALUES (?, ?)",
            (entry_id, ab["summary"]),
        )

        # Name entry: FTS-only so substring queries like +docs match chalie_docs.
        conn.execute(
            "INSERT INTO ability_search_entries(ability_id, text, kind) VALUES (?, ?, 'name')",
            (ability_id, ab["name"]),
        )
        name_entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_fts(rowid, text) VALUES (?, ?)",
            (name_entry_id, ab["name"]),
        )

    conn.commit()
    conn.close()


def _execute_query(ability: FindToolsAbility, query: str) -> tuple[object, list[str]]:
    proc = _make_stub_processor()
    ability.mp = proc
    result = ability.run({"query": query})
    return result, proc.active_tools


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindToolsDiscovery:

    def test_weather_discovered_via_abilities_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _real_embeddings: dict[str, object]
    ) -> None:
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_query(ability, "+weather")

        assert "weather" in active, (
            f"Expected 'weather' in active, got: {active}"
        )

    def test_fts_keyword_path_surfaces_ability_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _real_embeddings: dict[str, object]
    ) -> None:
        """The '+' prefix requires the term substring-match against the index.
        The name entry (kind='name') lets '+sandbox' find the ability named 'sandbox'
        and '+code' find 'code_eval' via trigram substring matching.
        """
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [
            {
                "name": "weather",
                "summary": _real_embeddings["weather_summary_text"],
                "embedding": _real_embeddings["weather_summary"],
            },
            {
                "name": "code_eval",
                "summary": _real_embeddings["code_summary_text"],
                "embedding": _real_embeddings["code_summary"],
            },
        ])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_query(ability, "+code")

        assert "code_eval" in active, (
            f"Expected 'code_eval' via FTS substring match on '+code', got: {active}"
        )

    def test_no_duplicates_when_ability_matches_both_name_and_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _real_embeddings: dict[str, object]
    ) -> None:
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_query(ability, "+weather")

        assert active.count("weather") <= 1, (
            f"Duplicate 'weather' in active: {active}"
        )

    def test_missing_db_returns_empty_gracefully(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        nonexistent = tmp_path / "does_not_exist" / "abilities.sqlite"
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", nonexistent)

        ability = FindToolsAbility()
        _, active = _execute_query(ability, "+weather")

        assert active == []

    def test_discoverable_allowlist_filters_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _real_embeddings: dict[str, object]
    ) -> None:
        """Indexes both 'memory' and 'weather'. ``memory`` is DISCOVERABLE=False
        globally, so even though it is in the tmp index and would otherwise rank,
        the global discovery roster excludes it. ``weather`` is DISCOVERABLE=True
        and survives.
        """
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [
            {
                "name": "memory",
                "summary": "Store and recall information from long-term memory.",
                "embedding": _real_embeddings["weather_summary"],
            },
            {
                "name": "weather",
                "summary": _real_embeddings["weather_summary_text"],
                "embedding": _real_embeddings["weather_summary"],
            },
        ])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_query(ability, "+weather")

        assert "memory" not in active, (
            f"'memory' (DISCOVERABLE=False) must be excluded by the global "
            f"discovery roster, got: {active}"
        )
        assert "weather" in active
