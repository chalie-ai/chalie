"""Feature tests for find_tools discovery via abilities.sqlite only.

Strategy: monkeypatch _DB_PATH on FindToolsAbility to a tmp_path
database populated with real embeddings. Real EmbeddingService is used.
``FindToolsAbility.execute()`` reads the calling MessageProcessor's
``DISCOVERABLE`` list via ``self.MessageProcessor``; tests bind a stub
processor for that lookup. Direct ``_query`` and ``_fallback`` calls
accept the allowlist as a positional arg so RRF ordering can be verified
by formula, not by semantic luck.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from abilities._search import RRF_K
from abilities.find_tools import FindToolsAbility
from services.message_processor import MessageProcessor
from tests.helpers import make_stub_config

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub processor — a *flat* MessageProcessor instance (no subclass: §7a / P1)
# carrying a custom DISCOVERABLE list for the find_tools query gate.
# ---------------------------------------------------------------------------


def _make_stub_processor(discoverable: list[str]) -> MessageProcessor:
    """find_tools reads ``mp.config.discoverable`` / ``mp.config.blocked``."""
    proc = object.__new__(MessageProcessor)
    proc.config = make_stub_config(discoverable=discoverable)
    proc._active_tools = []
    return proc



# ---------------------------------------------------------------------------
# Session-scoped real embeddings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _real_embeddings():
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
# Helpers
# ---------------------------------------------------------------------------

def _build_abilities_sqlite(path: Path, abilities: list) -> None:
    """Build a minimal abilities.sqlite at path.

    Each ability dict: {"name": str, "summary": str, "embedding": list[float]}.
    """
    from utils.build_ability_db import _create_schema, _load_sqlite_vec
    from services.embedding_utils import pack_embedding

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _load_sqlite_vec(conn)
    _create_schema(conn)

    for ab in abilities:
        conn.execute(
            "INSERT INTO abilities(name, summary) VALUES (?, ?)",
            (ab["name"], ab["summary"]),
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
        conn.execute(
            "INSERT INTO ability_search_fts(rowid, text) VALUES (?, ?)",
            (entry_id, ab["summary"]),
        )

    conn.commit()
    conn.close()


def _execute_with_discoverable(ability, query, discoverable):
    """The query path caps results at ``MAX_QUERY_RESULTS`` internally - there is no
    caller 'limit' knob (dropped in the v2 select/query split)."""
    proc = _make_stub_processor(discoverable=discoverable)
    ability.mp = proc
    result = ability.run({"query": query})
    return result, proc.active_tools


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindToolsDiscovery:

    def test_weather_discovered_via_abilities_db(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(ability, "weather forecast", ["weather"])

        assert "weather" in active, (
            f"Expected 'weather' in active, got: {active}"
        )

    def test_fts_keyword_path_surfaces_ability(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """Uses the query 'python sandbox' - FTS5 requires ALL query terms; the FTS
        hit lifts the result above ``MIN_RRF_SCORE`` where a vec-only signal would not.
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
        _, active = _execute_with_discoverable(
            ability, "python sandbox", ["weather", "code_eval"]
        )

        assert "code_eval" in active, (
            f"Expected 'code_eval' via FTS keyword, got: {active}"
        )

    def test_rrf_merges_vec_and_fts_results(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """``_query`` does NOT apply the relevance floor (that is layered on in
        ``_run_query``), so both abilities appear fused with weather ahead of code_eval.
        """
        from services.embedding_service import EmbeddingService
        from services.embedding_utils import pack_embedding

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
        blob = pack_embedding(EmbeddingService().generate_embedding("weather forecast"))
        rows = ability._query("weather forecast", blob, ["weather", "code_eval"])

        names = [r["key"] for r in rows]
        assert len(names) == len(set(names)), f"Duplicates in merged rows: {names}"
        assert set(names) == {"weather", "code_eval"}, (
            f"RRF should fuse both retrievers (weather dual-signal, code_eval via "
            f"vec). Got: {names}"
        )
        assert names[0] == "weather", (
            f"weather (dual vec+FTS signal) must outrank code_eval (vec only). "
            f"Got order: {names}"
        )

    def test_no_duplicates_when_ability_in_both_retrievers(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(
            ability, "weather forecast", ["weather"]
        )

        assert active.count("weather") <= 1, (
            f"Duplicate 'weather' in active: {active}"
        )

    def test_missing_db_returns_empty_gracefully(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "does_not_exist" / "abilities.sqlite"
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", nonexistent)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(ability, "weather forecast", ["weather"])

        assert active == []

    def test_discoverable_allowlist_filters_results(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """Indexes both 'memory' and 'weather'; processor lists only ['weather'].
        Even when 'memory' would otherwise rank, the gate excludes it.
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
        _, active = _execute_with_discoverable(ability, "weather forecast", ["weather"])

        assert "memory" not in active, (
            f"'memory' should be filtered by DISCOVERABLE allowlist, got: "
            f"{active}"
        )
        assert "weather" in active

    def test_empty_discoverable_returns_empty_results(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """SQL never executes when the allowlist is empty."""
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(ability, "weather forecast", [])

        assert active == []


# ---------------------------------------------------------------------------
# Gap fills
# ---------------------------------------------------------------------------


def _build_stub_db(path: Path, abilities: list) -> None:
    """Like _build_abilities_sqlite but accepts raw numpy float32 embeddings.

    Each ability dict: {"name": str, "summary": str, "embedding": np.ndarray}
    """
    from utils.build_ability_db import _create_schema, _load_sqlite_vec
    from services.embedding_utils import pack_embedding

    conn = sqlite3.connect(str(path))
    _load_sqlite_vec(conn)
    _create_schema(conn)
    for ab in abilities:
        conn.execute(
            "INSERT INTO abilities(name, summary) VALUES (?, ?)",
            (ab["name"], ab["summary"]),
        )
        ability_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_entries(ability_id, text, kind) VALUES (?, ?, ?)",
            (ability_id, ab["summary"], "summary"),
        )
        entry_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ability_search_vec(rowid, embedding) VALUES (?, ?)",
            (entry_id, pack_embedding(ab["embedding"].tolist())),
        )
        conn.execute(
            "INSERT INTO ability_search_fts(rowid, text) VALUES (?, ?)",
            (entry_id, ab["summary"]),
        )
    conn.commit()
    conn.close()


class TestFindToolsPhase3Gaps:

    def test_rrf_order_matches_formula_when_vec_and_fts_disagree(self, tmp_path, monkeypatch):
        """Fixture: ability_a vec rank 1 (no FTS hit); ability_b vec rank 2 with FTS
        rank 1 ('zetakeyword' in summary).

        RRF formula:
          score(a) = 1/(RRF_K+1) [vec rank 1]   = 0.0625
          score(b) = 1/(RRF_K+2) + 1/(RRF_K+1) = 0.0588 + 0.0625 = 0.1213

        ability_b must rank first.
        """
        from services.embedding_utils import pack_embedding

        db_path = tmp_path / "rrf_order.sqlite"

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0
        a_emb = np.zeros(768, dtype=np.float32)
        a_emb[0] = 0.99
        a_emb[1] = 0.01   # close to q_vec
        b_emb = np.zeros(768, dtype=np.float32)
        b_emb[0] = -0.99
        b_emb[1] = 0.01   # opposite direction (large vec distance)

        _build_stub_db(db_path, [
            {"name": "ability_a", "summary": "alphajet engine design",
             "embedding": a_emb},
            {"name": "ability_b", "summary": "zetakeyword turbine systems",
             "embedding": b_emb},
        ])

        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", db_path)
        ability = FindToolsAbility()
        blob = pack_embedding(q_vec.tolist())
        rows = ability._query("zetakeyword", blob, ["ability_a", "ability_b"])

        names = [r["key"] for r in rows]
        assert len(names) >= 2, f"Expected both abilities in results, got: {names}"
        assert names[0] == "ability_b", (
            f"ability_b should rank first (FTS win + vec rank 2 > vec-only rank 1). "
            f"Got order: {names}. "
            f"Expected score(b) = {1/(RRF_K+1) + 1/(RRF_K+2):.4f} > "
            f"score(a) = {1/(RRF_K+1):.4f}"
        )

    def test_fallback_keyword_search_queries_abilities_sqlite(self, tmp_path, monkeypatch):
        """_fallback queries ability_search_fts in abilities.sqlite (never tool_capability_profiles)."""
        db_path = tmp_path / "fallback.sqlite"
        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0
        _build_stub_db(db_path, [
            {"name": "sandboxer",
             "summary": "Execute Python code in a restricted sandbox",
             "embedding": q_vec},
        ])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", db_path)

        ability = FindToolsAbility()
        proc = _make_stub_processor(discoverable=["sandboxer"])
        ability.mp = proc
        ability._fallback("sandbox", ["sandboxer"])

        assert proc.active_tools == ["sandboxer"]

    def test_query_abilities_filters_by_allowlist(self, tmp_path, monkeypatch):
        """_query ignores rows whose name is outside the allowlist, regardless of vec or FTS rank."""
        from services.embedding_utils import pack_embedding

        db_path = tmp_path / "allowlist.sqlite"

        q_vec = np.zeros(768, dtype=np.float32)
        q_vec[0] = 1.0
        a_emb = np.zeros(768, dtype=np.float32)
        a_emb[0] = 0.99
        b_emb = np.zeros(768, dtype=np.float32)
        b_emb[0] = 0.95

        _build_stub_db(db_path, [
            {"name": "blocked_ability", "summary": "alphajet engine design", "embedding": a_emb},
            {"name": "allowed_ability", "summary": "alphajet engine design", "embedding": b_emb},
        ])

        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", db_path)
        ability = FindToolsAbility()
        blob = pack_embedding(q_vec.tolist())
        rows = ability._query("alphajet", blob, ["allowed_ability"])

        names = {r["key"] for r in rows}
        assert names == {"allowed_ability"}, (
            f"Expected only 'allowed_ability' through the allowlist, got: {names}"
        )

