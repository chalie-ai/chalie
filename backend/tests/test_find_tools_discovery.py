"""Feature tests for find_tools discovery via abilities.sqlite only.

Verifies:
- Weather discovered via vec k-NN in abilities.sqlite
- FTS5 BM25 path: ability with no vec match still surfaces via keyword
- RRF fusion: vec and FTS5 disagree — merged ranking is correct
- No duplicates when an ability ranks in both vec and FTS results
- RRF order matches formula when vec and FTS disagree on winner (stub embeddings)
- Fallback keyword search queries ability_search_fts in abilities.sqlite
- find_tools module has no reference to the old shared DB or tool_capability_profiles
- DISCOVERABLE allowlist gates which abilities surface for a given processor

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
    """Flat MessageProcessor carrying a config whose ``discoverable`` is the
    find_tools allow-list gate, plus an empty active_tools that find_tools
    appends discovered names onto via self.mp (TKT-835:
    find_tools reads ``mp.config.discoverable`` / ``mp.config.blocked``)."""
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
    Populates vec + FTS5 (contentless FTS5 needs explicit INSERT).
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


def _execute_with_discoverable(ability, query, discoverable, limit=None):
    """Run ability.run() inside a stub-processor binding; return
    (result_text, active_tools) so callers assert on the appended names."""
    proc = _make_stub_processor(discoverable=discoverable)
    params = {"query": query}
    if limit is not None:
        params["limit"] = limit
    ability.mp = proc
    result = ability.run(params)
    return result, proc.active_tools


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindToolsDiscovery:

    def test_weather_discovered_via_abilities_db(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """Weather in abilities.sqlite → 'weather' appears in active_tools."""
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
        """Ability with keyword match in FTS5 surfaces even when vec rank is low.

        Uses a query containing 'sandbox' — FTS5 picks it up directly from the
        code_eval summary text, independent of semantic similarity.
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
            ability, "sandbox python execution", ["weather", "code_eval"]
        )

        assert "code_eval" in active, (
            f"Expected 'code_eval' via FTS keyword, got: {active}"
        )

    def test_rrf_merges_vec_and_fts_results(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """With limit=2 and two abilities each winning one retriever, both appear.

        'weather forecast' is semantically close to weather (vec wins). 'sandbox'
        keyword in the query hits code_eval's summary via FTS5. RRF combines both.
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
            ability, "weather forecast sandbox", ["weather", "code_eval"], limit=2
        )

        assert len(active) == len(set(active)), (
            f"Duplicates in active: {active}"
        )
        assert set(active) == {"weather", "code_eval"}, (
            f"RRF should surface both abilities (weather via vec, code_eval via FTS). "
            f"Got: {active}"
        )

    def test_no_duplicates_when_ability_in_both_retrievers(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """Ability ranked by both vec k-NN and FTS5 appears exactly once."""
        new_db_path = tmp_path / "abilities.sqlite"
        _build_abilities_sqlite(new_db_path, [{
            "name": "weather",
            "summary": _real_embeddings["weather_summary_text"],
            "embedding": _real_embeddings["weather_summary"],
        }])
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", new_db_path)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(
            ability, "weather forecast", ["weather"], limit=5
        )

        assert active.count("weather") <= 1, (
            f"Duplicate 'weather' in active: {active}"
        )

    def test_missing_db_returns_empty_gracefully(self, tmp_path, monkeypatch):
        """abilities.sqlite does not exist → empty active_tools, no exception."""
        nonexistent = tmp_path / "does_not_exist" / "abilities.sqlite"
        monkeypatch.setattr(FindToolsAbility, "_DB_PATH", nonexistent)

        ability = FindToolsAbility()
        _, active = _execute_with_discoverable(ability, "weather forecast", ["weather"])

        assert active == []

    def test_discoverable_allowlist_filters_results(
        self, tmp_path, monkeypatch, _real_embeddings
    ):
        """Abilities outside the calling processor's DISCOVERABLE list never surface.

        Indexes both 'memory' and 'weather' in abilities.sqlite. Calling processor
        lists only ['weather']. Even when 'memory' would otherwise rank, the gate
        excludes it because it is not in the allowlist.
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
        """A processor with no DISCOVERABLE entries gets an empty result.

        find_tools is a no-op when the calling processor has no discoverable
        scope at all. The
        SQL never executes when the allowlist is empty.
        """
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

    Each ability dict: {"name": str, "summary": str, "embedding": np.ndarray}.
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
        """Ability winning FTS but losing vec ranks above the vec-only winner when
        the FTS boost produces a higher RRF score.

        Fixture:
        - ability_a: unit vector along dim-0 (very close to query_blob)
          → vec rank 1, not in FTS (query word 'zetakeyword' absent from summary)
        - ability_b: opposite unit vector (vec rank 2, distance ~1.99)
          → summary contains 'zetakeyword' → FTS rank 1

        RRF formula:
          score(a) = 1/(RRF_K+1) [vec rank 1]       = 0.0625
          score(b) = 1/(RRF_K+2) + 1/(RRF_K+1)     = 0.0588 + 0.0625 = 0.1213

        So ability_b must rank first in the merged output.
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
        rows = ability._query("zetakeyword", blob, 5, ["ability_a", "ability_b"])

        names = [r["key"] for r in rows]
        assert len(names) >= 2, f"Expected both abilities in results, got: {names}"
        assert names[0] == "ability_b", (
            f"ability_b should rank first (FTS win + vec rank 2 > vec-only rank 1). "
            f"Got order: {names}. "
            f"Expected score(b) = {1/(RRF_K+1) + 1/(RRF_K+2):.4f} > "
            f"score(a) = {1/(RRF_K+1):.4f}"
        )

    def test_fallback_keyword_search_queries_abilities_sqlite(self, tmp_path, monkeypatch):
        """_fallback hits ability_search_fts in abilities.sqlite and appends the
        hit to the bound processor's ACTIVE_TOOLS (never tool_capability_profiles)."""
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
        ability._fallback("sandbox", 5, ["sandboxer"])

        assert proc.active_tools == ["sandboxer"]

    def test_query_abilities_filters_by_allowlist(self, tmp_path, monkeypatch):
        """_query ignores rows whose name is outside the allowlist.

        Indexes two abilities; runs the query with only the second in `allow`.
        The first MUST NOT appear regardless of vec or FTS rank.
        """
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
        rows = ability._query("alphajet", blob, 5, ["allowed_ability"])

        names = {r["key"] for r in rows}
        assert names == {"allowed_ability"}, (
            f"Expected only 'allowed_ability' through the allowlist, got: {names}"
        )

