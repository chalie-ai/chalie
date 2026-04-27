"""
Tests for the search embedding cache feature.

Covers router.py, search.py, and generate_search_cache.py.
EmbeddingService is always mocked (569MB ONNX model).
"""

import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_embedding(dims=768, value=0.5):
    return [value] * dims


def _make_providers_db(tmp_path, providers=None, examples=None):
    db_path = tmp_path / "search_tool_providers.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE providers (id INTEGER PRIMARY KEY, name TEXT, enabled INTEGER DEFAULT 1)")
    conn.execute("CREATE TABLE provider_examples (id INTEGER PRIMARY KEY, provider_id INTEGER, example_query TEXT)")
    if providers:
        conn.executemany("INSERT INTO providers VALUES (?, ?, ?)", providers)
    if examples:
        conn.executemany("INSERT INTO provider_examples VALUES (?, ?, ?)", examples)
    conn.commit()
    conn.close()
    return db_path


class _FakeRow:
    def __init__(self, data): self._data = data
    def __getitem__(self, key): return self._data[key]
    def keys(self): return self._data.keys()


def _mock_router_conn(knn_rows, id_to_provider_rows):
    """Create a mock sqlite connection that returns knn_rows then id_to_provider_rows."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.side_effect = [knn_rows, id_to_provider_rows]
    return conn


# ── router.py ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRouteQuery:

    def test_returns_ranked_providers(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        conn = _mock_router_conn(
            knn_rows=[(1, 0.1), (2, 0.6)],  # brave=0.95, ddg=0.70
            id_to_provider_rows=[(1, 'brave'), (2, 'ddg')],
        )

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            result = r.route_query("privacy browser")

        assert len(result) >= 1
        assert result[0]['name'] == 'brave'
        assert result[0]['score'] > result[-1]['score'] if len(result) > 1 else True

    def test_gap_excludes_distant_providers(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        conn = _mock_router_conn(
            knn_rows=[(1, 0.1), (2, 0.8)],  # brave=0.95, poor=0.60 (gap > 0.10)
            id_to_provider_rows=[(1, 'brave'), (2, 'poor')],
        )

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            result = r.route_query("test")

        names = [p['name'] for p in result]
        assert 'brave' in names
        assert 'poor' not in names

    def test_max_3_providers(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        conn = _mock_router_conn(
            knn_rows=[(i, 0.01) for i in range(1, 6)],
            id_to_provider_rows=[(i, f'p{i}') for i in range(1, 6)],
        )

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            result = r.route_query("test")

        assert len(result) <= 3

    def test_min_score_filter(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        conn = _mock_router_conn(
            knn_rows=[(1, 1.2)],  # similarity = 0.4 < MIN_SCORE 0.50
            id_to_provider_rows=[(1, 'brave')],
        )

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            result = r.route_query("obscure query")

        assert result == []

    def test_embedding_failure_returns_empty(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.side_effect = RuntimeError("OOM")

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb):
            assert r.route_query("test") == []

    def test_db_failure_returns_empty(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', side_effect=sqlite3.OperationalError("no db")):
            assert r.route_query("test") == []

    def test_empty_knn_returns_empty(self):
        import tools.search.router as r
        mock_emb = MagicMock()
        mock_emb.generate_embedding.return_value = _make_embedding()

        conn = _mock_router_conn(knn_rows=[], id_to_provider_rows=[])

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            assert r.route_query("test") == []


# ── search.py ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSearchExecute:

    def _reset(self):
        import tools.search.search as s
        s._providers = None

    def test_empty_query(self):
        from tools.search.search import execute
        r = execute("topic", {})
        assert r['text'].startswith('EMPTY: no query supplied')

    def test_whitespace_query(self):
        from tools.search.search import execute
        r = execute("topic", {"query": "   "})
        assert r['text'].startswith('EMPTY: no query supplied')

    def test_forced_ddg(self):
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]) as m:
            from tools.search.search import execute
            r = execute("t", {"query": "test", "provider": "ddg"})
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')
        m.assert_called_once()

    def test_forced_known_provider(self, tmp_path):
        self._reset()
        db = _make_providers_db(tmp_path, providers=[(1, "brave", 1)])
        import tools.search.search as s
        with patch.object(s, '_DB', str(db)), \
             patch('tools.search.search.fetch_providers', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]) as m, \
             patch('tools.search.search.fetch_ddg_fallback'):
            r = s.execute("t", {"query": "test", "provider": "brave"})
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')
        m.assert_called_once()

    def test_forced_unknown_provider(self):
        self._reset()
        import tools.search.search as s
        s._providers = {'brave': {'name': 'brave'}}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[]):
            r = s.execute("t", {"query": "test", "provider": "nonexistent"})
        assert r['text'].startswith('EMPTY: zero results')
        assert 'Do NOT fabricate' in r['text']

    def test_router_failure_falls_back_to_ddg(self):
        self._reset()
        import tools.search.search as s
        s._providers = {}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]), \
             patch('tools.search.router.route_query', side_effect=Exception("boom")):
            r = s.execute("t", {"query": "test"})
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')

    def test_limit_clamped_high(self):
        self._reset()
        import tools.search.search as s
        s._providers = {}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[]) as m, \
             patch('tools.search.router.route_query', return_value=[]):
            s.execute("t", {"query": "test", "limit": 99})
        assert m.call_args[0][1] == 10

    def test_limit_clamped_low(self):
        self._reset()
        import tools.search.search as s
        s._providers = {}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[]) as m, \
             patch('tools.search.router.route_query', return_value=[]):
            s.execute("t", {"query": "test", "limit": -5})
        assert m.call_args[0][1] == 1

    def test_default_limit_5(self):
        self._reset()
        import tools.search.search as s
        s._providers = {}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[]) as m, \
             patch('tools.search.router.route_query', return_value=[]):
            s.execute("t", {"query": "test"})
        assert m.call_args[0][1] == 5

    def test_response_shape(self):
        self._reset()
        import tools.search.search as s
        s._providers = {}
        with patch('tools.search.search.fetch_ddg_fallback', return_value=[{"title": "r", "url": "http://x.com", "snippet": "desc"}]), \
             patch('tools.search.router.route_query', return_value=[]):
            r = s.execute("t", {"query": "test"})
        assert 'text' in r
        import json
        parsed = json.loads(r['text'])
        assert 'results' in parsed
        assert isinstance(parsed['results'], list)

    def test_load_providers_filters_disabled(self, tmp_path):
        self._reset()
        db = _make_providers_db(tmp_path, providers=[(1, "brave", 1), (2, "off", 0)])
        import tools.search.search as s
        with patch.object(s, '_DB', str(db)):
            p = s._load_providers()
        assert 'brave' in p
        assert 'off' not in p

    def test_load_providers_missing_db(self):
        self._reset()
        import tools.search.search as s
        with patch.object(s, '_DB', '/nonexistent/db'):
            p = s._load_providers()
        assert p == {}
        assert s._providers is None  # not cached on failure


# ── generate_search_cache.py ─────────────────────────────────────────────────

import importlib.util as _importlib_util  # noqa: E402

_GENERATOR = Path(__file__).resolve().parent.parent.parent / 'utils' / 'generate_search_cache.py'


def _load_gen():
    spec = _importlib_util.spec_from_file_location('gen', str(_GENERATOR))
    mod = _importlib_util.module_from_spec(spec)
    sys.modules.setdefault('sqlite_vec', types.ModuleType('sqlite_vec'))
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestGenerateSearchCache:

    def test_exits_on_missing_db(self, tmp_path):
        mod = _load_gen()
        mod._DB_PATH = tmp_path / "missing.sqlite"
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 1

    def test_exits_on_empty_examples(self):
        mod = _load_gen()
        mod._DB_PATH = Path(__file__)  # exists
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = []
        mod.sqlite3 = MagicMock()
        mod.sqlite3.connect.return_value = conn
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 1

    def test_embeds_all_examples(self):
        mod = _load_gen()
        mod._DB_PATH = Path(__file__)

        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [_make_embedding() for _ in range(3)]

        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            (1, "q1"), (2, "q2"), (3, "q3"),
        ]
        mod.sqlite3 = MagicMock()
        mod.sqlite3.connect.return_value = conn
        mod.EmbeddingService = MagicMock(return_value=mock_emb)

        fake_vec = types.ModuleType('sqlite_vec')
        fake_vec.load = MagicMock()
        with patch.dict('sys.modules', {'sqlite_vec': fake_vec}):
            mod.main()

        assert mock_emb.generate_embeddings_batch.call_count == 1
        assert len(mock_emb.generate_embeddings_batch.call_args[0][0]) == 3

    def test_batch_boundary(self):
        import numpy as np
        mod = _load_gen()
        mod._DB_PATH = Path(__file__)

        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.side_effect = lambda t: [np.zeros(768) for _ in t]

        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [(i, f"q{i}") for i in range(65)]
        mod.sqlite3 = MagicMock()
        mod.sqlite3.connect.return_value = conn
        mod.EmbeddingService = MagicMock(return_value=mock_emb)

        fake_vec = types.ModuleType('sqlite_vec')
        fake_vec.load = MagicMock()
        with patch.dict('sys.modules', {'sqlite_vec': fake_vec}):
            mod.main()

        assert mock_emb.generate_embeddings_batch.call_count == 3  # 32+32+1

