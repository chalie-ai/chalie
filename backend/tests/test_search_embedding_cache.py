"""
Tests for the search embedding cache feature.

Covers router.py, search.py, and generate_search_cache.py.
EmbeddingService is always mocked (569MB ONNX model).
"""

import sqlite3
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from services.file_mapper_service import FileMapperService


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
            knn_rows=[(1, 0.1), (2, 0.6)],  # resolves to brave at 0.95 and ddg at 0.70
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
            knn_rows=[(1, 0.1), (2, 0.8)],  # brave scores 0.95, second candidate 0.60, gap exceeds threshold
            id_to_provider_rows=[(1, 'brave'), (2, 'poor')],
        )

        with patch('services.embedding_service.EmbeddingService', return_value=mock_emb), \
             patch('tools.search.router.sqlite3.connect', return_value=conn):
            result = r.route_query("test")

        names = [p['name'] for p in result]
        assert 'brave' in names
        assert 'poor' not in names

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


# ── search.py ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSearchExecute:

    def _reset(self):
        from abilities.search import SearchAbility
        SearchAbility._providers = None

    def test_empty_query(self):
        from abilities.search import SearchAbility
        r = SearchAbility().execute("topic", {}, None)
        assert r['text'].startswith('EMPTY: no query supplied')

    def test_forced_ddg(self):
        with patch('abilities.search.fetch_ddg_fallback', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]) as m:
            from abilities.search import SearchAbility
            r = SearchAbility().execute("t", {"query": "test", "provider": "ddg"}, None)
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')
        m.assert_called_once()

    def test_forced_known_provider(self, tmp_path):
        self._reset()
        db = _make_providers_db(tmp_path, providers=[(1, "brave", 1)])
        from abilities.search import SearchAbility
        with patch.object(SearchAbility, '_DB', str(db)), \
             patch('abilities.search.fetch_providers', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]) as m, \
             patch('abilities.search.fetch_ddg_fallback'):
            r = SearchAbility().execute("t", {"query": "test", "provider": "brave"}, None)
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')
        m.assert_called_once()

    def test_router_failure_falls_back_to_ddg(self):
        self._reset()
        from abilities.search import SearchAbility
        SearchAbility._providers = {}
        with patch('abilities.search.fetch_ddg_fallback', return_value=[{"title": "r", "url": "http://x.com", "snippet": ""}]), \
             patch('tools.search.router.route_query', side_effect=Exception("boom")):
            r = SearchAbility().execute("t", {"query": "test"}, None)
        assert 'text' in r
        assert not r['text'].startswith('EMPTY:')

    def test_load_providers_filters_disabled(self, tmp_path):
        self._reset()
        db = _make_providers_db(tmp_path, providers=[(1, "brave", 1), (2, "off", 0)])
        from abilities.search import SearchAbility
        with patch.object(SearchAbility, '_DB', str(db)):
            p = SearchAbility._load_providers()
        assert 'brave' in p
        assert 'off' not in p




# ── generate_search_cache.py ─────────────────────────────────────────────────

import importlib.util as _importlib_util  # noqa: E402
import re as _re  # noqa: E402

_GENERATOR = FileMapperService.get_backend_path("utils", "generate_search_cache.py")

_VEC0_DDL = _re.compile(
    r'CREATE\s+VIRTUAL\s+TABLE\s+(\S+)\s+USING\s+vec0\([^)]*\)',
    _re.IGNORECASE,
)


class _VecConn:
    """Thin wrapper around a real sqlite3 connection.

    Rewrites ``CREATE VIRTUAL TABLE … USING vec0(…)`` DDL into a plain table
    so tests can run without the sqlite-vec extension installed.
    ``enable_load_extension`` is silenced because it raises on platforms where
    the extension API is disabled.
    """

    def __init__(self, conn):
        self._conn = conn

    def enable_load_extension(self, _flag):  # noqa: ANN001
        pass  # no-op: extension loading not needed in tests

    def execute(self, sql, *args):
        m = _VEC0_DDL.match(sql.strip())
        if m:
            table = m.group(1)
            sql = f"CREATE TABLE {table} (rowid INTEGER PRIMARY KEY, embedding BLOB)"
        return self._conn.execute(sql, *args)

    def commit(self):
        return self._conn.commit()

    def close(self):
        return self._conn.close()

    # Delegate any attribute not explicitly defined (e.g. ``load_extension``).
    def __getattr__(self, name):
        return getattr(self._conn, name)


def _make_providers_file(tmp_path, examples):
    """Create a real SQLite file with provider_examples rows and return its Path."""
    db_path = tmp_path / "search_tool_providers.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE provider_examples "
        "(id INTEGER PRIMARY KEY, provider_id INTEGER, example_query TEXT)"
    )
    conn.executemany(
        "INSERT INTO provider_examples VALUES (?, ?, ?)",
        examples,
    )
    conn.commit()
    conn.close()
    return db_path


def _load_gen():
    spec = _importlib_util.spec_from_file_location('gen', str(_GENERATOR))
    mod = _importlib_util.module_from_spec(spec)
    sys.modules.setdefault('sqlite_vec', types.ModuleType('sqlite_vec'))
    spec.loader.exec_module(mod)
    return mod


def _patch_gen(mod, db_path, mock_emb):
    """Patch the module so it uses a _VecConn-wrapped real connection."""
    real_sqlite3 = sqlite3

    def _connect(path, **kw):
        return _VecConn(real_sqlite3.connect(path, **kw))

    fake_sqlite3 = types.ModuleType('sqlite3')
    fake_sqlite3.connect = _connect

    fake_vec = types.ModuleType('sqlite_vec')
    fake_vec.load = MagicMock()

    mod._DB_PATH = db_path
    mod.sqlite3 = fake_sqlite3
    mod.EmbeddingService = MagicMock(return_value=mock_emb)

    return fake_vec


@pytest.mark.unit
class TestGenerateSearchCache:

    def test_embeds_all_examples(self, tmp_path):
        """All 3 examples are embedded and inserted into example_embeddings."""
        examples = [(1, 1, "q1"), (2, 1, "q2"), (3, 1, "q3")]
        db_path = _make_providers_file(tmp_path, examples)

        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.return_value = [[0.1] * 768 for _ in range(3)]

        mod = _load_gen()
        fake_vec = _patch_gen(mod, db_path, mock_emb)
        with patch.dict('sys.modules', {'sqlite_vec': fake_vec}):
            mod.main()

        # Exactly one batch call covering all 3 examples.
        assert mock_emb.generate_embeddings_batch.call_count == 1
        assert mock_emb.generate_embeddings_batch.call_args[0][0] == ["q1", "q2", "q3"]

        # Verify actual DB state: 3 rows written, rowids match example ids.
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT rowid FROM example_embeddings ORDER BY rowid"
        ).fetchall()
        conn.close()
        assert len(rows) == 3
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_batch_boundary(self, tmp_path):
        """65 examples split into 3 batches (32 + 32 + 1), all rows inserted."""
        examples = [(i, 1, f"q{i}") for i in range(1, 66)]
        db_path = _make_providers_file(tmp_path, examples)

        mock_emb = MagicMock()
        mock_emb.generate_embeddings_batch.side_effect = (
            lambda texts: [[0.0] * 768 for _ in texts]
        )

        mod = _load_gen()
        fake_vec = _patch_gen(mod, db_path, mock_emb)
        with patch.dict('sys.modules', {'sqlite_vec': fake_vec}):
            mod.main()

        assert mock_emb.generate_embeddings_batch.call_count == 3  # 32 + 32 + 1

        # Verify all 65 rows are persisted.
        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM example_embeddings"
        ).fetchone()[0]
        conn.close()
        assert count == 65


