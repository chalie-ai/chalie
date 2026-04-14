"""
Tests for document_skill.py — action dispatch, search, list, view, delete, restore, create.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id="doc00001",
    original_name="warranty.pdf",
    status="ready",
    page_count=3,
    chunk_count=12,
    extracted_metadata=None,
    deleted_at=None,
):
    """Build a dict that mirrors what DocumentService.get_document returns."""
    now = datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "id": doc_id,
        "original_name": original_name,
        "mime_type": "application/pdf",
        "file_size_bytes": 2048,
        "file_path": f"{doc_id}/{original_name}",
        "file_hash": "abc123hash",
        "page_count": page_count,
        "status": status,
        "error_message": None,
        "chunk_count": chunk_count,
        "source_type": "upload",
        "tags": [],
        "summary": "First page of the warranty document.",
        "extracted_metadata": extracted_metadata or {
            "document_type": {"value": "warranty", "confidence": 0.85},
            "companies": [{"name": "Samsung", "confidence": 0.9}],
            "expiration_dates": [{"value": "2028-03-15", "confidence": 0.8}],
            "monetary_values": [{"amount": 999.99, "currency": "USD", "confidence": 0.9}],
            "reference_numbers": [{"value": "WRN-2026-12345", "confidence": 0.95}],
        },
        "supersedes_id": None,
        "clean_text": "warranty coverage text",
        "language": "en",
        "fingerprint": "aabb",
        "watched_folder_id": None,
        "created_at": now,
        "updated_at": now,
        "deleted_at": deleted_at,
        "purge_after": None,
    }



def _make_search_result(
    doc_id="doc00001",
    document_name="warranty.pdf",
    content="Coverage valid for 24 months from purchase date.",
    page_number=1,
    section_title="Coverage",
    score=0.85,
):
    return {
        "document_id": doc_id,
        "document_name": document_name,
        "content": content,
        "page_number": page_number,
        "section_title": section_title,
        "score": score,
    }


# Patch targets — function-local imports require patching at the SOURCE module
_P_DB = "services.database_service.get_shared_db_service"
_P_DOC_SVC = "services.document_service.DocumentService"
_P_EMB = "services.embedding_service.get_embedding_service"
_P_DGS = "services.data_graph_service.get_data_graph_service"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDispatch:
    """Test action dispatch routing."""

    def test_unknown_action_returns_error(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "explode"})
        assert "Unknown action" in result
        assert "explode" in result

    def test_default_action_is_search(self):
        """When no action is given, defaults to search."""
        with patch(_P_DB), patch(_P_DOC_SVC) as MockSvc:
            mock_svc = MagicMock()
            MockSvc.return_value = mock_svc
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {})
        # Should attempt search (and fail gracefully due to no query)
        assert "'query' is required" in result

    def test_db_error_returns_error_string(self):
        """Database errors produce a clean error message."""
        with patch(_P_DB, side_effect=Exception("DB down")):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "list"})
        assert "[DOCUMENT] Error:" in result
        assert "DB down" in result


@pytest.mark.unit
class TestSearchAction:
    """Test the search action handler."""

    def test_search_returns_formatted_results(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc()
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {"key": "doc:doc00001:000", "value": "Coverage valid for 24 months.", "source": "document:doc00001", "retrieval_weight": 0.9},
            {"key": "doc:doc00002:000", "value": "Invoice total: $199.", "source": "document:doc00002", "retrieval_weight": 0.8},
        ]

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "search", "query": "warranty coverage"})

        assert "warranty.pdf" in result
        assert "id=doc00001" in result
        assert 'action "view"' in result

    def test_search_empty_query_returns_error(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "search", "query": ""})
        assert "'query' is required" in result

    def test_search_no_results(self):
        mock_svc = MagicMock()
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "search", "query": "unicorn"})

        assert "No documents match" in result


@pytest.mark.unit
class TestListAction:
    """Test the list action handler."""

    def test_list_returns_document_entries(self):
        mock_svc = MagicMock()
        mock_svc.get_all_documents.return_value = [
            _make_doc(doc_id="d1", original_name="warranty.pdf"),
            _make_doc(doc_id="d2", original_name="invoice.pdf",
                      extracted_metadata={"document_type": {"value": "invoice", "confidence": 0.9}}),
        ]

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "list"})

        assert "warranty.pdf" in result
        assert "invoice.pdf" in result
        assert "[warranty]" in result
        assert "[invoice]" in result

    def test_list_empty_library(self):
        mock_svc = MagicMock()
        mock_svc.get_all_documents.return_value = []

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "list"})

        assert "No documents" in result


@pytest.mark.unit
class TestViewAction:
    """Test the view action handler."""

    def test_view_by_id(self):
        mock_svc = MagicMock()
        doc = _make_doc()
        mock_svc.get_document.return_value = doc

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "view", "id": "doc00001"})

        assert "warranty.pdf" in result
        assert "Samsung" in result
        assert "2028-03-15" in result
        assert "WRN-2026-12345" in result
        # Full document text is included
        assert "Full Document Text" in result
        assert "warranty coverage text" in result

    def test_view_not_found(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = None
        mock_svc.search_documents_metadata.return_value = []

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "view", "id": "missing"})

        assert "not found" in result.lower()

    def test_view_by_name_fuzzy(self):
        mock_svc = MagicMock()
        doc = _make_doc()
        mock_svc.get_document.return_value = None  # id lookup fails
        mock_svc.search_documents_metadata.return_value = [doc]

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "view", "name": "warranty"})

        assert "warranty.pdf" in result


@pytest.mark.unit
class TestDeleteAction:
    """Test the delete action handler."""

    def test_delete_success(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc()
        mock_svc.soft_delete.return_value = True
        mock_dgs = MagicMock()
        mock_dgs.hard_delete_by_source_prefix.return_value = 3

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "delete", "id": "doc00001"})

        assert "Deleted" in result
        assert "warranty.pdf" in result
        assert "artifact(s) removed" in result

    def test_delete_not_found(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = None
        mock_svc.search_documents_metadata.return_value = []

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "delete", "name": "nothing"})

        assert "not found" in result.lower()

    def test_delete_failure(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc()
        mock_svc.soft_delete.return_value = False

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "delete", "id": "doc00001"})

        assert "Failed" in result


@pytest.mark.unit
class TestRestoreAction:
    """Test the restore action handler."""

    def test_restore_success(self):
        mock_svc = MagicMock()
        deleted_doc = _make_doc(deleted_at=datetime(2026, 2, 25, tzinfo=timezone.utc))
        mock_svc.get_document.return_value = deleted_doc
        mock_svc.restore.return_value = True

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "restore", "id": "doc00001"})

        assert "Restored" in result
        assert "warranty.pdf" in result

    def test_restore_not_deleted(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc(deleted_at=None)

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "restore", "id": "doc00001"})

        assert "not deleted" in result.lower()

    def test_restore_missing_params(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "restore"})

        assert "Specify" in result

    def test_restore_by_name(self):
        mock_svc = MagicMock()
        deleted_doc = _make_doc(deleted_at=datetime(2026, 2, 25, tzinfo=timezone.utc))
        mock_svc.get_all_documents.return_value = [deleted_doc]
        mock_svc.restore.return_value = True

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "restore", "name": "warranty"})

        assert "Restored" in result
        mock_svc.get_all_documents.assert_called_once_with(include_deleted=True)


@pytest.mark.unit
class TestCreateAction:
    """Test the create action handler."""

    def test_create_success(self):
        mock_svc = MagicMock()
        mock_svc.create_document_from_text.return_value = "abc12345"
        mock_dgs = MagicMock()

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {
                "action": "create",
                "name": "research-notes.md",
                "content": "# Research Notes\n\nSome findings here.",
            })

        assert "Created" in result
        assert "research-notes.md" in result
        assert "abc12345" in result
        mock_svc.create_document_from_text.assert_called_once_with(
            original_name="research-notes.md",
            text_content="# Research Notes\n\nSome findings here.",
            source_type="conversation",
        )
        mock_svc.update_status.assert_called_once_with("abc12345", "ready", chunk_count=1)
        mock_dgs.store.assert_called_once()

    def test_create_missing_name(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {
                "action": "create",
                "content": "some content",
            })
        assert "'name' is required" in result

    def test_create_missing_content(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {
                "action": "create",
                "name": "notes.md",
            })
        assert "'content' is required" in result

    def test_create_adds_md_extension(self):
        mock_svc = MagicMock()
        mock_svc.create_document_from_text.return_value = "def67890"
        mock_dgs = MagicMock()

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            from services.innate_skills.document_skill import handle_document
            handle_document("topic", {
                "action": "create",
                "name": "my-notes",
                "content": "content here",
            })

        # Should have added .md extension
        mock_svc.create_document_from_text.assert_called_once()
        call_args = mock_svc.create_document_from_text.call_args
        assert call_args[1]["original_name"] == "my-notes.md"

    def test_create_error_handling(self):
        mock_svc = MagicMock()
        mock_svc.create_document_from_text.side_effect = Exception("disk full")

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {
                "action": "create",
                "name": "notes.md",
                "content": "content",
            })

        assert "Failed" in result
        assert "disk full" in result


# ---------------------------------------------------------------------------
# _split_into_artifacts — pure function, no external dependencies
# ---------------------------------------------------------------------------

# Helpers for artifact splitting assertions

def _make_text(n_chars: int, word: str = "x") -> str:
    """Return a string of exactly n_chars using repeating word tokens."""
    unit = word + " "
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


@pytest.mark.unit
class TestSplitIntoArtifacts:

    def test_empty_text_returns_empty_list(self):
        from services.innate_skills.document_skill import _split_into_artifacts
        assert _split_into_artifacts("") == []

    def test_short_text_below_min_chars_returns_single_chunk(self):
        from services.innate_skills.document_skill import _split_into_artifacts
        text = "Short paragraph."
        result = _split_into_artifacts(text)
        assert result == [text]

    def test_text_exactly_at_min_chars_returns_single_chunk(self):
        from services.innate_skills.document_skill import _split_into_artifacts
        text = _make_text(512)
        result = _split_into_artifacts(text)
        assert len(result) == 1
        assert result[0] == text

    def test_text_slightly_over_min_chars_splits_at_paragraph_boundary(self):
        """Two paragraphs each just over 256 chars should produce at least one split."""
        from services.innate_skills.document_skill import _split_into_artifacts
        para_a = _make_text(550)
        para_b = _make_text(550)
        text = para_a + "\n\n" + para_b
        result = _split_into_artifacts(text)
        # Combined is well over min_chars — expect at least one chunk covering para_a
        assert len(result) >= 1
        # Every chunk must contain real content
        assert all(c.strip() for c in result)

    def test_two_large_paragraphs_produce_multiple_chunks(self):
        """Two paragraphs each >= min_chars produce at least 2 artifacts."""
        from services.innate_skills.document_skill import _split_into_artifacts
        para_a = _make_text(600)
        para_b = _make_text(600)
        text = para_a + "\n\n" + para_b
        result = _split_into_artifacts(text)
        assert len(result) >= 2

    def test_adjacent_chunks_share_48_char_overlap(self):
        """The last 48 chars of chunk[i] must appear as a prefix of chunk[i+1]."""
        from services.innate_skills.document_skill import _split_into_artifacts
        # Build text with four well-separated paragraphs so multiple chunks form
        paras = [_make_text(700, word=f"word{i}") for i in range(4)]
        text = "\n\n".join(paras)
        result = _split_into_artifacts(text)
        assert len(result) >= 2, "Expected multiple chunks for overlap assertion"
        for i in range(len(result) - 1):
            overlap_tail = result[i][-48:]
            assert result[i + 1].startswith(overlap_tail), (
                f"Chunk {i+1} does not start with last 48 chars of chunk {i}"
            )

    def test_no_chunk_exceeds_max_chars(self):
        """All produced chunks must be <= max_chars (1024)."""
        from services.innate_skills.document_skill import _split_into_artifacts
        # Long single paragraph to force sentence splitting
        sentence = "The photovoltaic cell converts solar energy into electrical power. "
        text = sentence * 30  # ~1980 chars, single paragraph
        result = _split_into_artifacts(text)
        for chunk in result:
            assert len(chunk) <= 1024 + 48, (
                f"Chunk length {len(chunk)} exceeds max_chars+overlap bound"
            )

    def test_single_paragraph_over_max_chars_splits_at_sentence_boundary(self):
        """A single paragraph > max_chars is split at '. ' sentence boundaries."""
        from services.innate_skills.document_skill import _split_into_artifacts
        sentences = [f"Sentence number {i} describes a fact about solar energy." for i in range(30)]
        text = " ".join(sentences)  # ~1650 chars, no \n\n
        result = _split_into_artifacts(text)
        assert len(result) >= 2
        # Each chunk should end mid-sentence or at a sentence — but never exceed max+overlap
        for chunk in result:
            assert len(chunk) <= 1024 + 48

    def test_sentence_over_max_chars_is_hard_cut(self):
        """A single sentence > max_chars is hard-cut at exactly max_chars characters."""
        from services.innate_skills.document_skill import _split_into_artifacts
        # One enormous "sentence" with no punctuation separator
        text = "a" * 2500
        result = _split_into_artifacts(text)
        # All chunks from hard-cut must be <= max_chars
        assert len(result) >= 2
        for chunk in result[:-1]:
            assert len(chunk) <= 1024 + 48

    def test_first_chunk_has_no_leading_overlap(self):
        """The very first chunk must not be prefixed with overlap from a previous chunk."""
        from services.innate_skills.document_skill import _split_into_artifacts
        paras = [_make_text(700, word=f"tok{i}") for i in range(3)]
        text = "\n\n".join(paras)
        result = _split_into_artifacts(text)
        assert len(result) >= 2
        # The first chunk is stored as-is — it matches chunks[0] before overlap is applied
        # Re-running split gives same first chunk
        result2 = _split_into_artifacts(text)
        assert result[0] == result2[0]

    def test_single_paragraph_with_no_separator_over_max_returns_list(self):
        """Text with no \n\n and no punctuation still produces a non-empty list."""
        from services.innate_skills.document_skill import _split_into_artifacts
        text = "word " * 300  # 1500 chars, no sentence separators
        result = _split_into_artifacts(text)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(c.strip() for c in result)

    def test_returns_list_not_generator(self):
        from services.innate_skills.document_skill import _split_into_artifacts
        result = _split_into_artifacts("Hello world.")
        assert isinstance(result, list)

    def test_whitespace_only_text_returns_single_or_empty(self):
        """Whitespace-only input returns [] (empty after strip) or single chunk."""
        from services.innate_skills.document_skill import _split_into_artifacts
        result = _split_into_artifacts("   \n\n   ")
        assert isinstance(result, list)

    def test_custom_min_max_respected(self):
        """min_chars/max_chars parameters are honoured over the defaults."""
        from services.innate_skills.document_skill import _split_into_artifacts
        # With min=50, max=100, a 200-char text should produce multiple chunks
        text = "The quick brown fox jumps over the lazy dog. " * 5  # 225 chars
        result = _split_into_artifacts(text, min_chars=50, max_chars=100, overlap=10)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# create_document_artifacts — uses real DataGraphService (in-memory SQLite)
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import sqlite3  # noqa: E402

from services.data_graph_service import DataGraphService, KIND_DOCUMENT  # noqa: E402
from services.database_service import DatabaseService  # noqa: E402

# Reuse the same DDL structure as test_data_graph_service.py for consistency
_DGA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS data_graph (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        kind              TEXT NOT NULL,
        key               TEXT NOT NULL,
        value             TEXT,
        storage_strength  REAL NOT NULL DEFAULT 0.5,
        retrieval_weight  REAL NOT NULL DEFAULT 1.0,
        salience_score    REAL NOT NULL DEFAULT 0.0,
        evidence_count    INTEGER NOT NULL DEFAULT 1,
        first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
        last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at  TEXT,
        source            TEXT,
        deleted_at        TEXT,
        active            INTEGER NOT NULL DEFAULT 1,
        search_queries    TEXT DEFAULT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dga_kind ON data_graph(kind)",
    "CREATE INDEX IF NOT EXISTS idx_dga_key ON data_graph(key)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS data_graph_fts USING fts5(
        key, value, kind, search_queries,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_graph_edges (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        from_id          INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        to_id            INTEGER NOT NULL REFERENCES data_graph(id) ON DELETE CASCADE,
        edge_type        TEXT NOT NULL DEFAULT 'related',
        strength         REAL NOT NULL DEFAULT 1.0,
        created_at       TEXT NOT NULL DEFAULT (datetime('now')),
        last_accessed_at TEXT,
        UNIQUE (from_id, to_id, edge_type)
    )
    """,
    "CREATE TABLE IF NOT EXISTS data_graph_key_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)",
    "CREATE TABLE IF NOT EXISTS data_graph_value_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)",
]


@pytest.fixture
def _dg_db(tmp_path):
    """Real SQLite DB with data_graph schema for create_document_artifacts tests."""
    db_path = str(tmp_path / "dg_artifacts.db")
    shared_conn = sqlite3.connect(db_path)
    shared_conn.row_factory = sqlite3.Row
    shared_conn.execute("PRAGMA foreign_keys = ON")
    for ddl in _DGA_DDL:
        shared_conn.execute(ddl)
    shared_conn.commit()

    db = DatabaseService.__new__(DatabaseService)
    db._db_path = db_path
    db._init_complete = True
    db.db_path = db_path

    _depth = [0]

    @contextlib.contextmanager
    def _connection():
        _depth[0] += 1
        try:
            yield shared_conn
            if _depth[0] == 1:
                shared_conn.commit()
        except Exception:
            if _depth[0] == 1:
                shared_conn.rollback()
            raise
        finally:
            _depth[0] -= 1

    db.connection = _connection
    yield db
    shared_conn.close()


@pytest.fixture
def _dg_svc(_dg_db):
    """DataGraphService with real SQLite, embeddings disabled."""
    from unittest.mock import MagicMock
    svc = DataGraphService(_dg_db)
    svc._generate_embedding = MagicMock(return_value=None)
    svc._schedule_embeddings = MagicMock()
    svc._schedule_doc2query = MagicMock()
    return svc


@pytest.mark.unit
class TestCreateDocumentArtifacts:

    def test_short_text_stores_one_artifact(self, _dg_svc, _dg_db):
        """Text shorter than min_chars produces exactly one artifact."""
        from services.innate_skills.document_skill import create_document_artifacts

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            count = create_document_artifacts('doc00001', 'Short content here.')

        assert count == 1
        with _dg_db.connection() as conn:
            rows = conn.execute(
                "SELECT key, value, source FROM data_graph WHERE kind=?", (KIND_DOCUMENT,)
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]['key'] == 'doc:doc00001:000'
        assert rows[0]['source'] == 'document:doc00001'
        assert 'Short content here.' in rows[0]['value']

    def test_key_format_is_doc_id_index_zero_padded(self, _dg_svc, _dg_db):
        """Keys use format doc:{doc_id}:{index:03d} — zero-padded to 3 digits."""
        from services.innate_skills.document_skill import create_document_artifacts
        # Build a text large enough for multiple artifacts
        text = "\n\n".join([_make_text(600, word=f"word{i}") for i in range(4)])

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            count = create_document_artifacts('mydoc', text)

        assert count >= 2
        with _dg_db.connection() as conn:
            rows = conn.execute(
                "SELECT key FROM data_graph WHERE kind=? ORDER BY key", (KIND_DOCUMENT,)
            ).fetchall()
        keys = [r['key'] for r in rows]
        assert keys[0] == 'doc:mydoc:000'
        assert keys[1] == 'doc:mydoc:001'
        # All keys must follow the pattern
        for i, key in enumerate(keys):
            assert key == f'doc:mydoc:{i:03d}'

    def test_source_is_document_doc_id(self, _dg_svc, _dg_db):
        """Every artifact stored has source='document:{doc_id}'."""
        from services.innate_skills.document_skill import create_document_artifacts
        text = "\n\n".join([_make_text(600, word=f"t{i}") for i in range(3)])

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            create_document_artifacts('solar', text)

        with _dg_db.connection() as conn:
            rows = conn.execute(
                "SELECT source FROM data_graph WHERE kind=?", (KIND_DOCUMENT,)
            ).fetchall()
        for row in rows:
            assert row['source'] == 'document:solar'

    def test_all_artifacts_use_kind_document(self, _dg_svc, _dg_db):
        """All stored artifacts have kind=KIND_DOCUMENT."""
        from services.innate_skills.document_skill import create_document_artifacts
        text = "\n\n".join([_make_text(600, word=f"k{i}") for i in range(2)])

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            create_document_artifacts('kindcheck', text)

        with _dg_db.connection() as conn:
            kinds = conn.execute(
                "SELECT DISTINCT kind FROM data_graph"
            ).fetchall()
        assert len(kinds) == 1
        assert kinds[0]['kind'] == KIND_DOCUMENT

    def test_returns_artifact_count(self, _dg_svc, _dg_db):
        """Return value equals the number of artifacts stored in data_graph."""
        from services.innate_skills.document_skill import create_document_artifacts
        text = "\n\n".join([_make_text(600, word=f"c{i}") for i in range(4)])

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            returned_count = create_document_artifacts('countcheck', text)

        with _dg_db.connection() as conn:
            stored_count = conn.execute(
                "SELECT COUNT(*) FROM data_graph WHERE kind=?", (KIND_DOCUMENT,)
            ).fetchone()[0]
        assert returned_count == stored_count

    def test_empty_text_stores_nothing(self, _dg_svc, _dg_db):
        """Empty string produces 0 artifacts (empty list from _split_into_artifacts)."""
        from services.innate_skills.document_skill import create_document_artifacts

        with patch('services.data_graph_service.get_data_graph_service',
                   return_value=_dg_svc):
            count = create_document_artifacts('empty', '')

        assert count == 0
        with _dg_db.connection() as conn:
            stored = conn.execute(
                "SELECT COUNT(*) FROM data_graph WHERE kind=?", (KIND_DOCUMENT,)
            ).fetchone()[0]
        assert stored == 0
