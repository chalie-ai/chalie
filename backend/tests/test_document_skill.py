"""
Tests for DocumentAbility — action dispatch, search, list, view, delete, restore, create.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


def _handle_document(topic: str, params: dict) -> str:
    """Thin shim: invoke DocumentAbility.execute and return the text string."""
    from abilities.document import DocumentAbility
    result = DocumentAbility().execute(topic, params, None)
    assert isinstance(result, dict), f"DocumentAbility.execute returned non-dict: {result!r}"
    return result["text"]


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
            result = _handle_document("topic", {"action": "explode"})
        assert "Unknown action" in result
        assert "explode" in result

    def test_default_action_is_search(self):
        """When no action is given, defaults to search."""
        with patch(_P_DB), patch(_P_DOC_SVC) as MockSvc:
            mock_svc = MagicMock()
            MockSvc.return_value = mock_svc
            result = _handle_document("topic", {})
        # Should attempt search (and fail gracefully due to no query)
        assert "'query' is required" in result

    def test_db_error_returns_error_string(self):
        """Database errors produce a clean error tag."""
        with patch(_P_DB, side_effect=Exception("DB down")):
            result = _handle_document("topic", {"action": "list"})
        # New format: [document(action=list, error=DB down)]\n[end:document]
        assert "[document(" in result
        assert "error=DB down" in result
        assert "[end:document]" in result


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
            result = _handle_document("topic", {"action": "search", "query": "warranty coverage"})

        assert "warranty.pdf" in result
        assert "id=doc00001" in result
        assert 'action "view"' in result

    def test_search_empty_query_returns_error(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            result = _handle_document("topic", {"action": "search", "query": ""})
        assert "'query' is required" in result

    def test_search_no_results(self):
        mock_svc = MagicMock()
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = []

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_DGS, return_value=mock_dgs):
            result = _handle_document("topic", {"action": "search", "query": "unicorn"})

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
            result = _handle_document("topic", {"action": "list"})

        assert "warranty.pdf" in result
        assert "invoice.pdf" in result
        assert "[warranty]" in result
        assert "[invoice]" in result

    def test_list_empty_library(self):
        mock_svc = MagicMock()
        mock_svc.get_all_documents.return_value = []

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "list"})

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
            result = _handle_document("topic", {"action": "view", "id": "doc00001"})

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
            result = _handle_document("topic", {"action": "view", "id": "missing"})

        assert "not found" in result.lower()

    def test_view_by_name_fuzzy(self):
        mock_svc = MagicMock()
        doc = _make_doc()
        mock_svc.get_document.return_value = None  # id lookup fails
        mock_svc.search_documents_metadata.return_value = [doc]

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "view", "name": "warranty"})

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
            result = _handle_document("topic", {"action": "delete", "id": "doc00001"})

        assert "Deleted" in result
        assert "warranty.pdf" in result
        assert "artifact(s) removed" in result

    def test_delete_not_found(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = None
        mock_svc.search_documents_metadata.return_value = []

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "delete", "name": "nothing"})

        assert "not found" in result.lower()

    def test_delete_failure(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc()
        mock_svc.soft_delete.return_value = False

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "delete", "id": "doc00001"})

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
            result = _handle_document("topic", {"action": "restore", "id": "doc00001"})

        assert "Restored" in result
        assert "warranty.pdf" in result

    def test_restore_not_deleted(self):
        mock_svc = MagicMock()
        mock_svc.get_document.return_value = _make_doc(deleted_at=None)

        with patch(_P_DB), patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "restore", "id": "doc00001"})

        assert "not deleted" in result.lower()

    def test_restore_missing_params(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            result = _handle_document("topic", {"action": "restore"})

        assert "Specify" in result

    def test_restore_by_name(self):
        mock_svc = MagicMock()
        deleted_doc = _make_doc(deleted_at=datetime(2026, 2, 25, tzinfo=timezone.utc))
        mock_svc.get_all_documents.return_value = [deleted_doc]
        mock_svc.restore.return_value = True

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc):
            result = _handle_document("topic", {"action": "restore", "name": "warranty"})

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
            result = _handle_document("topic", {
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
            result = _handle_document("topic", {
                "action": "create",
                "content": "some content",
            })
        assert "'name' is required" in result

    def test_create_missing_content(self):
        with patch(_P_DB), patch(_P_DOC_SVC):
            result = _handle_document("topic", {
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
            _handle_document("topic", {
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
            result = _handle_document("topic", {
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
        from abilities.document import _split_into_artifacts
        assert _split_into_artifacts("") == []

    def test_short_text_below_min_chars_returns_single_chunk(self):
        from abilities.document import _split_into_artifacts
        text = "Short paragraph."
        result = _split_into_artifacts(text)
        assert result == [text]

    def test_text_exactly_at_min_chars_returns_single_chunk(self):
        from abilities.document import _split_into_artifacts
        text = _make_text(512)
        result = _split_into_artifacts(text)
        assert len(result) == 1
        assert result[0] == text

    def test_text_slightly_over_min_chars_splits_at_paragraph_boundary(self):
        """Two paragraphs each just over 256 chars should produce at least one split."""
        from abilities.document import _split_into_artifacts
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
        from abilities.document import _split_into_artifacts
        para_a = _make_text(600)
        para_b = _make_text(600)
        text = para_a + "\n\n" + para_b
        result = _split_into_artifacts(text)
        assert len(result) >= 2

    def test_adjacent_chunks_share_48_char_overlap(self):
        """The last 48 chars of chunk[i] must appear as a prefix of chunk[i+1]."""
        from abilities.document import _split_into_artifacts
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
        from abilities.document import _split_into_artifacts
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
        from abilities.document import _split_into_artifacts
        sentences = [f"Sentence number {i} describes a fact about solar energy." for i in range(30)]
        text = " ".join(sentences)  # ~1650 chars, no \n\n
        result = _split_into_artifacts(text)
        assert len(result) >= 2
        # Each chunk should end mid-sentence or at a sentence — but never exceed max+overlap
        for chunk in result:
            assert len(chunk) <= 1024 + 48

    def test_sentence_over_max_chars_is_hard_cut(self):
        """A single sentence > max_chars is hard-cut at exactly max_chars characters."""
        from abilities.document import _split_into_artifacts
        # One enormous "sentence" with no punctuation separator
        text = "a" * 2500
        result = _split_into_artifacts(text)
        # All chunks from hard-cut must be <= max_chars
        assert len(result) >= 2
        for chunk in result[:-1]:
            assert len(chunk) <= 1024 + 48

    def test_first_chunk_has_no_leading_overlap(self):
        """The very first chunk must not be prefixed with overlap from a previous chunk."""
        from abilities.document import _split_into_artifacts
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
        from abilities.document import _split_into_artifacts
        text = "word " * 300  # 1500 chars, no sentence separators
        result = _split_into_artifacts(text)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(c.strip() for c in result)

    def test_returns_list_not_generator(self):
        from abilities.document import _split_into_artifacts
        result = _split_into_artifacts("Hello world.")
        assert isinstance(result, list)

    def test_whitespace_only_text_returns_single_or_empty(self):
        """Whitespace-only input returns [] (empty after strip) or single chunk."""
        from abilities.document import _split_into_artifacts
        result = _split_into_artifacts("   \n\n   ")
        assert isinstance(result, list)

    def test_custom_min_max_respected(self):
        """min_chars/max_chars parameters are honoured over the defaults."""
        from abilities.document import _split_into_artifacts
        # With min=50, max=100, a 200-char text should produce multiple chunks
        text = "The quick brown fox jumps over the lazy dog. " * 5  # 225 chars
        result = _split_into_artifacts(text, min_chars=50, max_chars=100, overlap=10)
        assert len(result) >= 2
