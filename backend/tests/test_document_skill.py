"""
Tests for document_skill.py — action dispatch, search, list, view, delete, restore.
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
        "created_at": now,
        "updated_at": now,
        "deleted_at": deleted_at,
        "purge_after": None,
    }


def _make_chunk(
    doc_id="doc00001",
    chunk_index=0,
    content="This warranty covers manufacturer defects for 24 months.",
    page_number=1,
    section_title="Coverage",
):
    return {
        "document_id": doc_id,
        "chunk_index": chunk_index,
        "content": content,
        "page_number": page_number,
        "section_title": section_title,
        "token_count": 15,
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
_P_CARD = "services.document_card_service.DocumentCardService"


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
        results = [
            _make_search_result(),
            _make_search_result(doc_id="doc00002", document_name="invoice.pdf", page_number=2),
        ]
        mock_svc.search_chunks.return_value = results
        mock_svc.get_document.return_value = _make_doc()

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_EMB) as mock_emb, \
             patch(_P_CARD):
            mock_emb_svc = MagicMock()
            mock_emb_svc.generate_embedding.return_value = [0.1] * 768
            mock_emb.return_value = mock_emb_svc
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "search", "query": "warranty coverage"})

        # Search returns lightweight identification, not content
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
        mock_svc.search_chunks.return_value = []

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_EMB) as mock_emb:
            mock_emb_svc = MagicMock()
            mock_emb_svc.generate_embedding.return_value = [0.1] * 768
            mock_emb.return_value = mock_emb_svc
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
        mock_svc.get_chunks_for_document.return_value = [_make_chunk()]

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_CARD):
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

        with patch(_P_DB), \
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_CARD):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "delete", "id": "doc00001"})

        assert "Deleted" in result
        assert "warranty.pdf" in result
        assert "30 days" in result

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
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_CARD):
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
             patch(_P_DOC_SVC, return_value=mock_svc), \
             patch(_P_CARD):
            from services.innate_skills.document_skill import handle_document
            result = handle_document("topic", {"action": "restore", "name": "warranty"})

        assert "Restored" in result
        mock_svc.get_all_documents.assert_called_once_with(include_deleted=True)
