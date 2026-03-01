"""
Tests for backend/api/documents.py — Documents API blueprint.

All tests mock _get_document_service() to isolate the HTTP layer from the
database-backed DocumentService. No external dependencies required.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from flask import Flask

from api.documents import documents_bp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_dict(
    doc_id="doc00001",
    original_name="warranty.pdf",
    status="ready",
    page_count=3,
    chunk_count=12,
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
        "file_hash": "sha256hash",
        "page_count": page_count,
        "status": status,
        "error_message": None,
        "chunk_count": chunk_count,
        "source_type": "upload",
        "tags": [],
        "summary": "Warranty document summary.",
        "extracted_metadata": {"document_type": {"value": "warranty", "confidence": 0.85}},
        "supersedes_id": None,
        "clean_text": "warranty coverage text",
        "language": "en",
        "fingerprint": "aabb",
        "created_at": now,
        "updated_at": now,
        "deleted_at": deleted_at,
        "purge_after": None,
    }


def _make_chunk_dict(chunk_index=0, content="Chunk content here."):
    return {
        "chunk_index": chunk_index,
        "content": content,
        "page_number": 1,
        "section_title": "Intro",
        "token_count": 10,
    }


# Patch targets — module-level helpers use direct paths, local imports use source module
_P_SVC = "api.documents._get_document_service"
_P_ENQ = "api.documents._enqueue_processing"
_P_EMB = "services.embedding_service.get_embedding_service"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDocumentsAPI:
    """Test all endpoints on the documents blueprint."""

    @pytest.fixture
    def client(self):
        """Create a minimal Flask test client with only the documents blueprint."""
        app = Flask(__name__)
        app.register_blueprint(documents_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for every test in this class."""
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /documents
    # ------------------------------------------------------------------

    def test_list_documents(self, client):
        docs = [_make_doc_dict(doc_id="d1"), _make_doc_dict(doc_id="d2", original_name="invoice.pdf")]
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_all_documents.return_value = docs

            resp = client.get("/documents")

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["items"]) == 2
        # Datetimes serialized to ISO strings
        assert isinstance(data["items"][0]["created_at"], str)
        # clean_text stripped from response
        assert "clean_text" not in data["items"][0]

    def test_list_documents_empty(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_all_documents.return_value = []

            resp = client.get("/documents")

        assert resp.status_code == 200
        assert resp.get_json()["items"] == []

    def test_list_documents_include_deleted(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_all_documents.return_value = []

            client.get("/documents?include_deleted=true")

        mock_svc.get_all_documents.assert_called_once_with(include_deleted=True)

    # ------------------------------------------------------------------
    # GET /documents/<id>
    # ------------------------------------------------------------------

    def test_get_document(self, client):
        doc = _make_doc_dict()
        chunks = [_make_chunk_dict(0), _make_chunk_dict(1)]
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc
            mock_svc.get_chunks_for_document.return_value = chunks

            resp = client.get("/documents/doc00001")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["item"]["id"] == "doc00001"
        assert len(data["item"]["chunks"]) == 2

    def test_get_document_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = None

            resp = client.get("/documents/missing")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # GET /documents/<id>/content
    # ------------------------------------------------------------------

    def test_get_content_paginates(self, client):
        doc = _make_doc_dict()
        chunks = [_make_chunk_dict(i, f"Chunk {i}") for i in range(10)]
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc
            mock_svc.get_chunks_for_document.return_value = chunks

            resp = client.get("/documents/doc00001/content?page=1&per_page=3")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_chunks"] == 10
        assert data["page"] == 1
        assert len(data["chunks"]) == 3

    def test_get_content_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = None

            resp = client.get("/documents/missing/content")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # DELETE /documents/<id>
    # ------------------------------------------------------------------

    def test_soft_delete(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.soft_delete.return_value = True

            resp = client.delete("/documents/doc00001")

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_soft_delete_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.soft_delete.return_value = False

            resp = client.delete("/documents/doc00001")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # POST /documents/<id>/restore
    # ------------------------------------------------------------------

    def test_restore(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.restore.return_value = True

            resp = client.post("/documents/doc00001/restore")

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_restore_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.restore.return_value = False

            resp = client.post("/documents/doc00001/restore")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # DELETE /documents/<id>/purge
    # ------------------------------------------------------------------

    def test_purge(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.hard_delete.return_value = True

            resp = client.delete("/documents/doc00001/purge")

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_purge_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.hard_delete.return_value = False

            resp = client.delete("/documents/missing/purge")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # POST /documents/<id>/confirm
    # ------------------------------------------------------------------

    def test_confirm_document(self, client):
        doc = _make_doc_dict(status="awaiting_confirmation")
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.post("/documents/doc00001/confirm")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["status"] == "ready"
        mock_svc.update_status.assert_called_once_with("doc00001", "ready", chunk_count=12)

    def test_confirm_wrong_status(self, client):
        doc = _make_doc_dict(status="ready")
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.post("/documents/doc00001/confirm")

        assert resp.status_code == 400
        assert "not awaiting" in resp.get_json()["error"].lower()

    def test_confirm_not_found(self, client):
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = None

            resp = client.post("/documents/missing/confirm")

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # POST /documents/<id>/augment
    # ------------------------------------------------------------------

    def test_augment_document(self, client):
        doc = _make_doc_dict(status="awaiting_confirmation")
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.post(
                "/documents/doc00001/augment",
                json={"context": "This is my Samsung TV warranty"},
            )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["status"] == "ready"
        # Verify metadata was updated with user context
        call_args = mock_svc.update_extracted_metadata.call_args
        updated_meta = call_args.kwargs.get("metadata") or call_args[1].get("metadata") or call_args[0][1]
        assert updated_meta["_user_context"] == "This is my Samsung TV warranty"

    def test_augment_empty_context(self, client):
        doc = _make_doc_dict(status="awaiting_confirmation")
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.post(
                "/documents/doc00001/augment",
                json={"context": ""},
            )

        assert resp.status_code == 400
        assert "context" in resp.get_json()["error"].lower()

    def test_augment_wrong_status(self, client):
        doc = _make_doc_dict(status="ready")
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.post(
                "/documents/doc00001/augment",
                json={"context": "extra info"},
            )

        assert resp.status_code == 400

    # ------------------------------------------------------------------
    # GET /documents/search
    # ------------------------------------------------------------------

    def test_search_returns_results(self, client):
        results = [{
            "document_id": "d1",
            "document_name": "warranty.pdf",
            "content": "Coverage for 24 months.",
            "page_number": 1,
            "section_title": "Coverage",
            "score": 0.9,
            "document_created_at": datetime(2026, 2, 26, tzinfo=timezone.utc),
        }]
        with patch(_P_SVC) as mock_get:
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.search_chunks.return_value = results

            with patch(_P_EMB) as mock_emb:
                mock_emb_svc = MagicMock()
                mock_emb_svc.generate_embedding.return_value = [0.1] * 768
                mock_emb.return_value = mock_emb_svc

                resp = client.get("/documents/search?q=warranty+coverage")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["query"] == "warranty coverage"
        assert len(data["results"]) == 1
        # Datetime serialized
        assert isinstance(data["results"][0]["document_created_at"], str)

    def test_search_missing_query(self, client):
        resp = client.get("/documents/search")
        assert resp.status_code == 400
        assert "required" in resp.get_json()["error"].lower()

    def test_search_empty_query(self, client):
        resp = client.get("/documents/search?q=")
        assert resp.status_code == 400

    # ------------------------------------------------------------------
    # POST /documents/upload
    # ------------------------------------------------------------------

    def test_upload_no_file(self, client):
        """Upload without a file field returns 400."""
        resp = client.post("/documents/upload", data={})
        assert resp.status_code == 400
        assert "No file" in resp.get_json()["error"]

    def test_upload_unsupported_extension(self, client):
        """Upload with a disallowed extension returns 400."""
        import io
        data = {"file": (io.BytesIO(b"malicious"), "virus.exe")}
        resp = client.post(
            "/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "not supported" in resp.get_json()["error"].lower()

    def test_upload_empty_file(self, client):
        """Upload with an empty file returns 400."""
        import io
        data = {"file": (io.BytesIO(b""), "empty.txt")}
        resp = client.post(
            "/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"].lower()

    def test_upload_success(self, client, tmp_path):
        """Successful upload creates record, saves file, enqueues processing."""
        import io

        with patch("api.documents.DOCUMENTS_ROOT", str(tmp_path)):
            with patch(_P_SVC) as mock_get:
                mock_svc = MagicMock()
                mock_get.return_value = mock_svc
                mock_svc.create_document.return_value = "abcd1234"
                mock_svc.find_duplicates.return_value = []

                with patch(_P_ENQ) as mock_enq:
                    data = {"file": (io.BytesIO(b"Hello PDF content"), "test.pdf")}
                    resp = client.post(
                        "/documents/upload",
                        data=data,
                        content_type="multipart/form-data",
                    )

        assert resp.status_code == 201
        body = resp.get_json()
        assert body["id"] == "abcd1234"
        assert body["original_name"] == "test.pdf"
        assert body["status"] == "pending"
        mock_enq.assert_called_once()

    def test_upload_with_duplicates(self, client, tmp_path):
        """Upload returns duplicate info when hash matches exist."""
        import io
        now = datetime(2026, 2, 26, tzinfo=timezone.utc)

        with patch("api.documents.DOCUMENTS_ROOT", str(tmp_path)):
            with patch(_P_SVC) as mock_get:
                mock_svc = MagicMock()
                mock_get.return_value = mock_svc
                mock_svc.create_document.return_value = "new00001"
                mock_svc.find_duplicates.return_value = [
                    {"id": "old00001", "original_name": "warranty_v1.pdf",
                     "match_type": "exact", "created_at": now},
                ]

                with patch(_P_ENQ):
                    data = {"file": (io.BytesIO(b"duplicate content"), "warranty.pdf")}
                    resp = client.post(
                        "/documents/upload",
                        data=data,
                        content_type="multipart/form-data",
                    )

        assert resp.status_code == 201
        body = resp.get_json()
        assert len(body["duplicates"]) == 1
        assert body["duplicates"][0]["match_type"] == "exact"


@pytest.mark.unit
class TestHelpers:
    """Test helper functions in the documents API module."""

    def test_sanitize_filename_strips_traversal(self):
        from api.documents import _sanitize_filename
        assert '/' not in _sanitize_filename('../../etc/passwd')
        assert '\\' not in _sanitize_filename('..\\windows\\system32')

    def test_sanitize_filename_strips_null_bytes(self):
        from api.documents import _sanitize_filename
        result = _sanitize_filename('file\x00.txt')
        assert '\x00' not in result

    def test_sanitize_filename_strips_leading_dots(self):
        from api.documents import _sanitize_filename
        result = _sanitize_filename('...hidden')
        assert not result.startswith('.')

    def test_sanitize_filename_empty_becomes_unnamed(self):
        from api.documents import _sanitize_filename
        assert _sanitize_filename('') == 'unnamed_document'

    def test_sanitize_filename_limits_length(self):
        from api.documents import _sanitize_filename
        long_name = 'a' * 300 + '.pdf'
        result = _sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith('.pdf')

    def test_validate_file_path_rejects_traversal(self, tmp_path):
        from api.documents import _validate_file_path
        with patch("api.documents.DOCUMENTS_ROOT", str(tmp_path)):
            assert _validate_file_path(str(tmp_path / "doc" / "file.pdf")) is True
            assert _validate_file_path("/etc/passwd") is False

    def test_serialize_doc_removes_clean_text(self):
        from api.documents import _serialize_doc
        doc = _make_doc_dict()
        result = _serialize_doc(doc)
        assert "clean_text" not in result

    def test_serialize_doc_converts_datetimes(self):
        from api.documents import _serialize_doc
        doc = _make_doc_dict()
        result = _serialize_doc(doc)
        assert isinstance(result["created_at"], str)
        assert isinstance(result["updated_at"], str)
