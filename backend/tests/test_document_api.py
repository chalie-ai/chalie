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
        "watched_folder_id": None,
        "created_at": now,
        "updated_at": now,
        "deleted_at": deleted_at,
        "purge_after": None,
    }




# Patch targets — module-level helpers use direct paths, local imports use source module
_P_SVC = "api.documents._get_document_service"
_P_ENQ = "api.documents._process_upload"
_P_EMB = "services.embedding_service.get_embedding_service"
_P_DGS = "services.data_graph_service.get_data_graph_service"


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

    # ------------------------------------------------------------------
    # GET /documents/<id>
    # ------------------------------------------------------------------

    def test_get_document(self, client):
        doc = _make_doc_dict()
        mock_rows = [("doc:doc00001:000", "Preview text one"), ("doc:doc00001:001", "Preview text two")]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows
        mock_dgs = MagicMock()
        mock_dgs.db.connection.return_value = mock_conn
        with patch(_P_SVC) as mock_get, \
             patch(_P_DGS, return_value=mock_dgs):
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.get("/documents/doc00001")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["item"]["id"] == "doc00001"
        assert len(data["item"]["artifacts"]) == 2
        assert data["item"]["artifacts"][0]["key"] == "doc:doc00001:000"

    # ------------------------------------------------------------------
    # GET /documents/<id>/content
    # ------------------------------------------------------------------

    def test_get_content_returns_artifacts(self, client):
        doc = _make_doc_dict()
        mock_rows = [("doc:doc00001:000", "Full text of artifact zero."),
                     ("doc:doc00001:001", "Full text of artifact one.")]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows
        mock_dgs = MagicMock()
        mock_dgs.db.connection.return_value = mock_conn
        with patch(_P_SVC) as mock_get, \
             patch(_P_DGS, return_value=mock_dgs):
            mock_svc = MagicMock()
            mock_get.return_value = mock_svc
            mock_svc.get_document.return_value = doc

            resp = client.get("/documents/doc00001/content")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_artifacts"] == 2
        assert data["artifacts"][0]["key"] == "doc:doc00001:000"
        assert data["artifacts"][0]["content"] == "Full text of artifact zero."

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

    # ------------------------------------------------------------------
    # GET /documents/search
    # ------------------------------------------------------------------

    def test_search_returns_results(self, client):
        # search_documents now calls DataGraphService.recall(), not the document service
        dg_results = [{
            "kind": "document",
            "key": "doc:d1:000",
            "value": "Coverage for 24 months.",
            "source": "document:d1",
        }]
        with patch("services.data_graph_service.get_data_graph_service") as mock_get_dgs:
            mock_dgs = MagicMock()
            mock_get_dgs.return_value = mock_dgs
            mock_dgs.recall.return_value = dg_results

            resp = client.get("/documents/search?q=warranty+coverage")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["query"] == "warranty coverage"
        assert len(data["results"]) == 1
        assert data["results"][0]["document_id"] == "d1"
        assert data["results"][0]["content"] == "Coverage for 24 months."


    # ------------------------------------------------------------------
    # POST /documents/upload
    # ------------------------------------------------------------------

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


@pytest.mark.unit
class TestHelpers:
    """Test helper functions in the documents API module."""

    def test_sanitize_filename_security(self):
        from api.documents import _sanitize_filename
        assert '/' not in _sanitize_filename('../../etc/passwd')
        assert '\\' not in _sanitize_filename('..\\windows\\system32')
        assert '\x00' not in _sanitize_filename('file\x00.txt')
        assert not _sanitize_filename('...hidden').startswith('.')

    def test_validate_file_path_rejects_traversal(self, tmp_path):
        from api.documents import _validate_file_path
        with patch("api.documents.DOCUMENTS_ROOT", str(tmp_path)):
            assert _validate_file_path(str(tmp_path / "doc" / "file.pdf")) is True
            assert _validate_file_path("/etc/passwd") is False


