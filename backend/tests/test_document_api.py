"""
Tests for backend/api/documents.py — Documents API blueprint.

Scope is deliberately narrow: only the pieces of this module that are
*deterministic and exercise real code* are unit-tested here —

  * input validation that runs before any service call (unsupported upload
    extension rejected with 400), and
  * the pure helper functions (`_sanitize_filename`, `_validate_file_path`).

The end-to-end behaviour of the document endpoints (upload → process →
list/get/content → confirm/augment → soft-delete, plus RAG search over
ingested documents) requires the real stack — DocumentService + a real
DataGraphService DB, the embedding pipeline, and background processing.
That behaviour is proven by the nightly scenarios, NOT by mocking the
DB-backed services here:

  * scenarios/060-document-lifecycle.yaml       — upload/confirm/augment/list/get/content/delete
  * scenarios/062-image-upload-and-context.yaml — image upload + context recall (search)
  * scenarios/063-pdf-upload-and-context.yaml   — pdf upload + context recall (search)

The previous mock-saturated endpoint tests (patching `_get_document_service`,
`_process_upload`, `get_data_graph_service`) asserted on patched return values
and mock call counts — they passed even when the feature was broken, so they
were removed in favour of the nightly scenarios above.
"""

import io

import pytest
from unittest.mock import patch
from flask import Flask

from api.documents import documents_bp
from services.file_mapper_service import FileMapperService


@pytest.mark.unit
class TestDocumentsAPI:
    """HTTP-layer validation that runs before any DB-backed service call."""

    @pytest.fixture
    def client(self):
        """Create a minimal Flask test client with only the documents blueprint."""
        app = Flask(__name__)
        app.register_blueprint(documents_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth so the request reaches the route's own validation.

        This only stubs the session check (a harness concession); no business
        logic / DB / LLM is faked.
        """
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    # ------------------------------------------------------------------
    # POST /documents/upload — extension allowlist
    # ------------------------------------------------------------------

    def test_upload_unsupported_extension(self, client):
        """Upload with a disallowed extension is rejected with 400 before any
        service call (real validation branch, no core-stack mocking)."""
        data = {"file": (io.BytesIO(b"malicious"), "virus.exe")}
        resp = client.post(
            "/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "not supported" in resp.get_json()["error"].lower()


@pytest.mark.unit
class TestHelpers:
    """Test pure helper functions in the documents API module."""

    def test_sanitize_filename_security_and_fallback(self):
        from api.documents import _sanitize_filename
        # Hardening is delegated to safe_filename (covered in test_filename_utils);
        # here we assert the document wrapper's fallback for empty results.
        assert '/' not in _sanitize_filename('../../etc/passwd')
        assert _sanitize_filename('..') == 'unnamed_document'
        assert _sanitize_filename('') == 'unnamed_document'

    def test_validate_file_path_rejects_traversal(self, tmp_path):
        from api.documents import _validate_file_path
        with patch.object(FileMapperService, "_DOCUMENTS_DIR", tmp_path):
            assert _validate_file_path(str(tmp_path / "doc" / "file.pdf")) is True
            assert _validate_file_path("/etc/passwd") is False
