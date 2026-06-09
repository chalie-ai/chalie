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
That behaviour is proven by the end-to-end scenario suite, NOT by mocking the
DB-backed services here:

  * the document lifecycle — upload/confirm/augment/list/get/content/delete
  * image upload + context recall (search)
  * pdf upload + context recall (search)

The previous mock-saturated endpoint tests (patching `_get_document_service`,
`_process_upload`, `get_data_graph_service`) asserted on patched return values
and mock call counts — they passed even when the feature was broken, so they
were removed in favour of the end-to-end scenarios above.

`TestDocumentUploadRealStack` closes the gap that deletion
left in the pre-merge `pytest -m unit` gate: the upload *happy path* was only
exercised by the end-to-end scenarios, never by a deterministic in-repo test. It
drives the real POST /documents/upload route through the `authed_client`
fixture (real Flask app + real SQLite + real MemoryStore, auth bypassed) and
asserts the real downstream effects — the synchronous ingest persisting a
`ready` documents row and copying the bytes into the document store on disk.
"""

import hashlib
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
class TestDocumentUploadRealStack:
    """Real-stack feature test for the upload happy path.

    Drives the real POST /documents/upload route end-to-end via `authed_client`
    (real Flask app, real SQLite, real MemoryStore — only the session check and
    secret are bypassed by the fixture's harness concession) and asserts every
    downstream effect of the synchronous ingest: the HTTP response,
    the persisted `documents` row, and the bytes copied into the document store
    on disk. Zero business-logic / service mocking.
    """

    def test_upload_text_document_persists_ready_row_and_file(self, authed_client, tmp_path):
        client, db_conn, _store = authed_client

        body = b"Quarterly revenue summary 2026. Invoice total 4200 EUR."

        # Redirect the document store to an isolated tmp root so the disk
        # assertion is deterministic and the real store is never polluted.
        # (Same seam the traversal test below uses; the staging temp path the
        # route writes to first is independent of _DOCUMENTS_DIR.)
        with patch.object(FileMapperService, "_DOCUMENTS_DIR", tmp_path):
            resp = client.post(
                "/documents/upload",
                data={"file": (io.BytesIO(body), "note.txt")},
                content_type="multipart/form-data",
            )

            # Route returns 201 with the TERMINAL status — ingest is synchronous,
            # so the response carries 'ready', never 'pending'.
            assert resp.status_code == 201, resp.get_data(as_text=True)
            payload = resp.get_json()
            assert payload["status"] == "ready", f"expected ready, got {payload!r}"
            assert payload["file_hash"] == hashlib.sha256(body).hexdigest()
            assert payload["file_size"] == len(body)
            assert payload["original_name"] == "note.txt"
            doc_id = payload["id"]

            # The real documents row landed via DocumentService.create_document.
            row = db_conn.execute(
                "SELECT status, file_hash, file_path, original_name, source_type "
                "FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            assert row is not None, "no documents row was persisted"
            assert row[0] == "ready"
            assert row[1] == payload["file_hash"]
            assert row[2] == f"{doc_id}/note.txt"
            assert row[3] == "note.txt"
            assert row[4] == "upload"

            # The bytes were really copied into the document store on disk.
            disk_path = FileMapperService.get_documents_path(doc_id, "note.txt")
            assert disk_path.exists(), f"file not written to store at {disk_path}"
            assert disk_path.read_bytes() == body


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
