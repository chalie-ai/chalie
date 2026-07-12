"""
Real-stack tests for the /api/documents endpoint + actions.

Every test drives the fully-assembled app (``create_app`` via the ``authed_client``
fixture) — no namespace mounting, no mocks on the service layer. Coverage:
the upload extension allowlist (400 before any disk write), the upload happy
path (201 + persisted DB row + bytes on disk), and the search query-param DTO
boundary (200 / 422). The broader lifecycle (confirm/augment/list/get/content/
delete) is exercised by end-to-end scenarios.

Filename hardening and document-path traversal defence are the responsibility of
``services.filename_utils.safe_filename`` and
``FileMapperService.validate_document_path`` respectively, and are covered by
``test_filename_utils.py`` and ``test_browser_screenshot_ingest.py``.
"""

import hashlib
import io
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from flask.testing import FlaskClient


@pytest.mark.unit
class TestDocumentUploadRealStack:
    """Upload path over the real app: extension gate, then 201 + DB row + disk bytes."""

    def test_upload_unsupported_extension(
        self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]"
    ) -> None:
        """A disallowed extension is rejected with 400 before any service call."""
        client, _db_conn, _store = authed_client

        resp = client.post(
            "/api/documents/upload",
            data={"file": (io.BytesIO(b"malicious"), "virus.exe")},
            content_type="multipart/form-data",
        )

        assert resp.status_code == 400, resp.get_data(as_text=True)
        payload = resp.get_json()
        assert payload["success"] is False
        assert "not supported" in str(payload["error"]).lower()

    def test_upload_text_document_persists_ready_row_and_file(
        self,
        authed_client: "tuple[FlaskClient, sqlite3.Connection, object]",
        tmp_path: Path,
    ) -> None:
        client, db_conn, _store = authed_client

        body = b"Quarterly revenue summary 2026. Invoice total 4200 EUR."

        # Redirect the document store to an isolated tmp root so the disk
        # assertion is deterministic and the real store is never polluted.
        # (The staging temp path the route writes to first is independent of
        # _DOCUMENTS_DIR.)
        with patch.object(FileMapperService, "_DOCUMENTS_DIR", tmp_path):
            resp = client.post(
                "/api/documents/upload",
                data={"file": (io.BytesIO(body), "note.txt")},
                content_type="multipart/form-data",
            )

            # Route returns 201 with the TERMINAL status — ingest is synchronous,
            # so the response carries 'ready', never 'pending'.
            assert resp.status_code == 201, resp.get_data(as_text=True)
            payload = resp.get_json()
            assert payload["success"] is True
            result = payload["result"]
            assert result["status"] == "ready", f"expected ready, got {result!r}"
            assert result["file_hash"] == hashlib.sha256(body).hexdigest()
            assert result["file_size"] == len(body)
            assert result["original_name"] == "note.txt"
            doc_id = result["id"]

            # The real documents row landed via DocumentService.create_document.
            row = db_conn.execute(
                "SELECT status, file_hash, file_path, original_name, source_type "
                "FROM documents WHERE id=?",
                (doc_id,),
            ).fetchone()
            assert row is not None, "no documents row was persisted"
            assert row[0] == "ready"
            assert row[1] == result["file_hash"]
            assert row[2] == f"{doc_id}/note.txt"
            assert row[3] == "note.txt"
            assert row[4] == "upload"

            # The bytes were really copied into the document store on disk.
            disk_path = FileMapperService.get_documents_path(doc_id, "note.txt")
            assert disk_path.exists(), f"file not written to store at {disk_path}"
            assert disk_path.read_bytes() == body


@pytest.mark.unit
class TestDocumentSearchRealStack:
    """GET /api/documents/search query-param handling over the real DB + data_graph."""

    def test_search_nonnumeric_limit_rejected_as_validation_error(
        self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]"
    ) -> None:
        """A non-numeric ?limit is a validation error (422), not a 500.

        The ``SearchQuery`` DTO enforces ``limit`` as an int (≤20); a malformed
        value is rejected up front when the action validates ``request.args``, so
        the request never reaches the data_graph recall path.
        """
        client, _db_conn, _store = authed_client

        good = client.get("/api/documents/search?q=quarterly+revenue&limit=5")
        bad = client.get("/api/documents/search?q=quarterly+revenue&limit=abc")

        assert good.status_code == 200, good.get_data(as_text=True)
        assert bad.status_code == 422, bad.get_data(as_text=True)

    def test_search_missing_query_rejected(
        self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]"
    ) -> None:
        """A missing ``q`` is rejected with 422 by the ``SearchQuery`` DTO boundary."""
        client, _db_conn, _store = authed_client

        resp = client.get("/api/documents/search")

        assert resp.status_code == 422, resp.get_data(as_text=True)
