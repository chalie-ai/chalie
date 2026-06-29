"""
Tests for the Documents API namespace.

Covers only deterministic, real-code paths: extension allowlist validation and
the pure helpers (_sanitize_filename, _validate_file_path). The full document
lifecycle (upload/confirm/augment/list/get/content/delete, RAG search) is
exercised by end-to-end scenarios, not by mocks here.
"""

import hashlib
import io
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from unittest.mock import patch

from api.documents import documents_ns
from tests.restx_test_app import mount_namespace
from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from flask.testing import FlaskClient


@pytest.mark.unit
class TestDocumentsAPI:
    """HTTP-layer validation that runs before any DB-backed service call."""

    @pytest.fixture
    def client(self) -> "FlaskClient":
        app = mount_namespace(documents_ns)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self) -> Generator[None, None, None]:
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    # ------------------------------------------------------------------
    # POST /documents/upload — extension allowlist
    # ------------------------------------------------------------------

    def test_upload_unsupported_extension(self, client: "FlaskClient") -> None:
        """Disallowed extension is rejected with 400 before any service call."""
        from typing import cast
        data = {"file": (io.BytesIO(b"malicious"), "virus.exe")}
        resp = client.post(
            "/documents/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "not supported" in cast(str, cast("dict[str, object]", resp.get_json())["error"]).lower()


@pytest.mark.unit
class TestDocumentUploadRealStack:
    """Upload happy path: asserts HTTP response, DB row, and disk bytes with zero mocking."""

    def test_upload_text_document_persists_ready_row_and_file(self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]", tmp_path: Path) -> None:
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
class TestDocumentSearchRealStack:
    """GET /documents/search query-param handling over the real DB + data_graph."""

    def test_search_nonnumeric_limit_rejected_as_validation_error(
        self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]"
    ) -> None:
        """A non-numeric ?limit is a validation error (422), not a 500.

        The DTO boundary enforces ``limit`` as an int (≤20); a malformed value is
        rejected up front by the ``@expects`` decorator's pydantic validation,
        so the request never reaches the data_graph recall path.
        """
        client, _db_conn, _store = authed_client

        good = client.get("/documents/search?q=quarterly+revenue&limit=5")
        bad = client.get("/documents/search?q=quarterly+revenue&limit=abc")

        assert good.status_code == 200, good.get_data(as_text=True)
        assert bad.status_code == 422, bad.get_data(as_text=True)

    def test_search_missing_query_rejected(self, authed_client: "tuple[FlaskClient, sqlite3.Connection, object]") -> None:
        """A missing ``q`` is rejected with 422 by the DTO boundary."""
        client, _db_conn, _store = authed_client

        resp = client.get("/documents/search")

        assert resp.status_code == 422, resp.get_data(as_text=True)


@pytest.mark.unit
class TestHelpers:

    def test_sanitize_filename_security_and_fallback(self) -> None:
        from api.documents import _sanitize_filename
        # Hardening is delegated to safe_filename (covered in test_filename_utils);
        # here we assert the document wrapper's fallback for empty results.
        assert '/' not in _sanitize_filename('../../etc/passwd')
        assert _sanitize_filename('..') == 'unnamed_document'
        assert _sanitize_filename('') == 'unnamed_document'

    def test_validate_file_path_rejects_traversal(self, tmp_path: Path) -> None:
        from api.documents import _validate_file_path
        with patch.object(FileMapperService, "_DOCUMENTS_DIR", tmp_path):
            assert _validate_file_path(str(tmp_path / "doc" / "file.pdf")) is True
            assert _validate_file_path("/etc/passwd") is False
