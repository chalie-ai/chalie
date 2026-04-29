"""
Feature tests for _resolve_file_tags (backend/api/websocket.py).

_resolve_file_tags is the blocking chokepoint that converts attached image IDs
and recently-uploaded documents into structured [tag] strings injected into the
LLM prompt via metadata['file_tags'].  These tests verify the five real-world
paths a user's file attachment can take.

All tests use the real production stack:
- Real DocumentService backed by a real on-disk SQLite DB built from schema.sql.
- get_shared_db_service is replaced with the test DatabaseService for the
  duration of each test (the same injection conftest.py does, without requiring
  SchemaConvergenceService which needs sqlite-vec and Python 3.10+).
- Zero mocks of production code.  DocumentService._write_queue is replaced with
  an inline (same-thread) executor so SQLite thread-local connections work
  correctly in single-threaded tests — this is infrastructure wiring, not a
  mock of the feature being tested.

The 10-second poll-timeout path is intentionally omitted: it burns real
wall-clock time that is unacceptable in CI, and the scenario is already
exercised end-to-end by nightly scenario 063.
"""

import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load api/websocket.py directly to avoid api/__init__.py which uses
# Python 3.10+ union type syntax (Path | None) and fails on Python 3.9.
# The function under test (_resolve_file_tags) has no dependency on anything
# defined in api/__init__.py — it only imports from services at call time.
# ---------------------------------------------------------------------------

def _load_ws_module():
    backend_dir = Path(__file__).resolve().parent.parent
    ws_path = backend_dir / "api" / "websocket.py"
    mod_name = "api.websocket"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(ws_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_ws = _load_ws_module()
_resolve_file_tags = _ws._resolve_file_tags

# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


class _InlineWriteQueue:
    """Execute DocumentService write closures on the calling thread.

    DocumentService normally submits writes to a background thread that owns
    its own SQLite thread-local connection.  In tests, the seeding connection
    IS the thread-local connection, so we must execute writes inline to avoid
    opening a second connection on a different thread and losing visibility of
    unseeded data.  This is not a mock — it is infrastructure wiring that keeps
    the single-threaded test environment coherent.
    """

    def submit_sync(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def _bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Apply schema.sql to a fresh SQLite connection, skipping virtual tables.

    Virtual tables (FTS5, vec0) require the sqlite-vec extension which is not
    available in every dev environment.  _resolve_file_tags only queries the
    plain `documents` table, so skipping virtual tables is safe here.
    """
    schema_sql = _SCHEMA_PATH.read_text()

    # Split on semicolons while keeping each statement whole.
    # Strip inline comments first to avoid false semicolons inside comments.
    clean = re.sub(r"--[^\n]*", "", schema_sql)
    statements = [s.strip() for s in clean.split(";") if s.strip()]

    for stmt in statements:
        upper = stmt.upper()
        # Skip statements that create virtual tables (need extensions).
        if "CREATE VIRTUAL TABLE" in upper:
            continue
        # Skip INSERT OR IGNORE seed data (not needed for these tests).
        if upper.startswith("INSERT"):
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            # Ignore errors on optional objects (e.g. indexes on missing
            # virtual tables).
            pass

    conn.commit()


@pytest.fixture
def _ft_db(tmp_path):
    """Yield a (DatabaseService, raw_conn) pair backed by a real temp SQLite file.

    The DatabaseService singleton is replaced for the duration of the test so
    that _resolve_file_tags → get_shared_db_service() sees this DB.
    """
    import services.database_service as _db_mod
    from services.database_service import DatabaseService

    db_path = str(tmp_path / "test_ft.db")

    # Build schema on a plain sqlite3 connection (no WAL/pragma magic needed
    # for seeding; DatabaseService will re-open with its own pragmas).
    seed_conn = sqlite3.connect(db_path)
    seed_conn.row_factory = sqlite3.Row
    _bootstrap_schema(seed_conn)

    db_service = DatabaseService(db_path)

    # Clear thread-local so DatabaseService opens a fresh connection to our file.
    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    try:
        yield db_service, seed_conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        try:
            seed_conn.close()
        except Exception:
            pass


def _insert_doc(
    seed_conn: sqlite3.Connection,
    doc_id: str,
    *,
    status: str = "pending",
    source_type: str = "upload",
    original_name: str = "test.png",
    mime_type: str = "image/png",
    extracted_metadata: dict = None,
    clean_text: str = None,
    created_offset_seconds: int = 0,
) -> None:
    """Seed a documents row directly via the seeding connection.

    created_offset_seconds < 0  → row was created that many seconds AGO
    created_offset_seconds == 0 → row created 'now'
    """
    meta_json = json.dumps(extracted_metadata or {})
    created_expr = (
        f"datetime('now', '{created_offset_seconds} seconds')"
        if created_offset_seconds != 0
        else "datetime('now')"
    )
    seed_conn.execute(
        f"""
        INSERT INTO documents
            (id, original_name, mime_type, file_size_bytes, file_path,
             file_hash, source_type, status, extracted_metadata, clean_text,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {created_expr}, datetime('now'))
        """,
        (
            doc_id, original_name, mime_type, 1024,
            f"{doc_id}/{original_name}", "deadbeef",
            source_type, status, meta_json, clean_text,
        ),
    )
    seed_conn.commit()


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ready_image_emits_ocr_tag(_ft_db):
    """
    When an image_id refers to a document with status='ready' and OCR text,
    _resolve_file_tags must return a single [image id=... ocr=...] tag
    containing that OCR text.

    This is the happy path: the user drags an image into chat, the vision
    analysis completes before the user hits send, and the LLM receives the
    extracted text.
    """
    _, seed_conn = _ft_db
    _insert_doc(
        seed_conn,
        "img00001",
        source_type="chat_image",
        status="ready",
        extracted_metadata={"ocr_text": "INVOICE #1234", "has_text": True},
    )

    tags = _resolve_file_tags(["img00001"], "req-001")

    assert len(tags) == 1
    assert tags[0].startswith("[image id=img00001 ocr=")
    assert "INVOICE #1234" in tags[0]


@pytest.mark.unit
def test_failed_image_emits_status_failed_tag(_ft_db):
    """
    When an image_id refers to a document with status='failed' (vision
    analysis raised an error), _resolve_file_tags must return
    [image id=... status=failed] so the LLM knows the image was attached
    but could not be read — rather than silently dropping it.
    """
    _, seed_conn = _ft_db
    _insert_doc(
        seed_conn,
        "img00002",
        source_type="chat_image",
        status="failed",
    )

    tags = _resolve_file_tags(["img00002"], "req-002")

    assert len(tags) == 1
    assert tags[0] == "[image id=img00002 status=failed]"


@pytest.mark.unit
def test_nonexistent_image_emits_not_found_tag(_ft_db):
    """
    When an image_id has no corresponding document record (deleted, never
    uploaded, or the client sent a stale ID), _resolve_file_tags must return
    [image id=... status=not_found] rather than silently omitting the image
    from the LLM context.
    """
    # Deliberately do NOT seed any document for this ID.
    tags = _resolve_file_tags(["ghostid0"], "req-003")

    assert len(tags) == 1
    assert tags[0] == "[image id=ghostid0 status=not_found]"


@pytest.mark.unit
def test_empty_image_ids_picks_up_recent_upload_document(_ft_db):
    """
    When image_ids is empty but a source_type='upload' document was created
    within the last 120 seconds and is ready with clean_text, _resolve_file_tags
    must inject [document id=... name=... content=...] with the document's
    clean_text.

    This is the drag-and-drop PDF / text-file path: the user uploads a PDF
    alongside their message, and even though no explicit image_ids are sent,
    the file context is injected automatically.
    """
    _, seed_conn = _ft_db
    _insert_doc(
        seed_conn,
        "doc00001",
        source_type="upload",
        original_name="report.pdf",
        mime_type="application/pdf",
        status="ready",
        clean_text="Quarterly revenue grew 18% year-over-year.",
    )

    tags = _resolve_file_tags([], "req-004")

    assert len(tags) == 1
    tag = tags[0]
    assert tag.startswith("[document id=doc00001")
    assert "report.pdf" in tag
    assert "Quarterly revenue grew 18%" in tag


@pytest.mark.unit
def test_empty_image_ids_picks_up_recent_chat_image_document(_ft_db):
    """
    When image_ids is empty but a source_type='chat_image' document was
    created within the last 120 seconds and is ready, _resolve_file_tags
    must inject [image id=... ocr=...] — the same OCR tag format used for
    explicitly-referenced images.

    This is the paste / drop path where the frontend sends the chat turn
    before the upload XHR has had a chance to attach the image_id to the
    message payload.
    """
    _, seed_conn = _ft_db
    _insert_doc(
        seed_conn,
        "img00003",
        source_type="chat_image",
        status="ready",
        extracted_metadata={"ocr_text": "Meeting notes: action items", "has_text": True},
    )

    tags = _resolve_file_tags([], "req-005")

    assert len(tags) == 1
    assert tags[0].startswith("[image id=img00003 ocr=")
    assert "Meeting notes: action items" in tags[0]
