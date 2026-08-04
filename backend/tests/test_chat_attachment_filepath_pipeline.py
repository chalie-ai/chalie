"""Feature tests: the chat-attachment filepath pipeline, end to end.

Drives the REAL two-stage production path — ``MessageProcessor._land_attachments``
(synchronous inside ``begin``: copy into the store + write the link row) then
``_seed_turn_zero`` (on the drive thread: index + dispatch by MIME) — against a
real test DB and real files, then follows the attachment through the real read
paths a client actually uses: the ``transcript_files`` link row, the turn
serializer's attachment projection, ``FileIndexService`` search, and the real
``/api/files/preview`` HTTP endpoint via ``authed_client``. ZERO mocks — the
``_DOCUMENTS_DIR`` / ``get_file_index_db_path`` patches are the same
conftest-blessed path redirection every docs-dir test uses (both must land
BEFORE any ingest runs).

The ordering between those two stages is itself load-bearing and has its own
test: see ``test_link_rows_exist_before_the_working_frame_is_broadcast``.
"""

from __future__ import annotations

import io
import sqlite3
import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from api.dto.attachment import Attachment
from services.turn_serializer_service import _fetch_attachments_for_transcripts
from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from models.tool_call import ToolCall
from services.file_index_service import FileIndexService
from services.file_mapper_service import FileMapperService
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path

if TYPE_CHECKING:
    from flask.testing import FlaskClient

    from contracts.json_serializable import JsonSerializable
    from services.memory_store import MemoryStore

    AuthedClient = tuple[FlaskClient, sqlite3.Connection, MemoryStore]

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: "ImageFont.FreeTypeFont | ImageFont.ImageFont | None" = None
    for candidate in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 72)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _redirect_docs_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic redirection every docs-dir test uses: both patches land BEFORE
    any ingest runs, so the real data/ dirs and data/file_index.sqlite are
    never touched."""
    docs_dir = tmp_path / "documents"
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(
        FileMapperService, "get_file_index_db_path", lambda *_: tmp_path / "file_index.sqlite"
    )
    return docs_dir


def _stage_attachment(name: str, content: bytes) -> str:
    """Stage exactly the way ``ThreadsEndpoint._stage_uploads`` does: a real
    tmp path under ``TMP_PATH_PREFIX`` with an 8-hex collision prefix on the
    basename, real bytes written to it."""
    tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{name}")
    with open(tmp_path, "wb") as fh:
        fh.write(content)
    return tmp_path


def _seed_attachment_turn(attachments: list[str]) -> tuple[MessageProcessor, int]:
    """Build the REAL MessageProcessor via its actual public constructor, then
    run the exact sequence ``begin()`` runs synchronously — turn allocation +
    anchoring input row inside one transaction, then ``_land_attachments``
    once it has committed — before driving the REAL ``_seed_turn_zero`` the
    way the drive thread does. Every method here is the production one, in
    production order. Stops short of spawning the drive thread / the ACT loop
    itself (no provider is configured in this test DB).
    Returns the transcript id as a plain ``int`` (narrowed from ``mp.uid``'s
    ``int | None`` type) since every caller needs it as a definite id."""
    mp = MessageProcessor(UserConfig(), -1, "Here's a file.", {"attachments": attachments})
    with mp.db.transaction():
        mp.turn_id = mp.transcript_service.allocate_turn()
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    assert mp.uid is not None
    uid = mp.uid
    mp._land_attachments()
    mp._seed_turn_zero()
    return mp, uid


def test_text_attachment_lands_in_uploads_indexed_linked_and_previewable(
    authed_client: "AuthedClient", tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full chain a chat client depends on for one text attachment: saved
    file -> transcript_files link -> file-index search hit -> the pill
    the turn serializer projects -> the pill's own preview URL serving the real bytes."""
    client, db_conn, _store = authed_client
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    content = b"The quarterly roadmap mentions a Valletta expansion."
    staged = _stage_attachment("brief.txt", content)
    _mp, uid = _seed_attachment_turn([staged])

    # File landed flat under documents/uploads/, staging prefix stripped.
    saved = docs_dir / "uploads" / "brief.txt"
    assert saved.is_file()
    assert saved.read_bytes() == content

    # transcript_files links this turn's input row to the saved relpath.
    rows = db_conn.execute(
        "SELECT transcript_id, path FROM transcript_files WHERE transcript_id = ?", (uid,)
    ).fetchall()
    assert [tuple(r) for r in rows] == [(uid, "uploads/brief.txt")]

    # FileIndexService finds it by its actual content, not just its filename.
    assert str(saved) in FileIndexService().search("Valletta")

    # the turn serializer projects it into the attachment pill the FE renders.
    by_id = _fetch_attachments_for_transcripts(db_conn, [uid])
    assert by_id[uid] == [{
        "filename": "brief.txt",
        "mime_type": "text/plain",
        "is_image": False,
        "url": "/api/files/preview/uploads/brief.txt",
    }]
    # The dict must satisfy the Attachment DTO — GET /api/threads/<id> validates it.
    Attachment.model_validate(by_id[uid][0])

    # The pill's own URL serves the real bytes through the real HTTP endpoint.
    resp = client.get(by_id[uid][0]["url"])
    assert resp.status_code == 200
    assert resp.data == content
    assert (resp.content_type or "").startswith("text/plain")


def test_deleted_attachment_file_is_omitted_from_the_pill_list(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file removed from disk after being linked must not render as a pill —
    matching the prior soft-delete semantics (a removed file renders nothing,
    not a broken link)."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    staged = _stage_attachment("note.txt", b"temporary note content")
    _mp, uid = _seed_attachment_turn([staged])

    saved = docs_dir / "uploads" / "note.txt"
    assert saved.is_file()
    saved.unlink()

    by_id = _fetch_attachments_for_transcripts(db, [uid])
    assert by_id == {}


def test_preview_endpoint_rejects_traversal_and_missing_and_idless_requests(
    authed_client: "AuthedClient", tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview endpoint's own hardening, independent of the ingest
    pipeline: a relpath that escapes the documents root 404s, a relpath that
    stays inside but names no file 404s, and a request with no id at all
    (the create-sentinel route) 404s rather than 500ing or leaking a listing."""
    client, _db_conn, _store = authed_client
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)
    (docs_dir / "uploads").mkdir(parents=True)
    (docs_dir / "uploads" / "real.txt").write_bytes(b"inside the root")
    # A real file OUTSIDE the documents root — the traversal escape target.
    (tmp_path / "secret.txt").write_bytes(b"should never be served")

    traversal = client.get("/api/files/preview/uploads/../../secret.txt")
    assert traversal.status_code == 404

    missing = client.get("/api/files/preview/uploads/does-not-exist.txt")
    assert missing.status_code == 404

    idless = client.get("/api/files/preview")
    assert idless.status_code == 404

    # Sanity: the same endpoint DOES serve a real, in-root file — the 404s
    # above are the guard firing, not a broken route.
    real = client.get("/api/files/preview/uploads/real.txt")
    assert real.status_code == 200
    assert real.data == b"inside the root"


def test_unparseable_attachment_is_still_stored_linked_and_dispatched(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty file fails FileParserService's extraction (ValueError) at index
    time — on the drive thread, long after the file was placed and linked. The
    turn must not crash, the pill must still render (file on disk AND link row
    present), and ``read`` is still dispatched so the failure surfaces loudly
    on the act trail instead of being silently dropped."""
    docs_dir = _redirect_docs_and_index(tmp_path, monkeypatch)

    staged = _stage_attachment("empty.txt", b"")
    mp, uid = _seed_attachment_turn([staged])  # must not raise

    saved = docs_dir / "uploads" / "empty.txt"
    assert saved.is_file()
    assert saved.read_bytes() == b""

    # Placement and linking do not depend on extraction succeeding.
    rows = db.execute(
        "SELECT path FROM transcript_files WHERE transcript_id = ?", (uid,)
    ).fetchall()
    assert [r[0] for r in rows] == ["uploads/empty.txt"]

    calls = ToolCall.by_turn(mp.channel, mp.turn_id)
    assert any(c.tool_name == "read" for c in calls), [c.tool_name for c in calls]


@pytest.mark.usefixtures("chat_provider")
def test_link_rows_exist_before_the_working_frame_is_broadcast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering guarantee the whole split exists for.

    ``working`` is the cue a client refetches the turn on, and turn-zero
    seeding is WS-silent — so a ``working`` frame emitted before the
    ``transcript_files`` rows land renders a permanently attachment-less user
    bubble, with nothing to signal the correction. This drives the REAL
    ``begin()`` (the only substitution is the LLM network boundary) and
    observes the live DB at the instant the frame is pushed.

    Fails on the pre-split ordering, where the rows were written on the drive
    thread after an ingest that takes ~9s for an image."""
    from services.database import Database
    from models.turn_execution import TurnExecution
    from models.provider_response import ProviderResponse
    from tests.test_message_processor_empty_completion import (
        _ScriptedProvider,
        _drain_background_turns,
    )

    _redirect_docs_and_index(tmp_path, monkeypatch)
    staged = _stage_attachment("agenda.txt", b"Board agenda: three items.")

    rows_at_working: list[list[str]] = []
    real_push = MessageProcessor.push_websocket

    def _observe(self: MessageProcessor, instance: "JsonSerializable | None") -> None:
        """Pass-through observer — records the committed link rows at the
        moment the working frame goes out, then delegates to the real emit."""
        if isinstance(instance, TurnExecution) and instance.state == TurnExecution.WORKING:
            rows_at_working.append([
                r[0] for r in Database.conn().execute(
                    "SELECT path FROM transcript_files WHERE transcript_id = ?", (self.uid,)
                ).fetchall()
            ])
        real_push(self, instance)

    monkeypatch.setattr(MessageProcessor, "push_websocket", _observe)

    mp = MessageProcessor(UserConfig(), -1, "Here's a file.", {"attachments": [staged]})
    provider = _ScriptedProvider(ProviderResponse(text="Noted.", model="scripted", tool_calls=None))
    with patch("services.provider_service.build_client", return_value=provider):
        mp.begin()
        mp.result()
        _drain_background_turns()

    assert rows_at_working == [["uploads/agenda.txt"]], rows_at_working


def test_image_attachment_dispatches_vision_not_read(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image attachment is described via ``vision``, never routed through
    the text-only ``read`` tool — verified on the real tool-call trail the
    dispatcher itself writes (models/tool_call.py), not a return-value check."""
    ProviderDbService().set_vision_provider(None)  # no provider configured -> OCR fallback rung
    _redirect_docs_and_index(tmp_path, monkeypatch)

    staged = _stage_attachment("photo.png", _png_with_text("HELLO"))
    mp, _uid = _seed_attachment_turn([staged])

    tool_names = [c.tool_name for c in ToolCall.by_turn(mp.channel, mp.turn_id)]
    assert "vision" in tool_names, tool_names
    assert "read" not in tool_names, tool_names
