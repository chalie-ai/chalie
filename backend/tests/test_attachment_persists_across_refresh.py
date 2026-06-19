"""Feature test: chat attachments survive a page refresh.

Drives the REAL production chain end-to-end with ZERO mocks: ``_seed_turn_zero``
must persist a ``transcript_docs(transcript_id, doc_id)`` link per upload, and
``get_recent_history`` must return those links as ``msg["attachments"]`` so the
frontend re-renders from ``/documents/<id>/preview``.

Guard bug: today preview is a browser-only ``blob:`` URL that dies on refresh.
NO vision provider -> deterministic OCR fork, no network.
"""

import io
import os

import pytest

from configs.channels import UserConfig
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.message_processor import MessageProcessor
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path
from services.transcript_service import turn_id_of_row, write_input_row
from api.conversation import get_recent_history

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font = None
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


def _write_tmp(name: str, data: bytes) -> str:
    path = new_tmp_path(name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, "Here is my receipt.", {"attachments": attachments})
    parent.config = UserConfig()
    # Mirror production _setup() exactly: write_input_row opens the next turn for
    # the channel atomically inside the INSERT; read the allocated turn_id back by
    # row id. The turn's seeded tool calls and its _cleanup_cancelled delete both
    # key off the same (channel, turn_id) boundary.
    parent.uid = write_input_row("user", "user", "Here is my receipt.")
    parent.turn_id = turn_id_of_row(parent.uid)
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def _linked_doc_ids(transcript_id: int) -> set:
    with get_shared_db_service().connection() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM transcript_docs WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchall()
    return {r[0] for r in rows}


def test_attachments_are_linked_and_served_on_refresh(db):
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    img_path = _write_tmp("refresh_receipt.png", _png_with_text("INVOICE"))
    txt_path = _write_tmp("refresh_notes.txt", b"plain text attachment body")
    img_name = os.path.basename(img_path)
    txt_name = os.path.basename(txt_path)

    svc = DocumentService(get_shared_db_service())
    before = {d["id"] for d in svc.get_all_documents()}

    parent = _build_parent([img_path, txt_path])
    # REAL production seed — uploads + (new) link writes happen here.
    parent._seed_turn_zero()

    new_docs = [d for d in svc.get_all_documents() if d["id"] not in before]
    assert len(new_docs) == 2, [(d["original_name"], d["status"]) for d in new_docs]
    by_name = {d["original_name"]: d for d in new_docs}
    img_doc, txt_doc = by_name[img_name], by_name[txt_name]

    # Step 1 assertion: BOTH uploads persisted a link to THIS user turn.
    assert _linked_doc_ids(parent.uid) == {img_doc["id"], txt_doc["id"]}

    # Step 2 assertion: the refresh rebuild serves those links on the user row.
    messages, _ = get_recent_history(limit=120, offset=0)
    mine = next((m for m in messages if m["id"] == str(parent.uid)), None)
    assert mine is not None, "user turn missing from rebuilt history"
    attachments = mine.get("attachments")
    assert attachments, "rebuilt user turn carries no attachments"

    att_by_id = {a["doc_id"]: a for a in attachments}
    assert set(att_by_id) == {img_doc["id"], txt_doc["id"]}

    img_att = att_by_id[img_doc["id"]]
    assert img_att["is_image"] is True
    assert img_att["filename"] == img_name
    assert img_att["url"] == f"/documents/{img_doc['id']}/preview"

    txt_att = att_by_id[txt_doc["id"]]
    assert txt_att["is_image"] is False
    assert txt_att["url"] == f"/documents/{txt_doc['id']}/preview"


def test_skipped_upload_writes_no_link(db):
    """Guards the 'link only successful uploads' contract — a skipped attachment
    must leave no orphan reference."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    good = _write_tmp("refresh_only_good.png", _png_with_text("OK"))
    bad = "/nonexistent/not_under_tmp_prefix.png"  # fails _read_attachment realpath guard
    good_name = os.path.basename(good)

    svc = DocumentService(get_shared_db_service())
    before = {d["id"] for d in svc.get_all_documents()}

    parent = _build_parent([good, bad])
    parent._seed_turn_zero()

    new_docs = [d for d in svc.get_all_documents() if d["id"] not in before]
    assert len(new_docs) == 1 and new_docs[0]["original_name"] == good_name
    good_id = new_docs[0]["id"]

    # Exactly one link — the skipped upload left no orphan reference.
    assert _linked_doc_ids(parent.uid) == {good_id}

    messages, _ = get_recent_history(limit=120, offset=0)
    mine = next((m for m in messages if m["id"] == str(parent.uid)), None)
    assert mine is not None
    assert [a["doc_id"] for a in mine.get("attachments", [])] == [good_id]


def test_cancel_after_attachment_deletes_turn_and_cascades_link(db):
    """Cancelling a turn that attached a file must delete the transcript row even
    though a ``transcript_docs`` link references it (FK CASCADE).  Guards a data
    integrity regression: with foreign_keys=ON and no cascade, the real
    ``_cleanup_cancelled`` DELETE FROM transcript would raise FOREIGN KEY
    constraint, get swallowed, and the cancelled turn would leak into refresh.
    The document row itself must survive — only the link cascades."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    img = _write_tmp("refresh_cancel.png", _png_with_text("BYE"))
    svc = DocumentService(get_shared_db_service())
    before = {d["id"] for d in svc.get_all_documents()}

    parent = _build_parent([img])
    parent._seed_turn_zero()
    uid = parent.uid
    new_docs = [d for d in svc.get_all_documents() if d["id"] not in before]
    assert len(new_docs) == 1
    doc_id = new_docs[0]["id"]
    assert _linked_doc_ids(uid) == {doc_id}  # link exists pre-cancel

    # REAL production cleanup — must not be blocked by the link.
    parent._cleanup_cancelled()

    with get_shared_db_service().connection() as conn:
        turn = conn.execute("SELECT 1 FROM transcript WHERE id = ?", (uid,)).fetchone()
    assert turn is None, "cancelled transcript row not deleted (FK blocked the delete?)"
    assert _linked_doc_ids(uid) == set(), "link not cascaded on transcript delete"
    # The document is independent of the turn — it must NOT be cascaded away.
    assert svc.get_document(doc_id) is not None
