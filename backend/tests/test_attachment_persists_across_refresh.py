"""Feature test: chat attachments survive a page refresh (TKT-842).

Drives the REAL production chain end-to-end with ZERO mocks:

  1. ``MessageProcessor._seed_turn_zero`` — the same method that fires before
     iteration 0 — uploads the turn's attachments AND must now persist a
     ``transcript_docs(transcript_id, doc_id)`` link for each one.
  2. ``api.conversation.get_recent_history`` — the same function that rebuilds the
     chat on page refresh — must return those links as ``msg["attachments"]`` so
     the frontend can re-render the image/chip from ``/documents/<id>/preview``.

The bug being guarded: today the live preview is a browser-only ``blob:`` URL that
dies on refresh, and nothing reconnects the persisted ``documents`` row to the
user's turn.  This test fails loudly if the link is never written (step 1) OR if
the history rebuild drops it (step 2) — the regression lives *between* those steps.

NO vision provider is configured -> the deterministic OCR fork, no network.
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
from services.transcript_service import write_input_row
from api.conversation import get_recent_history

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    """A small high-contrast PNG (image/* attachment)."""
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
    """Write *data* under the Chalie temp prefix (passes ``_read_attachment``'s
    realpath guard) and return the path."""
    path = new_tmp_path(name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires from (mirrors ``message_processor._setup``)."""
    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, "Here is my receipt.", {"attachments": attachments})
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", "Here is my receipt.")
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
    """One image + one non-image attachment -> ``_seed_turn_zero`` persists a
    ``transcript_docs`` link per upload -> ``get_recent_history`` returns them as
    ``msg["attachments"]`` with the correct ``is_image`` flag and preview URL,
    exactly as the refresh path needs to re-render the bubble."""
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
    """A good image + an unreadable path: the bad attachment is skipped before any
    upload, so it must NOT create a ``transcript_docs`` link, while the good one
    still does.  Locks the "link only successful uploads" contract — the rebuilt
    turn carries exactly one attachment, never a dangling/broken reference."""
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
