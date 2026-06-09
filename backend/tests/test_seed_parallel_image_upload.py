"""Feature test: turn-0 attachment uploads fan out in parallel at a barrier.

Drives the REAL ``MessageProcessor._seed_turn_zero`` — the SAME production method
that fires once before iteration 0 — with N real image attachments on a real
``UserConfig`` channel against the real test DB and real files.  ZERO mocks.

The parallelism contract: every attachment's ``document.upload`` (each of which
internally runs its own now-network-bound vision/OCR extraction) is dispatched on
a bounded thread pool whose ``with`` exit is the barrier, so all N MUST be fully
ingested before ``_seed_turn_zero`` returns.  The strongest real assertion — and
the one that catches a concurrency race (a lost upload, a duplicate id, a
half-written row) — is that after the seed EXACTLY N documents exist in the real
DB, each ``ready``, each with a distinct doc_id, and every original name present.
A race makes this FAIL; passing it is the barrier-safety proof, achieved with no
mocks (the writes funnel through thread-local connections + the single-threaded
document write queue; doc_id is a pre-generated random hex, not a rowid).

NO vision provider is configured -> the OCR fork, which is deterministic and
needs no network.  The fixtures are distinct named PNGs so each ingests as its
own document regardless of OCR readability.
"""

import io

import pytest

from configs.channels import UserConfig
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.message_processor import MessageProcessor
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path
from services.transcript_service import write_input_row

pytestmark = pytest.mark.unit


def _png_with_text(text: str) -> bytes:
    """A small high-contrast PNG rendering *text* (distinct per attachment)."""
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


def _write_attachment(label: str) -> str:
    """Write a distinct PNG under the Chalie temp prefix and return its path.

    ``_read_attachment`` enforces ``realpath`` startswith ``TMP_PATH_PREFIX``;
    ``new_tmp_path`` returns exactly such a path so the guard passes.
    """
    path = new_tmp_path(f"seed_parallel_{label}.png")
    with open(path, "wb") as fh:
        fh.write(_png_with_text(label.upper()))
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    """A real UserConfig MessageProcessor in the exact state ``_seed_turn_zero``
    fires from: input row written, ``active_tools`` seeded with the channel's
    always_available tier (message_processor._setup), attachments on metadata."""
    parent = object.__new__(MessageProcessor)
    MessageProcessor.__init__(parent, "Here are some files.", {"attachments": attachments})
    parent.config = UserConfig()
    parent.uid = write_input_row("user", "user", "Here are some files.")
    parent.active_tools = list(parent.config.always_available or [])
    return parent


def test_seed_uploads_all_attachments_in_parallel(db):
    """N image attachments -> _seed_turn_zero fans out the uploads on a pool and
    JOINS before returning -> EXACTLY N documents land, all ready, distinct ids,
    every name present.  A concurrency race (lost/duplicate/half-written upload)
    fails this; passing proves the barrier never drops or corrupts an upload."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    import os

    labels = ["alpha", "bravo", "charlie"]
    attachments = [_write_attachment(label) for label in labels]
    # _read_attachment uses os.path.basename, so the stored name is the file's
    # actual basename (which includes the Chalie temp prefix).
    names = [os.path.basename(p) for p in attachments]

    svc = DocumentService(get_shared_db_service())
    before = {d["id"] for d in svc.get_all_documents()}

    parent = _build_parent(attachments)
    # The REAL production method — the barrier (pool __exit__) joins all uploads.
    parent._seed_turn_zero()

    after = svc.get_all_documents()
    new_docs = [d for d in after if d["id"] not in before]

    # Exactly N landed — no upload lost, none duplicated.
    assert len(new_docs) == len(attachments), (
        f"expected {len(attachments)} new docs, got {len(new_docs)}: "
        f"{[(d['id'], d['original_name'], d['status']) for d in new_docs]}"
    )
    # Distinct doc_ids — the random-hex identity never collided under concurrency.
    assert len({d["id"] for d in new_docs}) == len(attachments)
    # Every original name is present — each distinct file ingested as itself.
    assert {d["original_name"] for d in new_docs} == set(names)
    # Each reached a terminal ready state and is independently retrievable.
    for doc in new_docs:
        assert doc["status"] == "ready", (doc["original_name"], doc["status"])
        fetched = svc.get_document(doc["id"])
        assert fetched is not None and fetched["id"] == doc["id"]


def test_seed_skips_unreadable_attachment_without_aborting_others(db):
    """A bad attachment path is logged and skipped WITHOUT aborting the rest:
    the good files still ingest.  Proves the per-task OSError guard does not take
    down the whole barrier."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    good = _write_attachment("delta")
    bad = "/nonexistent/not_under_tmp_prefix.png"  # fails the realpath guard

    svc = DocumentService(get_shared_db_service())
    before = {d["id"] for d in svc.get_all_documents()}

    parent = _build_parent([good, bad])
    parent._seed_turn_zero()

    import os

    new_docs = [d for d in svc.get_all_documents() if d["id"] not in before]
    assert len(new_docs) == 1, [
        (d["original_name"], d["status"]) for d in new_docs
    ]
    assert new_docs[0]["original_name"] == os.path.basename(good)
    assert new_docs[0]["status"] == "ready"
