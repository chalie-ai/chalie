"""Feature test: turn-0 attachment uploads fan out in parallel at a barrier.

Drives the REAL ``MessageProcessor._seed_turn_zero`` — the SAME production method
that fires once before iteration 0 — with N real image attachments on a real
``UserConfig`` channel against the real test DB and real files.  ZERO mocks.
"""

import io
from typing import cast

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from services.document_service import DocumentService
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path

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


def _write_attachment(label: str) -> str:
    """Paths must be under ``TMP_PATH_PREFIX`` so the guard passes."""
    path = new_tmp_path(f"seed_parallel_{label}.png")
    with open(path, "wb") as fh:
        fh.write(_png_with_text(label.upper()))
    return path


def _build_parent(attachments: "list[str]") -> MessageProcessor:
    """Build the REAL MessageProcessor via its actual public constructor — no
    private-field poking. The constructor is "constructed inert": it wires every
    coordinating service (``dispatch_service`` included) eagerly but performs no
    DB writes and never assigns ``self.uid`` outside ``begin()``'s transaction.
    ``_seed_turn_zero`` -> ``_seed_upload_attachment`` tolerates that: it uploads
    through the real, fully-wired ``dispatch_service`` regardless, and only skips
    the (here-irrelevant) input-row doc-link when ``self.uid`` is ``None``."""
    return MessageProcessor(
        UserConfig(), -1, "Here are some files.", {"attachments": attachments},
    )


def test_seed_uploads_all_attachments_in_parallel(db: object) -> None:
    """Proves the barrier joins all uploads before _seed_turn_zero returns."""
    ProviderDbService().set_vision_provider(None)

    import os

    labels = ["alpha", "bravo", "charlie"]
    attachments = [_write_attachment(label) for label in labels]
    # _read_attachment uses os.path.basename, so the stored name is the file's
    # actual basename (which includes the Chalie temp prefix).
    names = [os.path.basename(p) for p in attachments]

    svc = DocumentService()
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
        fetched = svc.get_document(cast(str, doc["id"]))
        assert fetched is not None and fetched["id"] == doc["id"]


def test_seed_skips_unreadable_attachment_without_aborting_others(db: object) -> None:
    """Proves the per-task OSError guard doesn't take down the whole barrier."""
    ProviderDbService().set_vision_provider(None)

    good = _write_attachment("delta")
    bad = "/nonexistent/not_under_tmp_prefix.png"  # fails the realpath guard

    svc = DocumentService()
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
