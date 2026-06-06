"""Feature test: an uploaded image is findable by its visual content (TKT-838).

END-TO-END SEARCHABILITY PROOF — the heart of the Vision Subagent addendum.

Drives the REAL ``DocumentAbility.run`` upload + search entry points against the
real test DB and real files. ZERO mocks: the OCR description produced by the
no-vision-provider fork flows through the SAME production pipeline
(``create_document_artifacts`` -> data_graph embed + FTS5) that every upload
uses, and ``document(action='search')`` recalls it via the REAL recall path
(FTS5 + vector). Proves the doc_id guardrail is truthful: an image with words
becomes a searchable document.

Second test proves the ``_run_upload_extraction`` image-aware branch: a textless
image (no vision provider -> empty OCR) is persisted ``ready`` (viewable /
re-queryable via the vision tool), NEVER ``failed``.

The OCR fixture word ('INVOICE') was EMPIRICALLY confirmed readable by RapidOCR
in this environment via ``image_context_service.analyze`` before this test was
written — ``ocr_text == 'INVOICE'`` (Task 6 commit notes).
"""

import base64
import io

import pytest

from abilities.document import DocumentAbility
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


def _ocrable_invoice_png_b64() -> str:
    """A PNG rendering the word 'INVOICE' large/high-contrast enough for RapidOCR.

    Built with PIL. RapidOCR's read of this exact fixture was empirically
    confirmed (analyze().ocr_text == 'INVOICE') before this test relied on it.
    """
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
    draw.text((20, 40), "INVOICE", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _blank_png_b64() -> str:
    """A 1x1 white PNG — no words, so RapidOCR yields empty text."""
    return base64.b64encode(
        base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            b"+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
    ).decode()


def _result_text(out):
    """DocumentAbility.run returns {'text': <tagged body>}; unwrap to the body."""
    if isinstance(out, dict):
        return out.get("text") or out.get("result") or ""
    return out


def test_uploaded_image_is_findable_via_document_search(db):
    """No vision provider -> OCR description -> the existing embed+FTS5 pipeline ->
    document.search finds the image by content (real recall, FTS5 + vector)."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "name": "inv.png",
        "content": _ocrable_invoice_png_b64(),
        "content_type": "image/png",
    })
    up_text = _result_text(up)
    assert "id=" in up_text, up_text
    assert "hash=" in up_text, up_text  # Task 6: result now carries the file hash

    found = ability.run({"action": "search", "query": "invoice"})
    found_text = _result_text(found)
    # The image surfaced via the REAL recall path. _handle_search renders the
    # matched document's original_name in the result body.
    assert "inv.png" in found_text, found_text


def test_textless_image_is_ready_not_failed(db):
    """A textless image with no vision provider (empty OCR) must persist 'ready',
    never 'failed' — directly proves the _run_upload_extraction image-aware branch."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "name": "blank.png",
        "content": _blank_png_b64(),
        "content_type": "image/png",
    })
    up_text = _result_text(up)
    assert "id=" in up_text, up_text

    # Extract the doc_id from the result body ("... (id=<hex>, hash=...)").
    doc_id = up_text.split("id=", 1)[1].split(",", 1)[0].split(")", 1)[0].strip()

    doc = DocumentService(get_shared_db_service()).get_document(doc_id)
    assert doc is not None, up_text
    assert doc["status"] == "ready", doc.get("status")
