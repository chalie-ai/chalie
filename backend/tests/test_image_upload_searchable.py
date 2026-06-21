"""Feature test: an uploaded image is findable by its visual content.

END-TO-END SEARCHABILITY PROOF — the heart of the Vision Subagent addendum.

Drives the REAL ``DocumentAbility.run`` upload + search entry points against the
real test DB and real files. ZERO mocks: the OCR description produced by the
no-vision-provider fork flows through the SAME production pipeline
(``create_document_artifacts`` -> data_graph embed + FTS5) that every upload
uses, and ``document(action='search')`` recalls it via the REAL recall path
(FTS5 + vector). Proves the [document_id] guardrail is truthful: an image with words
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
import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from abilities.document import DocumentAbility
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont, ImageFont as PILImageFont

pytestmark = pytest.mark.unit


def _ocrable_invoice_png_path() -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: "FreeTypeFont | PILImageFont | None" = None
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
    path = new_tmp_path("inv.png")
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return path


def _blank_png_path() -> str:
    """A 1x1 white PNG (no words -> empty OCR), written under the temp prefix."""
    path = new_tmp_path("blank.png")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            b"+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        ))
    return path


def test_uploaded_image_is_findable_via_document_search(db: sqlite3.Connection) -> None:
    """No vision provider -> OCR description -> the existing embed+FTS5 pipeline ->
    document.search finds the image by content (real recall, FTS5 + vector)."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "path": _ocrable_invoice_png_path(),
    })
    # : upload now returns a structured ToolResult body
    # ``{"id","hash","name","status"}`` (Task 6 hash preserved as a body key).
    assert up.status == "success", up
    assert cast(dict[str, object], up.body)["id"], up.body
    assert cast(dict[str, object], up.body)["hash"], up.body

    found = ability.run({"action": "search", "query": "invoice"})
    # : search returns a JSON list of document rows; the matched image
    # surfaces via the REAL recall path under its original_name.
    assert any("inv.png" in cast(str, cast(dict[str, object], row).get("name") or "") for row in cast(list[object], found.body)), found.body


def test_textless_image_is_ready_not_failed(db: sqlite3.Connection) -> None:
    """A textless image with no vision provider (empty OCR) must persist 'ready',
    never 'failed' — directly proves the _run_upload_extraction image-aware branch."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)

    ability = DocumentAbility(mp=None)
    up = ability.run({
        "action": "upload",
        "path": _blank_png_path(),
    })
    # : doc id comes straight off the structured upload body.
    assert up.status == "success", up
    [document_id] = cast(str, cast(dict[str, object], up.body)["id"])

    doc = DocumentService(get_shared_db_service()).get_document([document_id])
    assert doc is not None, up
    assert doc["status"] == "ready", doc.get("status")
