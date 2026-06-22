"""Feature test: browser screenshots land in the document pipeline.

Drives the REAL shared ingest (``ingest_file`` — the exact call
``BrowserAbility._screenshot`` makes) and then the REAL ``vision`` tool through
the REAL ``ToolDispatcher`` on a REAL web_browse delegate mp. Zero mocks; the
no-vision-provider fork exercises RapidOCR exactly as production does (the
'INVOICE' fixture's OCR readability was empirically proven).
Locks the contract end to end: png → screenshots/ subdir →
source_type='screenshot' → ready doc → vision reads it on the delegate channel.
"""

import io
import sqlite3
from typing import cast

import pytest

from abilities._dispatcher import ToolDispatcher
from abilities.document import ingest_file
from configs.channels.web_browse import WebBrowseConfig
from services.database_service import get_shared_db_service
from services.document_service import DocumentService
from services.file_mapper_service import FileMapperService
from services.message_processor import MessageProcessor
from services.processor_config import ProcessorConfig
from services.provider_db_service import ProviderDbService
from services.tmp_storage import new_tmp_path

pytestmark = pytest.mark.unit


def _invoice_png_path() -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    draw = ImageDraw.Draw(img)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
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
    path = new_tmp_path("shot.png")
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return path


def test_screenshot_ingest_lands_in_screenshots_subdir_and_vision_reads_it(db: sqlite3.Connection) -> None:
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    service = DocumentService(get_shared_db_service())

    ingested = ingest_file(
        service,
        _invoice_png_path(),
        name="screenshot-example.com.png",
        subdir="screenshots",
        source_type="screenshot",
    )

    assert not ingested.get("error"), ingested
    assert ingested["status"] == "ready", ingested
    doc = cast(dict[str, object], service.get_document(cast(str, ingested["id"])))
    assert doc["source_type"] == "screenshot"
    # The exact field BrowserAbility._screenshot hands back inline as data["vision"]:
    # ingest ran the png through the image extractor (OCR here, vision in prod) and
    # stored the readable content as clean_text — proving the screenshot self-describes.
    assert "INVOICE" in cast(str, doc.get("clean_text") or ""), doc.get("clean_text")
    assert cast(str, doc["file_path"]).startswith("screenshots/"), doc["file_path"]
    stored = FileMapperService.get_documents_path(cast(str, doc["file_path"]))
    assert stored.is_file(), f"file missing on disk: {stored}"
    assert FileMapperService.validate_document_path(str(stored))

    # The delegate reads its own screenshot via the REAL dispatch chokepoint.
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "look at the screenshot", {})
    mp.config = WebBrowseConfig(ProcessorConfig.PolicyChannel.CHAT)
    mp._setup()
    out = ToolDispatcher(mp).dispatch(
        "vision", {"image": ingested["id"], "query": "what text is in this image"}
    )
    assert "INVOICE" in str(out), out
