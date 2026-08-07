"""Feature test: browser screenshots land in the documents store.

Drives the REAL shared ingest (``FileParserService.ingest`` — the exact call
``BrowserAbility._screenshot`` makes) and then the REAL ``vision`` tool
through the REAL dispatch chokepoint (``mp.dispatch_service.dispatch``) on a
REAL web_browse delegate mp. Zero mocks; the no-vision-provider fork
exercises RapidOCR exactly as production does (the 'INVOICE' fixture's OCR
readability was empirically proven).

Locks the contract end to end: png -> flat screenshots/ subdir (no per-file
folder — collisions get ``_1``, ``_2``… before the extension) -> real OCR
extraction -> FileIndexService upsert -> vision reads the image by its
absolute filepath on the delegate channel.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from configs.channels.web_browse import WebBrowseConfig
from configs.enums.policy_channel import PolicyChannel
from controllers.message_processor import MessageProcessor
from services.file_index_service import FileIndexService
from services.file_mapper_service import FileMapperService
from services.file_parser_service import FileParserService
from services.provider_db_service import ProviderDbService
from services.tmp_storage import TmpStorage

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
    path = TmpStorage.new_tmp_path("shot.png")
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return path


def test_screenshot_ingest_lands_flat_in_screenshots_subdir_and_vision_reads_it(
    db: sqlite3.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProviderDbService().set_vision_provider(None)

    # Hermetic isolation (same style as test_snapshot_docs_roundtrip.py's
    # _redirect_snapshot_paths): both patched BEFORE FileParserService runs —
    # FileIndexService()'s db_path is resolved at call time inside ingest(),
    # so the classmethod patch must already be in place to reach it.
    documents_dir = tmp_path / "documents"
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", documents_dir)
    monkeypatch.setattr(
        FileMapperService, "get_file_index_db_path", lambda *_: tmp_path / "file_index.sqlite"
    )

    saved_path, text = FileParserService().ingest(
        _invoice_png_path(), name="screenshot-example.com.png", subdir="screenshots",
    )

    assert "INVOICE" in text, text
    expected_path = documents_dir / "screenshots" / "screenshot-example.com.png"
    assert saved_path == str(expected_path), saved_path
    assert Path(saved_path).is_file(), f"file missing on disk: {saved_path}"
    assert FileMapperService.validate_document_path(saved_path)

    # The index upsert: FileIndexService().search finds the image by its OCR'd
    # description content, not just its filename.
    assert saved_path in FileIndexService().search("INVOICE")

    # The delegate reads its own screenshot via the REAL dispatch chokepoint,
    # by absolute filepath — the new vision contract.
    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(
        mp, WebBrowseConfig(PolicyChannel.CHAT), raw_input="look at the screenshot",
    )
    mp._setup()
    out = mp.dispatch_service.dispatch(
        "vision", {"image": saved_path, "query": "what text is in this image"}
    )
    assert "INVOICE" in str(out), out
