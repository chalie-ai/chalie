"""Feature test: image extraction routes through describe_image().

Drives the REAL public entry point ``extract_text`` against the real test DB and
real PNG files. Proves the searchability-hinge rewrite of ``_extract_image``:

  1. No vision provider configured -> extract_text returns describe_image's OCR
     description WITHOUT the legacy 'vision model not configured' note (the old
     body always appended it; the new describe_image-routed path never does).
  2. A configured-but-unreachable vision provider makes extraction FAIL LOUD —
     the provider error bubbles out of extract_text and is NOT swallowed to ''.

Deterministic cross-step proofs only — the unit tier never asserts RapidOCR read
specific glyphs (that proof lives in the end-to-end evaluation suite).
"""

import base64

import pytest

from services.database_service import get_shared_db_service
from services.provider_db_service import ProviderDbService
from services.text_extractor import extract_text

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    # 1x1 PNG — real, decodable image bytes.
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_image_extraction_routes_through_describe_image_ocr_fork(db, tmp_path):
    """No vision provider -> extract_text returns describe_image's OCR description,
    WITHOUT the legacy 'vision model not configured' note (proves the re-route)."""
    ProviderDbService(get_shared_db_service()).set_vision_provider(None)
    p = tmp_path / "x.png"
    p.write_bytes(_png_bytes())

    out = extract_text(str(p))

    assert isinstance(out, str)
    # The old _extract_image always appended this note for a no-vision image; the
    # new describe_image-routed path never does. Deterministic proof of the rewrite.
    assert "vision model not configured" not in out.lower()


def test_image_extraction_provider_error_propagates_never_swallowed(db, tmp_path):
    """A configured-but-unreachable vision provider must make extraction FAIL LOUD
    (routes to describe_image's provider fork; error bubbles, not swallowed to '')."""
    svc = ProviderDbService(get_shared_db_service())
    provider = svc.create_provider({
        "name": "broken-vision", "platform": "ollama", "model": "llava",
        "host": "http://127.0.0.1:1", "api_key": "",
    })
    pid = provider["id"]
    # create_provider runs a live probe against the unreachable host and stores
    # supports_vision=0; force it resolvable exactly like the committed reference
    # test backend/tests/test_image_context_ocr_only.py does.
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (pid,))
    db.commit()
    svc.set_vision_provider(pid)

    p = tmp_path / "x.png"
    p.write_bytes(_png_bytes())

    with pytest.raises(Exception):
        extract_text(str(p))
