"""Feature test: image_context_service.analyze is OCR-only (TKT-838).

Proves the regression that even with a fully-resolvable vision provider
configured in the real DB, analyze() does pure local OCR and never dials the
provider — it is the no-vision-provider fallback for the new describe_image()
core. Drives the real analyze() entry point against the real test DB with a
real provider row created by the production ProviderDbService factory.
"""

import base64

import pytest

from services import image_context_service
from services.database_service import get_shared_db_service
from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    # 1x1 PNG — real, decodable image bytes.
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def test_analyze_is_ocr_only_even_with_vision_provider_configured(db):
    svc = ProviderDbService(get_shared_db_service())

    # Real factory creates the row. Platform 'ollama' is not key-requiring, so
    # create_provider runs a live vision probe against the unreachable host and
    # records supports_vision=0. Force the column to 1 directly in the real DB
    # so a genuine, RESOLVABLE vision provider is configured — exactly the state
    # a successful probe would have written.
    provider = svc.create_provider(
        {
            "name": "probe-vision",
            "platform": "ollama",
            "model": "llava",
            "host": "http://127.0.0.1:1",
            "api_key": "",
        }
    )
    pid = provider["id"]
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (pid,))
    db.commit()

    svc.set_vision_provider(pid)
    # A vision provider IS configured and resolvable.
    assert svc.get_vision_provider() is not None

    # Yet analyze does pure OCR: it never dials the (unreachable) provider.
    result = image_context_service.analyze(_png_bytes(), "image/png")

    assert result["vision_used"] is False
    assert "ocr_text" in result
    assert result["error"] is None
