"""Feature test: image_context_service.analyze is OCR-only.

Proves the regression that even with a fully-resolvable vision provider
configured in the real DB, analyze() does pure local OCR and never dials the
provider — it is the fallback rung of the ImageDescription core, used whenever
the vision provider is unset OR fails. Drives the real analyze() entry point
against the real test DB with a real provider row created by the production
ProviderDbService factory.
"""

import sqlite3

import pytest

from services import image_context_service
from services.provider_db_service import ProviderDbService
from tests.helpers import blank_png_bytes, force_vision_provider, ocrable_png_bytes

pytestmark = pytest.mark.unit


def test_analyze_is_ocr_only_even_with_vision_provider_configured(db: sqlite3.Connection) -> None:
    # A genuine, RESOLVABLE vision provider IS configured — and unreachable, so
    # any attempt to dial it would surface as an error rather than pass silently.
    force_vision_provider(db)
    assert ProviderDbService().get_vision_provider() is not None

    # Yet analyze does pure OCR: it never dials the provider, and reads the words.
    result = image_context_service.analyze(ocrable_png_bytes())

    assert result["vision_used"] is False
    assert "INVOICE" in str(result["ocr_text"]).upper()
    assert result["error"] is None


def test_analyze_reads_nothing_from_a_textless_image(db: sqlite3.Connection) -> None:
    """The empty-OCR pole: a textless image is NOT an analyze() error — it simply
    yields no text. Deciding that a description is mandatory is ImageDescription's
    job, not this service's."""
    result = image_context_service.analyze(blank_png_bytes())

    assert result["ocr_text"] == ""
    assert result["has_text"] is False
    assert result["error"] is None
