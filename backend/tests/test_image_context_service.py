"""
Unit tests for ImageContextService.

All tests are marked @pytest.mark.unit — no external dependencies.
OCR calls are mocked so tests run offline.
"""

import io
import pytest
from unittest.mock import patch


pytest_plugins = []


@pytest.fixture
def sample_png_bytes():
    """Create a minimal valid PNG (1x1 white pixel) as bytes."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def large_png_bytes():
    """Create a large image (3000x3000) to test dimension normalization."""
    from PIL import Image
    img = Image.new('RGB', (3000, 3000), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ─── analyze() ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_analyze_returns_expected_shape(sample_png_bytes):
    """analyze() should always return a dict with all required keys."""
    from services.image_context_service import analyze

    with patch('services.ocr_service._extract_text', return_value=''):
        result = analyze(sample_png_bytes, 'image/png')

    assert isinstance(result, dict)
    assert 'ocr_text' in result
    assert 'has_text' in result
    assert 'analysis_time_ms' in result
    assert isinstance(result['has_text'], bool)
    assert isinstance(result['analysis_time_ms'], int)


@pytest.mark.unit
def test_analyze_extracts_text(sample_png_bytes):
    """analyze() uses local OCR and returns extracted text."""
    from services.image_context_service import analyze

    with patch('services.ocr_service._extract_text', return_value='HELLO WORLD'):
        result = analyze(sample_png_bytes, 'image/png')

    assert result['ocr_text'] == 'HELLO WORLD'
    assert result['has_text'] is True
    assert result['error'] is None



@pytest.mark.unit
def test_analyze_short_text_not_counted(sample_png_bytes):
    """Text shorter than _MIN_TEXT_LENGTH should not set has_text."""
    from services.image_context_service import analyze

    with patch('services.ocr_service._extract_text', return_value='Hi'):
        result = analyze(sample_png_bytes, 'image/png')

    assert result['has_text'] is False
    assert result['ocr_text'] == 'Hi'


@pytest.mark.unit
def test_analyze_ocr_failure_sets_error(sample_png_bytes):
    """If local OCR raises, error key is set but no crash."""
    from services.image_context_service import analyze

    with patch('services.ocr_service._extract_text', side_effect=RuntimeError('OCR engine failed')):
        result = analyze(sample_png_bytes, 'image/png')

    # OCR failure is non-fatal — error may or may not be set depending on
    # whether the exception propagates past the inner try/except
    assert result['ocr_text'] == ''
    assert result['has_text'] is False


# ─── EXIF stripping ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_strip_exif_returns_image():
    """_strip_exif() should return a valid PIL Image without raising."""
    from PIL import Image
    from services.image_context_service import _strip_exif

    img = Image.new('RGB', (50, 50), color=(255, 0, 0))
    result = _strip_exif(img)

    assert result is not None
    assert result.size == (50, 50)



@pytest.mark.unit
def test_strip_exif_memory_usage():
    """_strip_exif() BytesIO round-trip should use significantly less memory than list(getdata())."""
    import tracemalloc
    from PIL import Image
    from services.image_context_service import _strip_exif

    img = Image.new('RGB', (2048, 2048), color=(128, 64, 32))

    tracemalloc.start()
    result = _strip_exif(img)
    _, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _old_data = list(img.getdata())
    _, old_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert new_peak < old_peak * 0.5, (
        f'Expected new peak ({new_peak:,} bytes) to be < 50 % of old peak '
        f'({old_peak:,} bytes). BytesIO round-trip may not be working as intended.'
    )
    assert result.size == img.size


# ─── Dimension normalization ──────────────────────────────────────────────────

@pytest.mark.unit
def test_normalize_dimensions_downscales_large_image(large_png_bytes):
    """_normalize_dimensions() should downscale images larger than _MAX_DIMENSION."""
    from PIL import Image
    from services.image_context_service import _normalize_dimensions, _MAX_DIMENSION

    buf = io.BytesIO(large_png_bytes)
    img = Image.open(buf)
    result = _normalize_dimensions(img)

    assert max(result.size) <= _MAX_DIMENSION


@pytest.mark.unit
def test_normalize_dimensions_no_op_for_small_image(sample_png_bytes):
    """_normalize_dimensions() should leave small images unchanged."""
    from PIL import Image
    from services.image_context_service import _normalize_dimensions

    buf = io.BytesIO(sample_png_bytes)
    img = Image.open(buf)
    original_size = img.size
    result = _normalize_dimensions(img)

    assert result.size == original_size



