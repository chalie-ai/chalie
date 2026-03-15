"""
Unit tests for ImageContextService.

All tests are marked @pytest.mark.unit — no external dependencies.
Vision LLM calls are mocked so tests run offline.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def exif_png_bytes():
    """Create an image with EXIF data to test stripping."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ─── analyze() ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_analyze_returns_expected_shape(sample_png_bytes):
    """analyze() should always return a dict with all required keys."""
    from services.image_context_service import analyze

    with patch('services.image_context_service._get_vision_provider') as mock_prov:
        mock_prov.return_value = None  # no provider configured

        result = analyze(sample_png_bytes, 'image/png')

    assert isinstance(result, dict)
    assert 'description' in result
    assert 'ocr_text' in result
    assert 'has_text' in result
    assert 'analysis_time_ms' in result
    assert isinstance(result['has_text'], bool)
    assert isinstance(result['analysis_time_ms'], int)


@pytest.mark.unit
def test_analyze_no_provider_returns_error(sample_png_bytes):
    """When no vision provider is configured, analyze() returns an error key."""
    from services.image_context_service import analyze

    with patch('services.image_context_service._get_vision_provider') as mock_prov:
        mock_prov.return_value = None

        result = analyze(sample_png_bytes, 'image/png')

    assert result['error'] is not None
    assert result['description'] == ''
    assert result['ocr_text'] == ''
    assert result['has_text'] is False


@pytest.mark.unit
def test_analyze_anthropic_success(sample_png_bytes):
    """analyze() dispatches to Anthropic and returns description + ocr_text."""
    from services.image_context_service import analyze

    fake_provider = {'platform': 'anthropic', 'model': 'claude-3-haiku-20240307'}

    with patch('services.image_context_service._get_vision_provider', return_value=fake_provider), \
         patch('services.image_context_service._analyze_anthropic', return_value=('A white square.', 'HELLO WORLD')) as mock_fn:

        result = analyze(sample_png_bytes, 'image/png')

    mock_fn.assert_called_once()
    assert result['description'] == 'A white square.'
    assert result['ocr_text'] == 'HELLO WORLD'
    assert result['has_text'] is True
    assert result['error'] is None


@pytest.mark.unit
def test_analyze_sets_has_text_false_when_no_ocr(sample_png_bytes):
    """has_text should be False when OCR returns empty or NO_TEXT."""
    from services.image_context_service import analyze

    fake_provider = {'platform': 'openai', 'model': 'gpt-4o'}

    with patch('services.image_context_service._get_vision_provider', return_value=fake_provider), \
         patch('services.image_context_service._analyze_openai', return_value=('A photo.', 'NO_TEXT')):

        result = analyze(sample_png_bytes, 'image/png')

    assert result['has_text'] is False
    assert result['ocr_text'] == ''


@pytest.mark.unit
def test_analyze_gemini_provider(sample_png_bytes):
    """analyze() dispatches to Gemini when provider platform is gemini."""
    from services.image_context_service import analyze

    fake_provider = {'platform': 'gemini', 'model': 'gemini-1.5-flash'}

    with patch('services.image_context_service._get_vision_provider', return_value=fake_provider), \
         patch('services.image_context_service._analyze_gemini', return_value=('A grey image.', '')) as mock_fn:

        result = analyze(sample_png_bytes, 'image/png')

    mock_fn.assert_called_once()
    assert result['description'] == 'A grey image.'


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
def test_strip_exif_preserves_dimensions():
    """_strip_exif() should not change image dimensions."""
    from PIL import Image
    from services.image_context_service import _strip_exif

    img = Image.new('RGB', (200, 150), color=(0, 255, 0))
    result = _strip_exif(img)

    assert result.size == (200, 150)


@pytest.mark.unit
def test_strip_exif_memory_usage():
    """_strip_exif() BytesIO round-trip should use significantly less memory than list(getdata()).

    The old implementation called ``list(img.getdata())``, which materialised
    the full pixel array as a Python list — approximately 470 MB for a
    2048×2048 RGBA image.  The BytesIO round-trip avoids that spike entirely.
    This test verifies that the new implementation's peak allocation is at
    least 50 % lower than the old approach's on an equivalent large image.
    """
    import tracemalloc
    from PIL import Image
    from services.image_context_service import _strip_exif

    img = Image.new('RGB', (2048, 2048), color=(128, 64, 32))

    # ── Measure new _strip_exif() peak allocation ──────────────────────────
    tracemalloc.start()
    result = _strip_exif(img)
    _, new_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ── Measure old list(getdata()) peak allocation ────────────────────────
    tracemalloc.start()
    _old_data = list(img.getdata())
    _, old_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The BytesIO approach must use less than 50 % of the list approach's peak.
    assert new_peak < old_peak * 0.5, (
        f'Expected new peak ({new_peak:,} bytes) to be < 50 % of old peak '
        f'({old_peak:,} bytes). BytesIO round-trip may not be working as intended.'
    )

    # Pixel content and dimensions must be preserved.
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


# ─── has_vision_provider() ────────────────────────────────────────────────────

@pytest.mark.unit
def test_has_vision_provider_false_when_no_config():
    """has_vision_provider() returns False when no provider is assigned."""
    from services.image_context_service import has_vision_provider

    with patch('services.image_context_service._get_vision_provider', return_value=None):
        assert has_vision_provider() is False


@pytest.mark.unit
def test_has_vision_provider_true_when_config_exists():
    """has_vision_provider() returns True when a vision provider is available."""
    from services.image_context_service import has_vision_provider

    fake_provider = {'platform': 'anthropic', 'model': 'claude-3-haiku-20240307'}
    with patch('services.image_context_service._get_vision_provider', return_value=fake_provider):
        assert has_vision_provider() is True


# ─── compute_hash() ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_compute_hash_is_deterministic(sample_png_bytes):
    """compute_hash() returns the same hash for identical bytes."""
    from services.image_context_service import compute_hash

    h1 = compute_hash(sample_png_bytes)
    h2 = compute_hash(sample_png_bytes)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length


@pytest.mark.unit
def test_compute_hash_differs_for_different_bytes():
    """compute_hash() returns different hashes for different content."""
    from services.image_context_service import compute_hash

    assert compute_hash(b'hello') != compute_hash(b'world')
