"""
Unit tests for image extraction in services/text_extractor.py.

All tests are marked @pytest.mark.unit — no external dependencies.
OCR calls and PIL are mocked so tests run offline without invoking
the real RapidOCR engine.
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── _extract_image via extract_text (MIME-based dispatch) ────────────────────

@pytest.mark.unit
def test_extract_text_png_returns_ocr_text():
    """extract_text routes image/png to OCR and returns the extracted string."""
    mock_img = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_img

    with patch.dict('sys.modules', {'PIL': mock_pil, 'PIL.Image': mock_pil.Image}):
        with patch('services.ocr_service._extract_text', return_value='HELLO CHALIE 2040') as mock_ocr:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/x.png', 'image/png')

    assert result == 'HELLO CHALIE 2040'
    mock_ocr.assert_called_once_with(mock_img)


@pytest.mark.unit
def test_extract_text_jpeg_routes_to_image_branch():
    """extract_text routes image/jpeg through _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='jpeg text') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/photo.jpg', 'image/jpeg')

    mock_img.assert_called_once_with('/tmp/photo.jpg')
    assert result == 'jpeg text'


@pytest.mark.unit
def test_extract_text_webp_routes_to_image_branch():
    """extract_text routes image/webp through _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='webp text') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/image.webp', 'image/webp')

    mock_img.assert_called_once_with('/tmp/image.webp')
    assert result == 'webp text'


@pytest.mark.unit
def test_extract_text_gif_routes_to_image_branch():
    """extract_text routes image/gif through _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/anim.gif', 'image/gif')

    mock_img.assert_called_once_with('/tmp/anim.gif')
    assert result == ''


@pytest.mark.unit
def test_extract_text_bmp_routes_to_image_branch():
    """extract_text routes image/bmp through _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='bmp text') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/bitmap.bmp', 'image/bmp')

    mock_img.assert_called_once_with('/tmp/bitmap.bmp')
    assert result == 'bmp text'


@pytest.mark.unit
def test_extract_text_tiff_routes_to_image_branch():
    """extract_text routes image/tiff through _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='tiff text') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/scan.tiff', 'image/tiff')

    mock_img.assert_called_once_with('/tmp/scan.tiff')
    assert result == 'tiff text'


# ─── image/* wildcard short-circuit ──────────────────────────────────────────

@pytest.mark.unit
def test_extract_text_heic_routes_via_image_wildcard():
    """Unregistered image/* MIME (e.g. image/heic) routes through the wildcard to _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='heic text') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/photo.heic', 'image/heic')

    mock_img.assert_called_once_with('/tmp/photo.heic')
    assert result == 'heic text'


@pytest.mark.unit
def test_extract_text_unknown_image_subtype_routes_via_wildcard():
    """Any image/* MIME not in the explicit map still routes to _extract_image."""
    with patch('services.text_extractor._extract_image', return_value='') as mock_img:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/img.unknown', 'image/x-custom-format')

    mock_img.assert_called_once()


# ─── ImportError graceful degradation ────────────────────────────────────────

@pytest.mark.unit
def test_extract_image_pil_import_error_returns_empty():
    """_extract_image returns '' and does not raise when PIL is missing."""
    with patch.dict('sys.modules', {'PIL': None, 'PIL.Image': None}):
        import importlib
        import services.text_extractor as te
        importlib.reload(te)
        result = te._extract_image('/tmp/x.png')

    assert result == ''


@pytest.mark.unit
def test_extract_image_ocr_exception_returns_empty():
    """_extract_image returns '' and does not raise when OCR raises."""
    mock_img = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_img

    with patch.dict('sys.modules', {'PIL': mock_pil, 'PIL.Image': mock_pil.Image}):
        with patch('services.ocr_service._extract_text', side_effect=RuntimeError('OCR engine failed')):
            from services.text_extractor import _extract_image
            result = _extract_image('/tmp/x.png')

    assert result == ''


@pytest.mark.unit
def test_extract_image_returns_empty_string_on_empty_ocr():
    """_extract_image returns '' when OCR produces no text."""
    mock_img = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_img

    with patch.dict('sys.modules', {'PIL': mock_pil, 'PIL.Image': mock_pil.Image}):
        with patch('services.ocr_service._extract_text', return_value=''):
            from services.text_extractor import _extract_image
            result = _extract_image('/tmp/blank.png')

    assert result == ''


# ─── Non-image paths unaffected ──────────────────────────────────────────────

@pytest.mark.unit
def test_extract_text_pdf_unaffected():
    """Image changes do not break the PDF dispatch path."""
    with patch('services.text_extractor._extract_pdf', return_value='pdf content') as mock_pdf:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/doc.pdf', 'application/pdf')

    mock_pdf.assert_called_once_with('/tmp/doc.pdf')
    assert result == 'pdf content'


@pytest.mark.unit
def test_extract_text_plain_text_unaffected():
    """Image changes do not break the text/* fallback path."""
    with patch('services.text_extractor._extract_plain', return_value='csv data') as mock_plain:
        from services.text_extractor import extract_text
        result = extract_text('/tmp/data.csv', 'text/csv')

    mock_plain.assert_called_once()
    assert result == 'csv data'
