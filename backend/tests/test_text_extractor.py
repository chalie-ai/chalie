"""
Tests for services/text_extractor.py — shared extraction library.
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── detect_mime_type ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDetectMimeType:
    def test_pdf(self):
        from services.text_extractor import detect_mime_type
        assert detect_mime_type('/path/to/doc.pdf') == 'application/pdf'


# ─── normalize_text ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizeText:
    def test_removes_control_chars(self):
        from services.text_extractor import normalize_text
        result = normalize_text("hello\x00\x01\x07world")
        assert '\x00' not in result
        assert 'hello' in result
        assert 'world' in result

    def test_none_safe(self):
        from services.text_extractor import normalize_text
        assert normalize_text(None) == ''


# ─── extract_html ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractHtml:
    def test_trafilatura_primary(self):
        html = "<html><body><article><p>Real article content here.</p></article></body></html>"
        mock_traf = MagicMock()
        mock_traf.extract.return_value = "Real article content here."
        with patch.dict('sys.modules', {'trafilatura': mock_traf}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html, url='https://example.com')
        assert 'Real article content' in result

    def test_trafilatura_returns_none_returns_empty(self):
        html = "<html><body><article>Actual content.</article></body></html>"
        mock_traf = MagicMock()
        mock_traf.extract.return_value = None
        with patch.dict('sys.modules', {'trafilatura': mock_traf}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        assert result == ''

    def test_trafilatura_import_error_returns_empty(self):
        html = "<html><body><p>Content here.</p></body></html>"
        with patch.dict('sys.modules', {'trafilatura': None}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        assert result == ''

    def test_empty_html_returns_empty(self):
        from services.text_extractor import extract_html
        assert extract_html('') == ''
        assert extract_html('   ') == ''


# ─── extract_text dispatch ────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractText:
    def test_dispatches_to_pdf_extractor(self):
        with patch('services.text_extractor._extract_pdf', return_value='pdf text') as mock_pdf:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/doc.pdf', 'application/pdf')
        mock_pdf.assert_called_once_with('/tmp/doc.pdf')
        assert result == 'pdf text'

    def test_unknown_mime_falls_back_to_plain(self):
        with patch('services.text_extractor._extract_plain', return_value='plain text') as mock_plain:
            from services.text_extractor import extract_text
            extract_text('/tmp/file.xyz', 'application/unknown-binary')
        mock_plain.assert_called_once()

    def test_dispatches_png_to_image_extractor(self, tmp_path):
        png_path = str(tmp_path / 'x.png')
        with patch('services.text_extractor._extract_image', return_value='HELLO CHALIE 2040') as mock_img:
            from services.text_extractor import extract_text
            result = extract_text(png_path, 'image/png')
        mock_img.assert_called_once_with(png_path)
        assert result == 'HELLO CHALIE 2040'

    def test_unregistered_image_subtype_routes_via_wildcard(self, tmp_path):
        """Unmapped image/* (e.g. image/heic) hits the wildcard, not _extract_plain."""
        heic_path = str(tmp_path / 'photo.heic')
        with patch('services.text_extractor._extract_image', return_value='heic text') as mock_img:
            from services.text_extractor import extract_text
            result = extract_text(heic_path, 'image/heic')
        mock_img.assert_called_once_with(heic_path)
        assert result == 'heic text'


# ─── _extract_image — delegates to image_context_service ─────────────────────

@pytest.mark.unit
class TestExtractImage:
    def test_returns_ocr_text_from_analyze(self, tmp_path):
        img_file = tmp_path / "scan.png"
        img_file.write_bytes(b"\x89PNG fake bytes")
        with patch('services.image_context_service.analyze',
                   return_value={'ocr_text': 'HELLO CHALIE 2040', 'error': None}) as mock_analyze:
            from services.text_extractor import _extract_image
            result = _extract_image(str(img_file))
        assert result == 'HELLO CHALIE 2040'
        mock_analyze.assert_called_once_with(b"\x89PNG fake bytes")

    def test_empty_ocr_returns_empty_string(self, tmp_path):
        img_file = tmp_path / "blank.png"
        img_file.write_bytes(b"data")
        with patch('services.image_context_service.analyze',
                   return_value={'ocr_text': '', 'error': None}):
            from services.text_extractor import _extract_image
            assert _extract_image(str(img_file)) == ''

    def test_unreadable_file_returns_empty(self):
        from services.text_extractor import _extract_image
        assert _extract_image('/nonexistent/path/x.png') == ''




# ─── PDF missing dependency ───────────────────────────────────────────────────

@pytest.mark.unit
class TestMissingDependency:
    def test_pdf_missing_pdfplumber_returns_empty(self):
        with patch.dict('sys.modules', {'pdfplumber': None}):
            from services import text_extractor
            import importlib
            importlib.reload(text_extractor)
            result = text_extractor._extract_pdf('/tmp/doc.pdf')
        assert result == ''


# ─── no-vision note ───────────────────────────────────────────────────────────

_NOTE = 'note: only text can be extracted. vision model not configured'


@pytest.mark.unit
def test_extract_image_appends_note_when_no_vision(tmp_path):
    from services import text_extractor
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    with patch("services.image_context_service.analyze",
               return_value={"ocr_text": "SOME TEXT", "vision_used": False, "error": None}):
        out = text_extractor._extract_image(str(img))
    assert out == f"SOME TEXT\n{_NOTE}"


@pytest.mark.unit
def test_extract_image_note_only_when_no_text_no_vision(tmp_path):
    from services import text_extractor
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    with patch("services.image_context_service.analyze",
               return_value={"ocr_text": "", "vision_used": False, "error": None}):
        out = text_extractor._extract_image(str(img))
    assert out == _NOTE


@pytest.mark.unit
def test_extract_image_no_note_when_vision_used(tmp_path):
    from services import text_extractor
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    with patch("services.image_context_service.analyze",
               return_value={"ocr_text": "a detailed description", "vision_used": True, "error": None}):
        out = text_extractor._extract_image(str(img))
    assert out == "a detailed description"


