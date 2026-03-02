"""
Tests for services/text_extractor.py — shared extraction library.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open


# ─── detect_mime_type ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDetectMimeType:
    def test_pdf(self):
        from services.text_extractor import detect_mime_type
        assert detect_mime_type('/path/to/doc.pdf') == 'application/pdf'

    def test_docx(self):
        from services.text_extractor import detect_mime_type
        result = detect_mime_type('/path/to/report.docx')
        assert 'wordprocessingml' in result

    def test_pptx(self):
        from services.text_extractor import detect_mime_type
        result = detect_mime_type('/path/to/slides.pptx')
        assert 'presentationml' in result

    def test_html(self):
        from services.text_extractor import detect_mime_type
        assert detect_mime_type('/path/to/page.html') == 'text/html'

    def test_markdown(self):
        from services.text_extractor import detect_mime_type
        result = detect_mime_type('/path/to/README.md')
        assert result is not None  # stdlib may return text/markdown or text/x-markdown

    def test_txt(self):
        from services.text_extractor import detect_mime_type
        assert detect_mime_type('/path/to/notes.txt') == 'text/plain'

    def test_unknown_extension_falls_back(self):
        from services.text_extractor import detect_mime_type
        result = detect_mime_type('/path/to/file.xyz_unknown')
        assert result == 'text/plain'


# ─── normalize_text ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizeText:
    def test_removes_control_chars(self):
        from services.text_extractor import normalize_text
        result = normalize_text("hello\x00\x01\x07world")
        assert '\x00' not in result
        assert 'hello' in result
        assert 'world' in result

    def test_collapses_multiple_spaces(self):
        from services.text_extractor import normalize_text
        result = normalize_text("hello    world")
        assert result == 'hello world'

    def test_collapses_excess_newlines(self):
        from services.text_extractor import normalize_text
        result = normalize_text("a\n\n\n\n\nb")
        assert result == 'a\n\nb'

    def test_empty_string(self):
        from services.text_extractor import normalize_text
        assert normalize_text('') == ''

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

    def test_trafilatura_returns_none_falls_to_bs4(self):
        html = (
            "<html><body>"
            "<nav>Nav noise</nav>"
            "<article>Actual content.</article>"
            "<footer>Footer noise</footer>"
            "</body></html>"
        )
        mock_traf = MagicMock()
        mock_traf.extract.return_value = None
        with patch.dict('sys.modules', {'trafilatura': mock_traf}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        # BS4 fallback should strip nav/footer
        assert 'Actual content' in result
        assert 'Nav noise' not in result
        assert 'Footer noise' not in result

    def test_bs4_strips_ad_classes(self):
        html = (
            "<html><body>"
            "<div class='ad-banner'>Buy now!</div>"
            "<div class='content'>The real content.</div>"
            "</body></html>"
        )
        mock_traf = MagicMock()
        mock_traf.extract.return_value = None
        with patch.dict('sys.modules', {'trafilatura': mock_traf}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        assert 'real content' in result
        assert 'Buy now' not in result

    def test_trafilatura_import_error_falls_to_bs4(self):
        html = "<html><body><p>Content here.</p></body></html>"
        with patch.dict('sys.modules', {'trafilatura': None}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        assert isinstance(result, str)

    def test_empty_html_returns_empty(self):
        from services.text_extractor import extract_html
        assert extract_html('') == ''
        assert extract_html('   ') == ''

    def test_short_content_accepted(self):
        """Short content is NOT rejected — no minimum char gate."""
        html = "<html><body><p>Hi.</p></body></html>"
        mock_traf = MagicMock()
        mock_traf.extract.return_value = "Hi."
        with patch.dict('sys.modules', {'trafilatura': mock_traf}):
            import importlib
            import services.text_extractor as te
            importlib.reload(te)
            result = te.extract_html(html)
        assert result == "Hi."


# ─── extract_text dispatch ────────────────────────────────────────────────────

@pytest.mark.unit
class TestExtractText:
    def test_dispatches_to_pdf_extractor(self):
        with patch('services.text_extractor._extract_pdf', return_value='pdf text') as mock_pdf:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/doc.pdf', 'application/pdf')
        mock_pdf.assert_called_once_with('/tmp/doc.pdf')
        assert result == 'pdf text'

    def test_dispatches_to_html_extractor(self):
        with patch('services.text_extractor._extract_html_file', return_value='html text') as mock_html:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/page.html', 'text/html')
        mock_html.assert_called_once_with('/tmp/page.html')
        assert result == 'html text'

    def test_auto_detects_mime_from_extension(self):
        with patch('services.text_extractor._extract_pdf', return_value='pdf content') as mock_pdf:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/document.pdf')
        mock_pdf.assert_called_once()

    def test_unknown_mime_falls_back_to_plain(self):
        with patch('services.text_extractor._extract_plain', return_value='plain text') as mock_plain:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/file.xyz', 'application/unknown-binary')
        mock_plain.assert_called_once()

    def test_text_subtype_falls_back_to_plain(self):
        with patch('services.text_extractor._extract_plain', return_value='csv content') as mock_plain:
            from services.text_extractor import extract_text
            result = extract_text('/tmp/data.csv', 'text/csv')
        mock_plain.assert_called_once()


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

    def test_docx_missing_python_docx_returns_empty(self):
        with patch.dict('sys.modules', {'docx': None}):
            from services import text_extractor
            import importlib
            importlib.reload(text_extractor)
            result = text_extractor._extract_docx('/tmp/doc.docx')
        assert result == ''

    def test_pptx_missing_python_pptx_returns_empty(self):
        with patch.dict('sys.modules', {'pptx': None}):
            from services import text_extractor
            import importlib
            importlib.reload(text_extractor)
            result = text_extractor._extract_pptx('/tmp/slides.pptx')
        assert result == ''
