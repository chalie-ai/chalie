"""
Text Extractor — Shared text extraction library for files and HTML strings.

Pure functions — no database, no MemoryStore, no Chalie services.
Used by DocumentProcessingService (file pipeline) and the `read` innate skill
(URL fetch + local file read).

Supported formats:
  - PDF         (pdfplumber)
  - DOCX        (python-docx)
  - PPTX        (python-pptx)
  - HTML        (trafilatura → BeautifulSoup → regex strip)
  - Plain text  (direct read)
  - Markdown    (direct read)
  - Any text/*  (direct read)

All heavy-library imports are lazy so missing optional deps degrade gracefully.
"""

import logging
import mimetypes
import re

logger = logging.getLogger(__name__)


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_text(file_path: str, mime_type: str = None) -> str:
    """
    Extract plain text from a local file.

    Dispatches to a format-specific extractor based on MIME type.
    If mime_type is not provided, it is inferred from the file extension.

    Returns empty string on failure (never raises).
    """
    if not mime_type:
        mime_type = detect_mime_type(file_path)

    extractors = {
        'application/pdf': _extract_pdf,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': _extract_docx,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': _extract_pptx,
        'text/html': _extract_html_file,
        'text/plain': _extract_plain,
        'text/markdown': _extract_plain,
    }

    extractor = extractors.get(mime_type)
    if extractor:
        return extractor(file_path)

    # Fallback: any text/* type (code files, CSV, etc.)
    if mime_type and mime_type.startswith('text/'):
        return _extract_plain(file_path)

    logger.warning(f"[TEXT EXTRACTOR] Unsupported mime type '{mime_type}' — attempting plain read")
    return _extract_plain(file_path)


def extract_html(html: str, url: str = None) -> str:
    """
    Extract clean, readable text from an HTML string.

    Extraction pipeline:
      1. trafilatura — best-in-class article extraction with boilerplate removal
      2. BeautifulSoup — aggressive noise stripping (ads, nav, sidebars, hidden elements)
      3. Regex strip — last resort, removes all tags and normalizes whitespace

    Returns empty string if all methods fail or produce no content.
    """
    if not html or not html.strip():
        return ''

    # 1. trafilatura (primary) — trusts its result regardless of length
    try:
        import trafilatura
        content = trafilatura.extract(html, url=url, include_comments=False)
        if content and content.strip():
            return content.strip()
    except ImportError:
        logger.warning('[TEXT EXTRACTOR] trafilatura not installed — falling back to BeautifulSoup')
    except Exception as e:
        logger.debug(f'[TEXT EXTRACTOR] trafilatura failed: {e}')

    # 2. BeautifulSoup with aggressive ad/noise stripping
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')

        # Remove structural noise elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'noscript', 'iframe', 'form']):
            tag.decompose()

        # Remove ad-like and UI noise elements by class/id pattern
        _noise_pattern = re.compile(
            r'ad[-_s]|sidebar|banner|popup|modal|cookie|consent|newsletter|'
            r'promo|overlay|lightbox|widget|share[-_]|social[-_]|comment[-_]',
            re.IGNORECASE,
        )
        for el in soup.find_all(class_=_noise_pattern):
            el.decompose()
        for el in soup.find_all(id=_noise_pattern):
            el.decompose()

        # Remove hidden elements
        for el in soup.find_all(style=re.compile(r'display\s*:\s*none', re.IGNORECASE)):
            el.decompose()

        text = soup.get_text(separator='\n', strip=True)
        if text.strip():
            return text.strip()

    except ImportError:
        logger.warning('[TEXT EXTRACTOR] BeautifulSoup not installed — falling back to regex strip')
    except Exception as e:
        logger.debug(f'[TEXT EXTRACTOR] BeautifulSoup extraction failed: {e}')

    # 3. Regex strip (last resort)
    return _strip_html_tags(html)


def normalize_text(text: str) -> str:
    """
    Normalize extracted text: strip control characters and collapse whitespace.

    - Removes control chars (except newlines and tabs)
    - Collapses multiple spaces/tabs to single space
    - Collapses 3+ consecutive newlines to 2
    """
    if not text:
        return ''
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def detect_mime_type(file_path: str) -> str:
    """
    Detect MIME type from file extension using stdlib mimetypes.

    Falls back to 'text/plain' for unknown extensions.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'text/plain'


# ─── Format-specific extractors (internal) ───────────────────────────────────

def _extract_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber with table detection."""
    try:
        import pdfplumber
    except ImportError:
        logger.error('[TEXT EXTRACTOR] pdfplumber not installed — cannot extract PDF')
        return ''

    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ''

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        rows = []
                        for row in table:
                            cells = [str(cell or '').strip() for cell in row]
                            rows.append(' | '.join(cells))
                        page_text += '\n' + '\n'.join(rows)

                if page_text.strip():
                    pages.append(f"[Page {i + 1}]\n{page_text.strip()}")

        return '\n\n'.join(pages)

    except Exception as e:
        logger.error(f'[TEXT EXTRACTOR] PDF extraction failed: {e}')
        return ''


def _extract_docx(path: str) -> str:
    """Extract text from DOCX with paragraph and table support."""
    try:
        from docx import Document
    except ImportError:
        logger.error('[TEXT EXTRACTOR] python-docx not installed — cannot extract DOCX')
        return ''

    try:
        doc = Document(path)
        parts = []

        for element in doc.element.body:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            if tag == 'p':
                for para in doc.paragraphs:
                    if para._element == element:
                        text = para.text.strip()
                        if text:
                            if para.style and para.style.name.startswith('Heading'):
                                level = para.style.name.replace('Heading ', '').replace('Heading', '1')
                                try:
                                    level = int(level)
                                except ValueError:
                                    level = 1
                                parts.append(f"{'#' * level} {text}")
                            else:
                                parts.append(text)
                        break
            elif tag == 'tbl':
                for table in doc.tables:
                    if table._element == element:
                        rows = []
                        for row in table.rows:
                            cells = [cell.text.strip() for cell in row.cells]
                            rows.append(' | '.join(cells))
                        parts.append('\n'.join(rows))
                        break

        return '\n\n'.join(parts)

    except Exception as e:
        logger.error(f'[TEXT EXTRACTOR] DOCX extraction failed: {e}')
        return ''


def _extract_pptx(path: str) -> str:
    """Extract text from PowerPoint slides as labelled sections."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.error('[TEXT EXTRACTOR] python-pptx not installed — cannot extract PPTX')
        return ''

    try:
        prs = Presentation(path)
        slides = []

        for i, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            texts.append(text)
            if texts:
                slides.append(f"[Slide {i + 1}]\n" + '\n'.join(texts))

        return '\n\n'.join(slides)

    except Exception as e:
        logger.error(f'[TEXT EXTRACTOR] PPTX extraction failed: {e}')
        return ''


def _extract_html_file(path: str) -> str:
    """Extract text from an HTML file on disk. Delegates to extract_html()."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
        return extract_html(raw)
    except Exception as e:
        logger.error(f'[TEXT EXTRACTOR] HTML file extraction failed: {e}')
        return ''


def _extract_plain(path: str) -> str:
    """Read a plain text file with UTF-8 encoding, replacing undecodable bytes."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        logger.error(f'[TEXT EXTRACTOR] Plain text read failed: {e}')
        return ''


def _strip_html_tags(html: str) -> str:
    """
    Last-resort HTML-to-text: remove scripts/styles/tags and decode entities.

    Not as clean as trafilatura or BeautifulSoup but requires zero dependencies.
    """
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', text).strip()
