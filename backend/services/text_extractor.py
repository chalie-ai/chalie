"""
Text Extractor — Shared text extraction library for files and HTML strings.

Pure functions — no database, no MemoryStore, no Chalie services.
Used by DocumentProcessingService (file pipeline) and the `read` innate skill
(URL fetch + local file read).

Supported formats:
  - PDF         (pdfplumber)
  - DOCX        (python-docx)
  - PPTX        (python-pptx)
  - HTML        (trafilatura)
  - Plain text  (direct read)
  - Markdown    (direct read)
  - Any text/*  (direct read)
  - Image       (PNG/JPEG/WEBP/GIF/BMP/TIFF via RapidOCR)

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

    Most formats return an empty string on failure. The exception is image
    extraction, which propagates provider errors when a vision provider is
    configured but failing (TKT-838) — fail loud, never swallowed.
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
        'image/png': _extract_image,
        'image/jpeg': _extract_image,
        'image/webp': _extract_image,
        'image/gif': _extract_image,
        'image/bmp': _extract_image,
        'image/tiff': _extract_image,
    }

    extractor = extractors.get(mime_type)
    if extractor:
        return extractor(file_path)

    # Fallback: any image/* type (e.g. image/heic) via OCR
    if mime_type and mime_type.startswith('image/'):
        return _extract_image(file_path)

    # Fallback: any text/* type (code files, CSV, etc.)
    if mime_type and mime_type.startswith('text/'):
        return _extract_plain(file_path)

    logger.warning(f"[TEXT EXTRACTOR] Unsupported mime type '{mime_type}' — attempting plain read")
    return _extract_plain(file_path)


def extract_html(html: str, url: str = None) -> str:
    """
    Extract clean, readable text from an HTML string via trafilatura.

    Returns empty string if extraction fails or produces no content.
    """
    if not html or not html.strip():
        return ''

    try:
        import trafilatura
        content = trafilatura.extract(html, url=url, include_comments=False, include_links=True)
        if content and content.strip():
            return content.strip()
    except Exception as e:
        logger.debug(f'[TEXT EXTRACTOR] trafilatura failed: {e}')

    return ''


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


def _extract_image(path: str) -> str:
    """Image 'text' = a rich vision description (OCR fallback when no vision
    provider is configured). Routed through the shared describe_image() core so an
    uploaded image is embedded + FTS5-indexed by the normal document pipeline and
    becomes searchable by its visual content. A configured-but-failing vision
    provider raises (never swallowed) — the upload pipeline decides how to surface
    that (TKT-838). Only the description is returned; the user-facing no-vision note
    is intentionally NOT indexed (it is index noise)."""
    import mimetypes  # noqa: PLC0415
    from abilities.vision import RICH_INDEX_PROMPT, describe_image  # noqa: PLC0415
    from services.processor_config import ProcessorConfig  # noqa: PLC0415

    mime_type = mimetypes.guess_type(path)[0] or 'image/png'
    out = describe_image(path, mime_type, RICH_INDEX_PROMPT,
                         policy_channel=ProcessorConfig.POLICY_CHANNEL.CHAT)
    return out['description'] or ''
