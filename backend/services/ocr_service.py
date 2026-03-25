"""
OCR Service — Extract text from image-only PDFs and images.

Single engine: RapidOCR (rapidocr_onnxruntime). No provider APIs, no Tesseract.
Uses pdfplumber for PDF→PIL conversion.
"""

import logging
import threading

logger = logging.getLogger(__name__)

MAX_OCR_PAGES = 20

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Lazy-init and cache the RapidOCR engine (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _engine = RapidOCR()
    return _engine


def _extract_text(img) -> str:
    """Run RapidOCR on a single image, return concatenated text in reading order."""
    result, _ = _get_engine()(img)
    if not result:
        return ''
    return '\n'.join(region[1] for region in result)


def ocr_pdf(path: str, max_pages: int = MAX_OCR_PAGES) -> str:
    """
    Extract text from an image-only PDF via OCR.

    Returns empty string on failure (never raises).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error('[OCR] pdfplumber not installed')
        return ''

    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                try:
                    img = page.to_image(resolution=200).original
                    text = _extract_text(img)
                    if text.strip():
                        pages.append(f'[Page {i + 1}]\n{text.strip()}')
                except Exception as e:
                    logger.warning(f'[OCR] Failed on page {i + 1}: {e}')

        combined = '\n\n'.join(pages)
        if combined:
            logger.info(f'[OCR] Extracted {len(combined)} chars from PDF ({len(pages)} pages)')
        return combined

    except Exception as e:
        logger.error(f'[OCR] PDF OCR failed: {e}')
        return ''


def ocr_image(path: str) -> str:
    """
    Extract text from a single image file via OCR.

    Returns empty string on failure (never raises).
    """
    try:
        text = _extract_text(path)
        if text.strip():
            logger.info(f'[OCR] Extracted {len(text)} chars from image')
        return text
    except Exception as e:
        logger.error(f'[OCR] Image OCR failed: {e}')
        return ''
