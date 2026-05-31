"""
Image Context Service — Analyze images attached to chat messages.

Uses local RapidOCR for text extraction. No external provider APIs required.

Safety invariants applied before OCR:
  - EXIF metadata stripped (removes GPS, device IDs, timestamps)
  - Dimensions normalized to max 2048px
"""

import io
import logging
import time

logger = logging.getLogger(__name__)

# Max dimension for image normalization (either side)
_MAX_DIMENSION = 2048

# Minimum OCR text length to consider 'has_text' = True
_MIN_TEXT_LENGTH = 10


def analyze(image_bytes: bytes, _mime_type: str = 'image/png') -> dict:
    """
    Analyze an image for chat context via local OCR.

    Applies safety preprocessing (EXIF strip, dimension normalization) then
    runs RapidOCR for text extraction. No external providers needed.

    Returns:
        dict with keys:
            ocr_text (str): Extracted text (empty string if none)
            has_text (bool): Whether meaningful text was found
            analysis_time_ms (int): Total analysis duration
            error (str | None): Error message if analysis failed
            vision_used (bool): True when a vision provider handled the image
    """
    start = time.time()

    result = {
        'ocr_text': '',
        'has_text': False,
        'analysis_time_ms': 0,
        'error': None,
        'vision_used': False,
    }

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img = _strip_exif(img)
        img = _normalize_dimensions(img)

        # Branch: a configured vision provider → comprehensive vision extraction
        # (NO OCR). Otherwise → local OCR only (caller appends the no-vision note).
        from services.database_service import get_shared_db_service
        from services.provider_db_service import ProviderDbService
        try:
            vision_provider = ProviderDbService(get_shared_db_service()).get_vision_provider()
        except Exception:
            vision_provider = None

        if vision_provider:
            from services import vision_service
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            config = vision_service.build_vision_config(vision_provider)
            vision_text = vision_service.send_image_with_config(
                config, buf.getvalue(), vision_service.DOCUMENT_VISION_PROMPT,
                mime_type='image/png',
            )
            result['vision_used'] = True
            result['ocr_text'] = (vision_text or '').strip()
            result['has_text'] = bool(result['ocr_text'])
            if not vision_text:
                result['error'] = 'vision extraction returned no text'
        else:
            from services.ocr_service import _extract_text
            ocr_text = _extract_text(img)
            result['ocr_text'] = ocr_text.strip() if ocr_text else ''
            result['has_text'] = len(result['ocr_text']) >= _MIN_TEXT_LENGTH

    except ImportError:
        result['error'] = 'Pillow (PIL) not installed'
        logger.warning('[IMAGE CTX] PIL not available — cannot analyze image')
    except Exception as e:
        result['error'] = str(e)
        logger.warning(f'[IMAGE CTX] Analysis failed: {e}')

    result['analysis_time_ms'] = int((time.time() - start) * 1000)
    return result


# ─── Preprocessing ───────────────────────────────────────────────────────────

def _strip_exif(img) -> object:
    """Return a copy of the PIL Image with EXIF metadata removed.

    Uses a BytesIO PNG round-trip: PNG encoder does not write EXIF by default.
    Falls back to returning the original image unchanged on any error.
    """
    try:
        from PIL import Image
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        clean = Image.open(buf)
        clean.load()
        return clean
    except Exception as e:
        logger.debug(f'[IMAGE CTX] EXIF strip failed (non-fatal): {e}')
        return img


def _normalize_dimensions(img) -> object:
    """Downscale image so neither dimension exceeds _MAX_DIMENSION."""
    try:
        w, h = img.size
        if w <= _MAX_DIMENSION and h <= _MAX_DIMENSION:
            return img
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION))
        logger.debug(f'[IMAGE CTX] Downscaled from {w}x{h} to {img.size[0]}x{img.size[1]}')
        return img
    except Exception as e:
        logger.debug(f'[IMAGE CTX] Dimension normalization failed (non-fatal): {e}')
        return img
