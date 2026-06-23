"""Image Context Service — analyze images attached to chat messages via local RapidOCR.

Safety invariants applied before OCR: EXIF stripped (removes GPS, device
IDs, timestamps); dimensions normalised to max 2048 px.
"""

import io
import logging
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# Max dimension for image normalization (either side)
_MAX_DIMENSION = 2048

# Minimum OCR text length to consider 'has_text' = True
_MIN_TEXT_LENGTH = 10


def analyze(image_bytes: bytes, _mime_type: str = 'image/png') -> dict[str, object]:
    """Applies safety preprocessing (EXIF strip, dimension normalisation) then
    runs RapidOCR for text extraction. No external providers needed.

    ``vision_used`` is always False — ``analyze`` is OCR-only; the vision
    path lives in the vision tool / describe_image() core, for which this
    is the no-vision-provider fallback.
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

        img = cast("PILImage", Image.open(io.BytesIO(image_bytes)))
        img = _strip_exif(img)
        img = _normalize_dimensions(img)

        from services.ocr_service import _extract_text
        ocr_text = _extract_text(img)
        result['ocr_text'] = ocr_text.strip() if ocr_text else ''
        result['has_text'] = len(cast(str, result['ocr_text'])) >= _MIN_TEXT_LENGTH

    except ImportError:
        result['error'] = 'Pillow (PIL) not installed'
        logger.warning('[IMAGE CTX] PIL not available — cannot analyze image')
    except Exception as e:
        result['error'] = str(e)
        logger.warning(f'[IMAGE CTX] Analysis failed: {e}')

    result['analysis_time_ms'] = int((time.time() - start) * 1000)
    return result


# ─── Preprocessing ───────────────────────────────────────────────────────────

def _strip_exif(img: "PILImage") -> "PILImage":
    """Uses a BytesIO PNG round-trip: PNG encoder does not write EXIF by
    default. Falls back to the original image unchanged on any error."""
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


def _normalize_dimensions(img: "PILImage") -> "PILImage":
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
