"""OCR Service — RapidOCR (rapidocr_onnxruntime) text extraction.

Single private entry point: `_extract_text(img)`. Callers:
- services.image_context_service
- tools.browser.browser
"""

import threading

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
