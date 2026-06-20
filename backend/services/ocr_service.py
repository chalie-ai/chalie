"""OCR Service — RapidOCR (rapidocr_onnxruntime) text extraction.

Single private entry point: `_extract_text(img)`. Callers:
- services.image_context_service
- tools.browser.browser
"""

import threading
from typing import Callable, cast

_engine = None
_engine_lock = threading.Lock()


def _get_engine() -> object:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _engine = RapidOCR()
    return _engine


def _extract_text(img: object) -> str:
    result, _ = cast(Callable[[object], tuple[list[list[object]], object]], _get_engine())(img)
    if not result:
        return ''
    return '\n'.join(cast(str, region[1]) for region in result)
