"""
OCR Service — Extract text from image-only PDFs and images.

Cascade:
  1. Vision LLM (Gemini, Anthropic, OpenAI) — best quality, requires API key
  2. Tesseract (pytesseract) — offline fallback, requires system install

Uses pdfplumber's .to_image() for PDF→PIL conversion (no poppler dependency).
"""

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Max pages to OCR per document (cost control)
MAX_OCR_PAGES = 20

# Hard timeout for each vision LLM OCR call (seconds).
# Prevents the document processing pipeline from hanging indefinitely on
# slow or unresponsive providers (observed 310s+ hangs without this).
_OCR_TIMEOUT = 60


def ocr_pdf(path: str, max_pages: int = MAX_OCR_PAGES) -> str:
    """
    Extract text from an image-only PDF via OCR.

    Tries Vision LLM first, falls back to tesseract.
    Returns empty string on complete failure (never raises).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error('[OCR] pdfplumber not installed')
        return ''

    try:
        images = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                try:
                    img = page.to_image(resolution=200).original  # PIL.Image
                    images.append((i + 1, img))
                except Exception as e:
                    logger.warning(f'[OCR] Failed to render page {i + 1}: {e}')

        if not images:
            return ''

        # Try Vision LLM first
        text = _ocr_with_vision_llm(images)
        if text and text.strip():
            logger.info(f'[OCR] Vision LLM extracted {len(text)} chars from {len(images)} pages')
            return text

        # Fallback to tesseract
        text = _ocr_with_tesseract(images)
        if text and text.strip():
            logger.info(f'[OCR] Tesseract extracted {len(text)} chars from {len(images)} pages')
            return text

        logger.warning(f'[OCR] All OCR methods failed for {path}')
        return ''

    except Exception as e:
        logger.error(f'[OCR] PDF OCR failed: {e}')
        return ''


def ocr_image(path: str) -> str:
    """
    Extract text from a single image file via OCR.

    Tries Vision LLM first, falls back to tesseract.
    """
    try:
        from PIL import Image
        img = Image.open(path)
        images = [(1, img)]

        text = _ocr_with_vision_llm(images)
        if text and text.strip():
            return text

        text = _ocr_with_tesseract(images)
        if text and text.strip():
            return text

        return ''
    except Exception as e:
        logger.error(f'[OCR] Image OCR failed: {e}')
        return ''


# ─── Vision LLM OCR ─────────────────────────────────────────────────────────

def _ocr_with_vision_llm(images: list) -> str:
    """Send page images to a vision-capable LLM for text extraction."""
    provider_config = _get_vision_provider()
    if not provider_config:
        logger.debug('[OCR] No vision-capable provider available')
        return ''

    platform = provider_config.get('platform', '')
    try:
        if platform == 'gemini':
            return _ocr_gemini(provider_config, images)
        elif platform == 'anthropic':
            return _ocr_anthropic(provider_config, images)
        elif platform == 'openai':
            return _ocr_openai(provider_config, images)
        else:
            logger.debug(f'[OCR] Provider {platform} does not support vision')
            return ''
    except Exception as e:
        logger.warning(f'[OCR] Vision LLM OCR failed ({platform}): {e}')
        return ''


_VISION_PLATFORMS = {'gemini', 'anthropic', 'openai'}


def _get_vision_provider() -> Optional[dict]:
    """
    Find the vision provider for OCR via the job-provider assignment system.

    Delegates to ProviderCacheService.resolve_for_job so the resolution order
    (job-specific assignment → first available vision provider) is handled centrally.
    """
    try:
        from services.provider_cache_service import ProviderCacheService
        return ProviderCacheService.resolve_for_job('document-ocr', platforms=_VISION_PLATFORMS)
    except Exception as e:
        logger.debug(f'[OCR] Failed to resolve vision provider: {e}')
        return None


_OCR_PROMPT = (
    "Extract ALL text from this document image. "
    "Preserve the original structure (paragraphs, lists, tables). "
    "Output only the extracted text, nothing else. "
    "If no text is visible, respond with exactly: NO_TEXT"
)


def _pil_to_base64(img, format='PNG') -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _pil_to_bytes(img, format='PNG') -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def _ocr_gemini(config: dict, images: list) -> str:
    """OCR via Google Gemini vision."""
    from google import genai

    from services.llm_service import _resolve_api_key
    api_key = _resolve_api_key(config)
    client = genai.Client(api_key=api_key, http_options={"timeout": _OCR_TIMEOUT * 1000})
    model = config.get('model')

    pages = []
    for page_num, img in images:
        img_bytes = _pil_to_bytes(img)
        response = client.models.generate_content(
            model=model,
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                _OCR_PROMPT,
            ],
        )
        text = response.text if response.text else ''
        if text.strip() and text.strip() != 'NO_TEXT':
            pages.append(f'[Page {page_num}]\n{text.strip()}')

    return '\n\n'.join(pages)


def _ocr_anthropic(config: dict, images: list) -> str:
    """OCR via Anthropic Claude vision."""
    import anthropic

    from services.llm_service import _resolve_api_key
    api_key = _resolve_api_key(config)
    client = anthropic.Anthropic(api_key=api_key)
    model = config.get('model')

    pages = []
    for page_num, img in images:
        b64 = _pil_to_base64(img)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            timeout=_OCR_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                    {'type': 'text', 'text': _OCR_PROMPT},
                ],
            }],
        )
        text = response.content[0].text if response.content else ''
        if text.strip() and text.strip() != 'NO_TEXT':
            pages.append(f'[Page {page_num}]\n{text.strip()}')

    return '\n\n'.join(pages)


def _ocr_openai(config: dict, images: list) -> str:
    """OCR via OpenAI GPT-4 vision."""
    import openai

    from services.llm_service import _resolve_api_key
    api_key = _resolve_api_key(config)
    client = openai.OpenAI(api_key=api_key)
    model = config.get('model')

    pages = []
    for page_num, img in images:
        b64 = _pil_to_base64(img)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            timeout=_OCR_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                    {'type': 'text', 'text': _OCR_PROMPT},
                ],
            }],
        )
        text = response.choices[0].message.content if response.choices else ''
        if text.strip() and text.strip() != 'NO_TEXT':
            pages.append(f'[Page {page_num}]\n{text.strip()}')

    return '\n\n'.join(pages)


# ─── Tesseract OCR ──────────────────────────────────────────────────────────

def _ocr_with_tesseract(images: list) -> str:
    """OCR via pytesseract (offline fallback)."""
    try:
        import pytesseract
    except ImportError:
        logger.debug('[OCR] pytesseract not installed — skipping tesseract fallback')
        return ''

    pages = []
    for page_num, img in images:
        try:
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                pages.append(f'[Page {page_num}]\n{text.strip()}')
        except Exception as e:
            logger.warning(f'[OCR] Tesseract failed on page {page_num}: {e}')

    return '\n\n'.join(pages)
