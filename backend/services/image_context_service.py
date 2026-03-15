"""
Image Context Service — Analyze images attached to chat messages.

Dual-mode analysis:
  1. Visual description — What is in the image?
  2. OCR text extraction — Is there readable text?

Safety invariants applied before any LLM call:
  - EXIF metadata stripped (removes GPS, device IDs, timestamps)
  - Dimensions normalized to max 2048px (reduces API cost and latency)
  - SHA-256 hash deduplication (same image → same image_id in same session)

Provider job: 'chat-vision' (falls back to 'document-ocr' if unassigned).
"""

import base64
import hashlib
import io
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Vision platforms that support multimodal input
_VISION_PLATFORMS = {'gemini', 'anthropic', 'openai'}

# Max dimension for image normalization (either side)
_MAX_DIMENSION = 2048

# Per-call timeout (seconds) — shorter than document OCR's 60s; chat is interactive
_VISION_TIMEOUT = 30

_DESCRIBE_PROMPT = (
    "Describe what you see in this image concisely and specifically. "
    "Include: objects, text, people, setting, spatial layout, and any notable details. "
    "Be factual and precise. Respond in 2–4 sentences. "
    "If the image contains mostly text (a document, receipt, sign, etc.), "
    "focus on what the text says and what kind of document it appears to be."
)

_OCR_PROMPT = (
    "Extract ALL text visible in this image exactly as it appears. "
    "Preserve the original structure (paragraphs, lists, tables, columns). "
    "Output only the extracted text, nothing else. "
    "If no text is visible, respond with exactly: NO_TEXT"
)

# Minimum OCR text length to consider 'has_text' = True
_MIN_TEXT_LENGTH = 10


def analyze(image_bytes: bytes, mime_type: str = 'image/png') -> dict:
    """
    Analyze an image for chat context.

    Applies safety preprocessing (EXIF strip, dimension normalization) then
    calls the configured vision provider for description + OCR.

    Returns:
        dict with keys:
            description (str): Visual description of the image
            ocr_text (str): Extracted text (empty string if none)
            has_text (bool): Whether meaningful text was found
            analysis_time_ms (int): Total analysis duration
            error (str | None): Error message if analysis failed
    """
    start = time.time()

    result = {
        'description': '',
        'ocr_text': '',
        'has_text': False,
        'analysis_time_ms': 0,
        'error': None,
    }

    try:
        from PIL import Image

        # Load, strip EXIF, normalize dimensions
        img = Image.open(io.BytesIO(image_bytes))
        img = _strip_exif(img)
        img = _normalize_dimensions(img)

        provider_config = _get_vision_provider()
        if not provider_config:
            result['error'] = 'No vision-capable provider configured'
            result['analysis_time_ms'] = int((time.time() - start) * 1000)
            return result

        platform = provider_config.get('platform', '')

        if platform == 'gemini':
            description, ocr_text = _analyze_gemini(provider_config, img)
        elif platform == 'anthropic':
            description, ocr_text = _analyze_anthropic(provider_config, img)
        elif platform == 'openai':
            description, ocr_text = _analyze_openai(provider_config, img)
        else:
            result['error'] = f'Provider {platform} does not support vision'
            result['analysis_time_ms'] = int((time.time() - start) * 1000)
            return result

        result['description'] = description
        result['ocr_text'] = ocr_text if (ocr_text and ocr_text.strip() != 'NO_TEXT') else ''
        result['has_text'] = len(result['ocr_text']) >= _MIN_TEXT_LENGTH

    except ImportError:
        result['error'] = 'Pillow (PIL) not installed'
        logger.warning('[IMAGE CTX] PIL not available — cannot analyze image')
    except Exception as e:
        result['error'] = str(e)
        logger.warning(f'[IMAGE CTX] Analysis failed: {e}')

    result['analysis_time_ms'] = int((time.time() - start) * 1000)
    return result


def has_vision_provider() -> bool:
    """Check whether a vision-capable provider is configured for image analysis.

    Returns:
        ``True`` if at least one Gemini, Anthropic, or OpenAI provider is
        assigned to the ``chat-vision`` or ``document-ocr`` job.
    """
    return _get_vision_provider() is not None


def compute_hash(image_bytes: bytes) -> str:
    """Compute the SHA-256 hash of image bytes for within-session deduplication.

    Args:
        image_bytes: Raw image bytes to hash.

    Returns:
        Lowercase hex-encoded SHA-256 digest string.
    """
    return hashlib.sha256(image_bytes).hexdigest()


# ─── Preprocessing ───────────────────────────────────────────────────────────

def _strip_exif(img) -> object:
    """Return a copy of the PIL Image with EXIF metadata removed.

    Strips GPS coordinates, device identifiers, timestamps, and all other
    EXIF tags that phones embed automatically.

    Implementation uses a BytesIO PNG round-trip: the image is saved to an
    in-memory buffer as PNG (PIL's PNG encoder does not write EXIF by default)
    and immediately re-opened.  This is dramatically more memory-efficient than
    the previous ``list(img.getdata())`` approach, which materialised the
    entire pixel array as a Python list — up to ~470 MB for a 2048×2048 RGBA
    image.

    Args:
        img: PIL Image object whose EXIF/metadata should be stripped.

    Returns:
        A new PIL Image with identical pixel content and no embedded metadata.
        Falls back to returning the original image unchanged on any error
        (non-fatal — analysis can still proceed with residual metadata).
    """
    try:
        from PIL import Image
        buf = io.BytesIO()
        # PNG format never carries EXIF in PIL's default encoder, so a
        # save+reload cycle produces a completely metadata-free image.
        img.save(buf, format='PNG')
        buf.seek(0)
        clean = Image.open(buf)
        # Force pixel data to load now so the BytesIO buffer can be
        # garbage-collected rather than kept alive by lazy loading.
        clean.load()
        return clean
    except Exception as e:
        logger.debug(f'[IMAGE CTX] EXIF strip failed (non-fatal): {e}')
        return img


def _normalize_dimensions(img) -> object:
    """
    Downscale image so neither dimension exceeds _MAX_DIMENSION.

    Uses LANCZOS resampling for quality. No-op if already within bounds.
    """
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


# ─── Provider Resolution ──────────────────────────────────────────────────────

def _get_vision_provider() -> Optional[dict]:
    """
    Resolve the vision provider for chat-image analysis.

    Resolution order:
      1. Job-specific assignment for 'chat-vision'
      2. Job-specific assignment for 'document-ocr' (fallback — no extra config needed)
      3. First available vision-capable provider
    """
    try:
        from services.provider_cache_service import ProviderCacheService
        # Try chat-vision job first
        config = ProviderCacheService.resolve_for_job('chat-vision', platforms=_VISION_PLATFORMS)
        if config:
            return config
        # Fall back to document-ocr job (same vision providers, different default model)
        return ProviderCacheService.resolve_for_job('document-ocr', platforms=_VISION_PLATFORMS)
    except Exception as e:
        logger.debug(f'[IMAGE CTX] Provider resolution failed: {e}')
        return None


# ─── Image Helpers ────────────────────────────────────────────────────────────

def _img_to_base64(img, format='PNG') -> str:
    """Encode a PIL Image as a base64 string for API payloads.

    Args:
        img: PIL Image object to encode.
        format: Target image format (default ``'PNG'``).

    Returns:
        Base64-encoded string of the image bytes.
    """
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _img_to_bytes(img, format='PNG') -> bytes:
    """Serialize a PIL Image to raw bytes.

    Args:
        img: PIL Image object to serialize.
        format: Target image format (default ``'PNG'``).

    Returns:
        Bytes object containing the encoded image data.
    """
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# ─── Provider Implementations ─────────────────────────────────────────────────

def _analyze_gemini(config: dict, img) -> tuple:
    """Analyze an image using Google Gemini for visual description and OCR.

    Args:
        config: Provider config dict with ``model`` and API credential keys.
        img: Pre-processed PIL Image (EXIF-stripped, dimension-normalized).

    Returns:
        Tuple of ``(description, ocr_text)`` strings.  Each may be empty on
        provider failure.
    """
    from google import genai
    from services.llm_service import _resolve_api_key

    api_key = _resolve_api_key(config)
    client = genai.Client(api_key=api_key, http_options={"timeout": _VISION_TIMEOUT * 1000})
    model = config.get('model')
    img_bytes = _img_to_bytes(img)

    try:
        desc_response = client.models.generate_content(
            model=model,
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                _DESCRIBE_PROMPT,
            ],
        )
        description = (desc_response.text or '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] Gemini describe failed: {e}')
        description = ''

    try:
        ocr_response = client.models.generate_content(
            model=model,
            contents=[
                genai.types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
                _OCR_PROMPT,
            ],
        )
        ocr_text = (ocr_response.text or '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] Gemini OCR failed: {e}')
        ocr_text = ''

    return description, ocr_text


def _analyze_anthropic(config: dict, img) -> tuple:
    """Analyze an image using Anthropic Claude for visual description and OCR.

    Args:
        config: Provider config dict with ``model`` and API credential keys.
        img: Pre-processed PIL Image (EXIF-stripped, dimension-normalized).

    Returns:
        Tuple of ``(description, ocr_text)`` strings.  Each may be empty on
        provider failure.
    """
    import anthropic
    from services.llm_service import _resolve_api_key

    api_key = _resolve_api_key(config)
    client = anthropic.Anthropic(api_key=api_key)
    model = config.get('model')
    b64 = _img_to_base64(img)
    image_block = {
        'type': 'image',
        'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64},
    }

    try:
        desc_response = client.messages.create(
            model=model,
            max_tokens=512,
            timeout=_VISION_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [image_block, {'type': 'text', 'text': _DESCRIBE_PROMPT}],
            }],
        )
        description = (desc_response.content[0].text if desc_response.content else '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] Anthropic describe failed: {e}')
        description = ''

    try:
        ocr_response = client.messages.create(
            model=model,
            max_tokens=2048,
            timeout=_VISION_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [image_block, {'type': 'text', 'text': _OCR_PROMPT}],
            }],
        )
        ocr_text = (ocr_response.content[0].text if ocr_response.content else '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] Anthropic OCR failed: {e}')
        ocr_text = ''

    return description, ocr_text


def _analyze_openai(config: dict, img) -> tuple:
    """Analyze an image using OpenAI GPT-4 Vision for visual description and OCR.

    Args:
        config: Provider config dict with ``model`` and API credential keys.
        img: Pre-processed PIL Image (EXIF-stripped, dimension-normalized).

    Returns:
        Tuple of ``(description, ocr_text)`` strings.  Each may be empty on
        provider failure.
    """
    import openai
    from services.llm_service import _resolve_api_key

    api_key = _resolve_api_key(config)
    client = openai.OpenAI(api_key=api_key)
    model = config.get('model')
    b64 = _img_to_base64(img)
    image_block = {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}

    try:
        desc_response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            timeout=_VISION_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [image_block, {'type': 'text', 'text': _DESCRIBE_PROMPT}],
            }],
        )
        description = (desc_response.choices[0].message.content if desc_response.choices else '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] OpenAI describe failed: {e}')
        description = ''

    try:
        ocr_response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            timeout=_VISION_TIMEOUT,
            messages=[{
                'role': 'user',
                'content': [image_block, {'type': 'text', 'text': _OCR_PROMPT}],
            }],
        )
        ocr_text = (ocr_response.choices[0].message.content if ocr_response.choices else '').strip()
    except Exception as e:
        logger.warning(f'[IMAGE CTX] OpenAI OCR failed: {e}')
        ocr_text = ''

    return description, ocr_text
