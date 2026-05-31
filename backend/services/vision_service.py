"""Vision service — one-shot multimodal (text + image → text) calls.

Reuses the per-platform converters in llm_service / ollama_service by attaching
an ``image`` key to a normalized message and sending it via ``send_messages``.
The converters translate the key into each provider's native image block; the
text send-path is unchanged for image-free messages. Fully defensive: every
public function returns ``None`` on failure so callers never raise.
"""

import base64
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_VISION_SYSTEM = (
    "You are a precise vision assistant. Follow the user's instructions exactly."
)

# Comprehensive prompt for document-image understanding (NOT the probe).
DOCUMENT_VISION_PROMPT = (
    "Describe this image comprehensively for document understanding. "
    "Transcribe ALL visible text exactly as written. Describe any diagrams, "
    "charts, tables, UI elements, photographs, people, objects, and the overall "
    "layout. Be thorough and literal — do not summarise away detail."
)


def build_vision_config(provider: Dict[str, Any]) -> Dict[str, Any]:
    """Build an llm_service config dict from a provider row for a vision call.

    ``timeout`` bounds Anthropic/OpenAI calls (GeminiService ignores it).
    ``max_tokens`` is intentionally omitted — the services use their own ceiling.
    """
    config: Dict[str, Any] = {
        'platform': provider.get('platform', ''),
        'model': provider.get('model', ''),
        'timeout': 60,
    }
    if provider.get('api_key'):
        config['api_key'] = provider['api_key']
    if provider.get('host'):
        config['host'] = provider['host']
    return config


def send_image_with_config(config: Dict[str, Any], image_bytes: bytes,
                           prompt: str, mime_type: str = 'image/png') -> Optional[str]:
    """Send one text+image message to an explicit provider config; return text.

    Returns None on any failure (build error, network, empty response).
    """
    try:
        from services.llm_service import create_llm_service
        b64 = base64.b64encode(image_bytes).decode('ascii')
        message = {
            'role': 'user',
            'content': prompt,
            'image': {'data': b64, 'mime_type': mime_type},
        }
        llm = create_llm_service(config)
        response = llm.send_messages(_VISION_SYSTEM, [message])
        text = (response.text or '').strip()
        return text or None
    except Exception as exc:
        logger.warning("[Vision] send_image_with_config failed: %s", exc)
        return None


def describe_image(image_bytes: bytes, prompt: str = DOCUMENT_VISION_PROMPT,
                   mime_type: str = 'image/png') -> Optional[str]:
    """Describe an image using the configured vision provider, or None.

    Resolves the provider via ProviderDbService.get_vision_provider().
    """
    try:
        from services.database_service import get_shared_db_service
        from services.provider_db_service import ProviderDbService
        provider = ProviderDbService(get_shared_db_service()).get_vision_provider()
        if not provider:
            return None
        config = build_vision_config(provider)
        return send_image_with_config(config, image_bytes, prompt, mime_type)
    except Exception as exc:
        logger.warning("[Vision] describe_image failed: %s", exc)
        return None
