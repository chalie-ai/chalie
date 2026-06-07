"""Vision service — one-shot multimodal (text + image → text) calls.

Builds a ProviderApiRequest with type=VISION and calls through the factory
directly (no mp / no Providers facade) since this is a pure infrastructure probe
with no transcript or act-loop context. Fully defensive: every public function
returns None on failure so callers never raise.

Depends on: services.provider_api (ProviderApiRequest, ProviderType),
            services.llm_clients.factory (build_client).
Consumed by: services.vision_probe (capability probe).
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
    """Build a client config dict from a provider row for a vision call.

    ``timeout`` bounds Anthropic/OpenAI calls (GeminiClient ignores it).
    ``max_tokens`` is intentionally omitted — the DTO formula is used.
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

    Builds a ProviderApiRequest and calls the thin client directly (no Providers
    facade — this is a probe path with no mp context). Returns None on any failure.
    """
    try:
        from services.provider_api import ProviderApiRequest, ProviderType, ThinkingLevel  # noqa: PLC0415
        from services.llm_clients.factory import build_client  # noqa: PLC0415

        b64 = base64.b64encode(image_bytes).decode('ascii')
        message = {
            'role': 'user',
            'content': prompt,
            'image': {'data': b64, 'mime_type': mime_type},
        }
        dto = ProviderApiRequest(
            system=_VISION_SYSTEM,
            messages=[message],
            type=ProviderType.VISION,
            thinking_mode=ThinkingLevel.LOW,
            cache_prefix=False,
        )
        client = build_client(config)
        response = client.send(dto)
        text = (response.text or '').strip()
        return text or None
    except Exception as exc:
        logger.warning("[Vision] send_image_with_config failed: %s", exc)
        return None
