import base64
import logging
from typing import Optional, Dict, cast

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

logger = logging.getLogger(__name__)

_VISION_SYSTEM = (
    "You are a precise vision assistant. Follow the user's instructions exactly."
)



def build_vision_config(provider: Dict[str, object]) -> Dict[str, object]:
    config: Dict[str, object] = {
        'platform': provider.get('platform', ''),
        'model': provider.get('model', ''),
    }
    if provider.get('api_key'):
        config['api_key'] = provider['api_key']
    if provider.get('host'):
        config['host'] = provider['host']
    return config


def send_image_with_config(config: Dict[str, object], image_bytes: bytes,
                           prompt: str, mime_type: str = 'image/png') -> Optional[str]:
    """Send one text+image message to an explicit provider config; return text.

    Builds a ProviderApiRequest and calls the thin client directly: this probes a
    CANDIDATE provider (not the live-configured one), so it cannot route through
    ProviderService.send, which resolves the configured provider. The call is still
    bounded — every client enforces PROVIDER_CALL_TIMEOUT_S at its HTTP boundary.
    Returns None on any failure.
    """
    try:
        from services.provider_api import ProviderApiRequest  # noqa: PLC0415
        from services.llm_clients.factory import build_client  # noqa: PLC0415

        b64 = base64.b64encode(image_bytes).decode('ascii')
        message = {
            'role': 'user',
            'content': prompt,
            'image': {'data': b64, 'mime_type': mime_type},
        }
        dto = ProviderApiRequest(
            system=_VISION_SYSTEM,
            messages=[cast(dict[str, object], message)],
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
