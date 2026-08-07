import base64
import logging
from typing import Dict, Optional, cast

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

logger = logging.getLogger(__name__)

_VISION_SYSTEM = (
    "You are a precise vision assistant. Follow the user's instructions exactly."
)

#: Output ceiling for the candidate-provider probe. The expected reply is a small
#: JSON object describing a test image; this leaves ample room for it without
#: needing a context window the candidate has not been pinned to yet.
_PROBE_MAX_TOKENS = 2048


class VisionService:
    """Vision service — send images to provider clients."""

    @staticmethod
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

    @staticmethod
    def send_image_with_config(config: Dict[str, object], image_bytes: bytes,
                               prompt: str, mime_type: str = 'image/png') -> Optional[str]:
        """Send one text+image message to an explicit provider config; return text.

        Builds a ProviderApiRequest and calls the thin client directly: this probes a
        CANDIDATE provider (not the live-configured one), so it cannot route through
        ProviderService.send, which resolves the configured provider. The call is still
        bounded — every client enforces PROVIDER_CALL_TIMEOUT_S at its HTTP boundary.
        Returns None on any failure.

        ``max_tokens`` is explicit because this is the one send that has no context
        window to size against: a candidate has no ``providers.context_window`` to
        pin, and the reply being asked for is one short JSON object either way.
        Every other send derives its budget from that column via
        ``ProviderService.send``.
        """
        try:
            from services.provider_api import ProviderApiRequest  # noqa: PLC0415
            from services.llm_clients.factory import Factory  # noqa: PLC0415

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
                max_tokens=_PROBE_MAX_TOKENS,
            )
            client = Factory.build_client(config)
            response = client.send(dto)
            text = (response.text or '').strip()
            return text or None
        except Exception as exc:
            logger.warning("[Vision] send_image_with_config failed: %s", exc)
            return None
