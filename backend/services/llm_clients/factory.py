"""
Provider client factory — dispatches a config dict to the correct thin client.

Replaces the create_llm_service / _build_service pair from llm_service.py.
No FallbackLLMService, no LoggingLLMService — those concerns live in
ProviderService (telemetry) and have been deleted (fallback is dead code).

Dispatch and validation are both derived from the registry: the platform names
a class, and that class states whether a key and a host are required. There is
no per-platform branch here, and adding a provider never touches this file.

Consumed by: services.provider_service (send / _resolve).
"""

from __future__ import annotations

from contracts.provider_client import ProviderClient
from services.llm_clients.registry import PROVIDERS_BY_PLATFORM


def build_client(config: dict[str, object]) -> ProviderClient:
    """Return a ProviderClient for the given provider config dict."""
    platform = config.get('platform')
    if not platform:
        raise ValueError(
            "LLM config missing 'platform'. No provider configured — "
            "add one via POST /api/providers/-1"
        )

    model = config.get('model')
    if not model:
        raise ValueError(
            "LLM config missing 'model'. Configure it via the providers API"
        )

    client_class = PROVIDERS_BY_PLATFORM.get(str(platform))
    if client_class is None:
        raise ValueError(f"Unknown platform: {platform}")

    if client_class.REQUIRES_KEY and not config.get('api_key'):
        raise ValueError(f"{platform} provider requires 'api_key' field")

    if client_class.REQUIRES_HOST and not config.get('host'):
        raise ValueError(
            f"{platform} provider requires 'host' field "
            f"(e.g. '{client_class.DEFAULT_BASE_URL}')"
        )

    return client_class(config)
