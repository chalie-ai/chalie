"""
Provider client factory — dispatches a config dict to the correct thin client.

Replaces the create_llm_service / _build_service pair from llm_service.py.
No FallbackLLMService, no LoggingLLMService — those concerns live in
Providers (telemetry) and have been deleted (fallback is dead code).

Consumed by: services.providers (_resolve), services.provider_token_limits.
"""

from __future__ import annotations

from services.llm_clients.base import ProviderClient


def build_client(config: dict) -> ProviderClient:
    """Return a ProviderClient for the given provider config dict."""
    platform = config.get('platform')
    if not platform:
        raise ValueError(
            "LLM config missing 'platform'. No provider configured — "
            "add one via POST /api/providers"
        )

    model = config.get('model')
    if not model:
        raise ValueError(
            "LLM config missing 'model'. Configure it via POST /api/providers"
        )

    if platform == 'ollama':
        if not config.get('host'):
            raise ValueError(
                "Ollama provider requires 'host' field (e.g., 'http://localhost:11434')"
            )
        from services.llm_clients.ollama import OllamaClient  # noqa: PLC0415
        return OllamaClient(config)

    if platform == 'anthropic':
        if not config.get('api_key'):
            raise ValueError("Anthropic provider requires 'api_key' field")
        from services.llm_clients.anthropic import AnthropicClient  # noqa: PLC0415
        return AnthropicClient(config)

    if platform in ('openai', 'openai_compatible'):
        if not config.get('api_key'):
            raise ValueError(f"{platform} provider requires 'api_key' field")
        if platform == 'openai_compatible' and not config.get('host'):
            raise ValueError(
                "openai_compatible provider requires 'host' field "
                "(base URL, e.g. 'https://api.minimax.io/v1')"
            )
        from services.llm_clients.openai import OpenAIClient  # noqa: PLC0415
        return OpenAIClient(config)

    if platform == 'gemini':
        if not config.get('api_key'):
            raise ValueError("Gemini provider requires 'api_key' field")
        from services.llm_clients.gemini import GeminiClient  # noqa: PLC0415
        return GeminiClient(config)

    raise ValueError(f"Unknown platform: {platform}")
