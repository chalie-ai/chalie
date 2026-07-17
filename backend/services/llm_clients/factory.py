"""
Provider client factory — dispatches a config dict to the correct thin client.

Replaces the create_llm_service / _build_service pair from llm_service.py.
No FallbackLLMService, no LoggingLLMService — those concerns live in
ProviderService (telemetry) and have been deleted (fallback is dead code).

Consumed by: services.provider_service (send / _resolve).
"""

from __future__ import annotations

from contracts.provider_client import ProviderClient


def _build_ollama(config: dict[str, object]) -> ProviderClient:
    if not config.get('host'):
        raise ValueError(
            "Ollama provider requires 'host' field (e.g., 'http://localhost:11434')"
        )
    from services.llm_clients.ollama import OllamaClient  # noqa: PLC0415
    return OllamaClient(config)


def _build_anthropic(config: dict[str, object]) -> ProviderClient:
    if not config.get('api_key'):
        raise ValueError("Anthropic provider requires 'api_key' field")
    from services.llm_clients.anthropic import AnthropicClient  # noqa: PLC0415
    return AnthropicClient(config)


def _build_openai(config: dict[str, object], platform: object) -> ProviderClient:
    if not config.get('api_key'):
        raise ValueError(f"{platform} provider requires 'api_key' field")
    if platform == 'openai_compatible' and not config.get('host'):
        raise ValueError(
            "openai_compatible provider requires 'host' field "
            "(base URL, e.g. 'https://api.minimax.io/v1')"
        )
    from services.llm_clients.openai import OpenAIClient  # noqa: PLC0415
    return OpenAIClient(config)


def _build_gemini(config: dict[str, object]) -> ProviderClient:
    if not config.get('api_key'):
        raise ValueError("Gemini provider requires 'api_key' field")
    from services.llm_clients.gemini import GeminiClient  # noqa: PLC0415
    return GeminiClient(config)


def _build_codex_cli(config: dict[str, object]) -> ProviderClient:
    from services.llm_clients.codex_cli import CodexCliClient  # noqa: PLC0415
    return CodexCliClient(config)


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

    if platform == 'ollama':
        return _build_ollama(config)

    if platform == 'anthropic':
        return _build_anthropic(config)

    if platform in ('openai', 'openai_compatible'):
        return _build_openai(config, platform)

    if platform == 'gemini':
        return _build_gemini(config)

    if platform == 'codex_cli':
        return _build_codex_cli(config)

    raise ValueError(f"Unknown platform: {platform}")
