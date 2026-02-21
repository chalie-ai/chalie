"""
LLM Service Factory — creates the right LLM client based on platform.

Usage:
    from services.llm_service import create_llm_service
    llm = create_llm_service(config)
    response = llm.send_message(system_prompt, user_message)
    text = response.text
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_api_key(config: dict) -> str:
    """
    Resolve API key from provider config (from database).

    Args:
        config: Provider config dict that must contain 'api_key' field

    Returns:
        API key string from database

    Raises:
        ValueError if api_key is not found in config
    """
    api_key = config.get('api_key')

    if not api_key:
        raise ValueError(
            "API key not found in provider configuration. "
            "Store the API key in the database via POST /providers or update via PUT /providers/<id>"
        )

    return api_key


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    latency_ms: Optional[int] = None


def _call_with_retry(fn, max_retries=2, backoff=1.0):
    """Retry fn() up to max_retries times with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


class FallbackLLMService:
    """Wraps a primary + fallback service. On primary failure, invokes fallback."""

    def __init__(self, primary, fallback):
        self._primary = primary
        self._fallback = fallback

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        try:
            return self._primary.send_message(system_prompt, user_message, stream=stream)
        except Exception as e:
            logger.warning(f"Primary LLM failed, using fallback: {e}")
            return self._fallback.send_message(system_prompt, user_message, stream=stream)


def _build_service(config: dict):
    """Build a single LLM service from a config dict."""
    platform = config.get('platform')
    if not platform:
        raise ValueError(
            "LLM config missing 'platform'. No provider configured — add one via POST /api/providers"
        )

    model = config.get('model')
    if not model:
        raise ValueError(
            "LLM config missing 'model'. Configure it via POST /api/providers"
        )

    if platform == 'ollama':
        host = config.get('host')
        if not host:
            raise ValueError(
                "Ollama provider requires 'host' field (e.g., 'http://localhost:11434')"
            )
        from services.ollama_service import OllamaService
        return OllamaService(config)
    elif platform == 'anthropic':
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError(
                "Anthropic provider requires 'api_key' field"
            )
        return AnthropicService(config)
    elif platform == 'openai':
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError(
                "OpenAI provider requires 'api_key' field"
            )
        return OpenAIService(config)
    elif platform == 'gemini':
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError(
                "Gemini provider requires 'api_key' field"
            )
        return GeminiService(config)
    raise ValueError(f"Unknown platform: {platform}")


def create_llm_service(config: dict):
    """
    Create an LLM service based on the platform field in config.

    Args:
        config: Dict with at least 'platform' (defaults to 'ollama').

    Returns:
        LLM service instance.
    """
    primary = _build_service(config)
    fallback_name = config.get('fallback_provider')
    if fallback_name:
        # Get fallback provider from config service
        try:
            from services.config_service import ConfigService
            providers = ConfigService.get_providers()
            if fallback_name in providers:
                fallback_config = dict(providers[fallback_name])
                fallback_config['platform'] = providers[fallback_name].get('platform', 'ollama')
                fallback = _build_service(fallback_config)
                return FallbackLLMService(primary, fallback)
        except Exception as e:
            logger.warning(f"Failed to load fallback provider '{fallback_name}': {e}")
    return primary


class AnthropicService:
    """Anthropic Claude API client."""

    # Anthropic's API requires max_tokens; use a large ceiling so the model
    # can decide how much output to generate naturally.
    _MAX_TOKENS = 16384

    def __init__(self, config: dict):
        self._config = config
        self.model = config.get('model', 'claude-haiku-4-5-20251001')
        self.timeout = config.get('timeout', 120)

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        import anthropic

        api_key = _resolve_api_key(self._config)
        client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

        start_time = time.time()

        def _call():
            return client.messages.create(
                model=self.model,
                max_tokens=self._MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

        response = _call_with_retry(_call)
        latency_ms = int((time.time() - start_time) * 1000)

        text = response.content[0].text if response.content else ""

        logger.info(
            f"[AnthropicService] model={response.model}, "
            f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}, "
            f"latency={latency_ms}ms"
        )

        return LLMResponse(
            text=text,
            model=response.model,
            provider='anthropic',
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency_ms,
        )


class OpenAIService:
    """OpenAI API client."""

    def __init__(self, config: dict):
        self._config = config
        self.model = config.get('model', 'gpt-4o-mini')
        self.timeout = config.get('timeout', 120)

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        from openai import OpenAI

        api_key = _resolve_api_key(self._config)
        client = OpenAI(api_key=api_key, timeout=self.timeout)

        start_time = time.time()

        def _call():
            return client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )

        response = _call_with_retry(_call)
        latency_ms = int((time.time() - start_time) * 1000)

        text = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        if not text or not text.strip():
            logger.warning(
                f"[OpenAIService] Empty response from model={response.model}, "
                f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}, "
                f"latency={latency_ms}ms, finish_reason={finish_reason}. "
                f"Content was: {repr(response.choices[0].message.content)}"
            )
        else:
            logger.info(
                f"[OpenAIService] model={response.model}, "
                f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}, "
                f"latency={latency_ms}ms"
            )

        return LLMResponse(
            text=text,
            model=response.model,
            provider='openai',
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )


class GeminiService:
    """Google Gemini API client."""

    def __init__(self, config: dict):
        self._config = config
        self.model = config.get('model', 'gemini-2.0-flash')

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        import google.generativeai as genai

        api_key = _resolve_api_key(self._config)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )

        start_time = time.time()

        def _call():
            return model.generate_content(user_message)

        response = _call_with_retry(_call)
        latency_ms = int((time.time() - start_time) * 1000)

        text = response.text if response.text else ""
        if not text:
            finish_reason = getattr(response, 'finish_reason', 'unknown')
            logger.warning(f"[GeminiService] Empty response, finish_reason={finish_reason}")
            raise ValueError(f"Empty Gemini response (finish_reason={finish_reason})")

        usage = getattr(response, 'usage_metadata', None)
        tokens_input = getattr(usage, 'prompt_token_count', None) if usage else None
        tokens_output = getattr(usage, 'candidates_token_count', None) if usage else None

        logger.info(
            f"[GeminiService] model={self.model}, "
            f"tokens={tokens_input}+{tokens_output}, "
            f"latency={latency_ms}ms"
        )

        return LLMResponse(
            text=text,
            model=self.model,
            provider='gemini',
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
        )
