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


class RateLimitError(Exception):
    """Raised when an LLM provider returns HTTP 429."""
    def __init__(self, message: str, retry_after: float = None, provider: str = None):
        """Initialise the rate-limit error with optional retry metadata.

        Args:
            message: Human-readable description of the rate-limit condition.
            retry_after: Suggested wait time in seconds extracted from the
                provider's ``Retry-After`` response header, or ``None`` if
                the header was absent or could not be parsed.
            provider: Name of the provider that returned HTTP 429
                (e.g., ``'anthropic'``, ``'openai'``), or ``None`` if unknown.
        """
        super().__init__(message)
        self.retry_after = retry_after  # seconds, from Retry-After header
        self.provider = provider


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
    """Structured response returned by all LLM service implementations.

    Attributes:
        text: The generated text content from the model.
        model: The model identifier reported by the provider
            (e.g., ``'claude-haiku-4-5-20251001'``, ``'gpt-4o-mini'``).
        provider: Name of the provider that handled the request
            (``'anthropic'``, ``'openai'``, ``'gemini'``, ``'ollama'``),
            or ``None`` if not set by the implementation.
        tokens_input: Number of input/prompt tokens consumed, if reported
            by the provider.
        tokens_output: Number of output/completion tokens generated, if
            reported by the provider.
        latency_ms: End-to-end round-trip latency in milliseconds from
            request dispatch to response receipt.
    """

    text: str
    model: str
    provider: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    latency_ms: Optional[int] = None


def _call_with_retry(fn, max_retries=2, backoff=1.0):
    """Retry fn() up to max_retries times with exponential backoff.

    Rate limits (RateLimitError) get special treatment: longer backoff
    (Retry-After or 30s default, max 120s) and a separate retry counter
    (3 attempts) that doesn't consume generic retries.
    """
    attempt = 0
    rate_limit_retries = 0
    max_rate_limit_retries = 3
    while attempt <= max_retries:
        try:
            return fn()
        except RateLimitError as e:
            rate_limit_retries += 1
            if rate_limit_retries > max_rate_limit_retries:
                raise
            wait = min(e.retry_after or 30.0, 120.0)
            logger.warning(
                f"Rate limited by {e.provider or 'provider'} "
                f"(attempt {rate_limit_retries}/{max_rate_limit_retries}). "
                f"Waiting {wait:.0f}s..."
            )
            time.sleep(wait)
            # Don't increment attempt — rate limits have their own counter
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1


class FallbackLLMService:
    """Wraps a primary + fallback service. On primary failure, invokes fallback."""

    def __init__(self, primary, fallback):
        """Initialise the fallback wrapper with a primary and secondary client.

        Args:
            primary: The preferred LLM service instance to invoke first.
            fallback: The LLM service to invoke when the primary raises any
                exception other than ``RateLimitError``.
        """
        self._primary = primary
        self._fallback = fallback

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message, falling back to the secondary service on primary failure.

        ``RateLimitError`` from the primary is re-raised immediately without
        trying the fallback, because the fallback provider is likely to be
        rate-limited under the same load conditions.

        Args:
            system_prompt: Instruction context placed in the system role.
            user_message: The user-turn content to send to the model.
            stream: If ``True``, request streamed output (passed through to
                the underlying service; not yet supported by most providers).

        Returns:
            LLMResponse from whichever service successfully handled the request.

        Raises:
            RateLimitError: If the primary service returns HTTP 429.
            Exception: If both the primary and fallback services raise errors.
        """
        try:
            return self._primary.send_message(system_prompt, user_message, stream=stream)
        except RateLimitError:
            raise  # Rate limits propagate — fallback provider is likely also rate-limited
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


class RefreshableLLMService:
    """
    LLM service wrapper that auto-refreshes when provider configuration changes.

    Detects provider cache version changes (via MemoryStore invalidation) and re-creates
    the underlying LLM client, so workers don't need to restart when providers change
    via the Brain UI.
    """

    def __init__(self, agent_name: str):
        """Initialise the wrapper without immediately creating the underlying client.

        The underlying LLM client is lazily created on the first call to
        :meth:`send_message` and is transparently re-created whenever the
        provider cache version changes (i.e., when a provider is added,
        updated, or reassigned via the Brain UI).

        Args:
            agent_name: Agent configuration name used to resolve provider
                settings via ``ConfigService.resolve_agent_config``
                (e.g., ``'cognitive-drift'``, ``'mode-reflection'``).
        """
        self._agent_name = agent_name
        self._version = None  # Last seen provider cache version
        self._service = None  # Underlying LLM service

    def _ensure_fresh(self):
        """Re-create the underlying service if the provider cache version has changed."""
        from services.provider_cache_service import ProviderCacheService
        from services.config_service import ConfigService

        # Warm cache and get current version
        ProviderCacheService.get_providers()
        current_version = ProviderCacheService._version

        if current_version != self._version:
            logger.debug(
                f"[RefreshableLLM] Provider version changed ({self._version} → {current_version}), "
                f"re-creating LLM service for agent '{self._agent_name}'"
            )
            config = ConfigService.resolve_agent_config(self._agent_name)
            primary = _build_service(config)

            # Handle fallback provider
            fallback_name = config.get('fallback_provider')
            if fallback_name:
                try:
                    providers = ConfigService.get_providers()
                    if fallback_name in providers:
                        fallback_config = dict(providers[fallback_name])
                        fallback = _build_service(fallback_config)
                        primary = FallbackLLMService(primary, fallback)
                except Exception as e:
                    logger.warning(f"[RefreshableLLM] Failed to load fallback '{fallback_name}': {e}")

            self._service = primary
            self._version = current_version

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message, refreshing the underlying client if the provider config has changed.

        Calls :meth:`_ensure_fresh` before each request so that provider
        configuration updates (e.g., new API key or model) are picked up
        automatically without restarting the worker process.

        Args:
            system_prompt: Instruction context placed in the system role.
            user_message: The user-turn content to send to the model.
            stream: If ``True``, request streamed output (passed through to
                the underlying service; not yet supported by most providers).

        Returns:
            LLMResponse from the (potentially freshly re-created) underlying
            service instance.

        Raises:
            Any exception raised by the underlying LLM service.
        """
        self._ensure_fresh()
        return self._service.send_message(system_prompt, user_message, stream=stream)


def create_refreshable_llm_service(agent_name: str) -> RefreshableLLMService:
    """
    Create an LLM service that auto-refreshes when provider configuration changes.

    Use this instead of create_llm_service() for long-lived services that store
    the LLM client as an instance variable. The underlying client is re-created
    automatically when the provider cache version changes (e.g., after a provider
    is added, updated, or reassigned via the Brain UI).

    Args:
        agent_name: Agent config name (e.g., 'cognitive-drift', 'mode-reflection')

    Returns:
        RefreshableLLMService that transparently re-creates its client on changes.
    """
    return RefreshableLLMService(agent_name)


class AnthropicService:
    """Anthropic Claude API client."""

    # Anthropic's API requires max_tokens; use a large ceiling so the model
    # can decide how much output to generate naturally.
    _MAX_TOKENS = 16384

    def __init__(self, config: dict):
        """Initialise the Anthropic client with provider configuration.

        Args:
            config: Provider config dict sourced from the database.
                Required key: ``api_key``.
                Optional keys: ``model`` (default ``'claude-haiku-4-5-20251001'``),
                ``timeout`` (seconds, default ``120``).
        """
        self._config = config
        self.model = config.get('model', 'claude-haiku-4-5-20251001')
        self.timeout = config.get('timeout', 120)

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message to the Anthropic Messages API.

        Args:
            system_prompt: Text placed in the ``system`` role of the request.
            user_message: Text placed in the ``user`` role of the request.
            stream: Must be ``False``; streaming is not yet implemented.

        Returns:
            LLMResponse populated with the generated text, model identifier,
            token counts, and round-trip latency.

        Raises:
            NotImplementedError: If ``stream=True`` is requested.
            RateLimitError: If the API returns HTTP 429.
            anthropic.APIError: For other Anthropic API errors after retries
                are exhausted.
        """
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        import anthropic

        api_key = _resolve_api_key(self._config)
        client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

        start_time = time.time()

        def _call():
            try:
                return client.messages.create(
                    model=self.model,
                    max_tokens=self._MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
            except anthropic.RateLimitError as e:
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                raise RateLimitError(str(e), retry_after=retry_after, provider='anthropic') from e

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
        """Initialise the OpenAI client with provider configuration.

        Args:
            config: Provider config dict sourced from the database.
                Required key: ``api_key``.
                Optional keys: ``model`` (default ``'gpt-4o-mini'``),
                ``timeout`` (seconds, default ``120``),
                ``format`` (``'text'`` or ``'json'``, default ``'text'``).
        """
        self._config = config
        self.model = config.get('model', 'gpt-4o-mini')
        self.timeout = config.get('timeout', 120)
        self.format = config.get('format', 'text')

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message to the OpenAI Chat Completions API.

        When ``format='json'`` is set in the provider config, the request
        includes ``response_format={"type": "json_object"}`` so the model
        is instructed to return valid JSON.

        Args:
            system_prompt: Text placed in the ``system`` role of the request.
            user_message: Text placed in the ``user`` role of the request.
            stream: Must be ``False``; streaming is not yet implemented.

        Returns:
            LLMResponse populated with the generated text, model identifier,
            token counts, and round-trip latency.

        Raises:
            NotImplementedError: If ``stream=True`` is requested.
            RateLimitError: If the API returns HTTP 429.
            openai.APIError: For other OpenAI API errors after retries are
                exhausted.
        """
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        import openai as openai_mod
        from openai import OpenAI

        api_key = _resolve_api_key(self._config)
        client = OpenAI(api_key=api_key, timeout=self.timeout)

        start_time = time.time()

        create_kwargs = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if self.format == 'json':
            create_kwargs['response_format'] = {"type": "json_object"}

        def _call():
            try:
                return client.chat.completions.create(**create_kwargs)
            except openai_mod.RateLimitError as e:
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                raise RateLimitError(str(e), retry_after=retry_after, provider='openai') from e

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
        """Initialise the Gemini client with provider configuration.

        Args:
            config: Provider config dict sourced from the database.
                Required key: ``api_key``.
                Optional keys: ``model`` (default ``'gemini-2.5-flash'``),
                ``format`` (``'text'`` or ``'json'``, default ``'text'``).
                When ``format='json'``, the request uses
                ``response_mime_type='application/json'``.
        """
        self._config = config
        self.model = config.get('model', 'gemini-2.5-flash')
        self.format = config.get('format', 'text')

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message to the Google Gemini generative AI API.

        Args:
            system_prompt: Instruction passed as the system instruction in
                ``GenerateContentConfig``.
            user_message: The user-turn content to generate a response for.
            stream: Must be ``False``; streaming is not yet implemented.

        Returns:
            LLMResponse populated with the generated text, model identifier,
            token counts (from ``usage_metadata``), and round-trip latency.

        Raises:
            NotImplementedError: If ``stream=True`` is requested.
            RuntimeError: If the ``google-genai`` package is not installed.
            RateLimitError: If the API raises ``ResourceExhausted`` (HTTP 429).
            ValueError: If the model returns an empty response.
        """
        if stream:
            raise NotImplementedError("Streaming not yet supported")

        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Run: pip install google-genai"
            )

        api_key = _resolve_api_key(self._config)
        client = genai.Client(api_key=api_key)

        start_time = time.time()

        gen_config_kwargs = {'system_instruction': system_prompt}
        if self.format == 'json':
            gen_config_kwargs['response_mime_type'] = 'application/json'

        def _call():
            try:
                return client.models.generate_content(
                    model=self.model,
                    contents=user_message,
                    config=genai.types.GenerateContentConfig(**gen_config_kwargs),
                )
            except Exception as e:
                # Gemini SDK raises google.api_core.exceptions.ResourceExhausted for 429
                ename = type(e).__name__
                if 'ResourceExhausted' in ename or '429' in str(e):
                    raise RateLimitError(str(e), retry_after=None, provider='gemini') from e
                raise

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
