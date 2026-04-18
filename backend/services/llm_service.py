"""
LLM Service Factory — creates the right LLM client based on platform.

Usage:
    from services.llm_service import create_llm_service
    llm = create_llm_service(config)
    response = llm.send_message(system_prompt, user_message)
    text = response.text
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks emitted by reasoning
    models (MiniMax, DeepSeek-R1, Qwen-reasoning, etc.) served via OpenAI-
    compatible endpoints. Real OpenAI never emits these, so stripping is safe
    for all routes through OpenAIService."""
    if not text or "<think>" not in text.lower():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def estimate_tokens(text: str) -> int:
    """Fast token estimate (~1.3 tokens per whitespace-delimited word).

    Used as a fallback when provider-specific counting is unavailable,
    and for quick budget checks where exact counts aren't critical.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


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


class NonRetryableError(Exception):
    """Raised for permanent provider errors (5xx) that should not be retried."""


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
        tokens_thinking: CoT/reasoning tokens charged separately by the
            provider (OpenAI reasoning_tokens, Gemini thoughts_token_count).
            Anthropic: extended thinking tokens are INCLUDED in
            ``output_tokens`` per the Anthropic API docs — the SDK does not
            report a separate field, so ``tokens_thinking`` stays ``None``
            and thinking cost is reflected in ``tokens_output``. If a future
            SDK version exposes it separately, extract here and subtract
            from ``tokens_output`` to avoid double-counting.
        tokens_cache_read: Prompt-cache read tokens (Anthropic only).
        tokens_cache_create: Prompt-cache write tokens (Anthropic only).
        latency_ms: End-to-end round-trip latency in milliseconds from
            request dispatch to response receipt.
    """

    text: str
    model: str
    provider: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_thinking: Optional[int] = None
    tokens_cache_read: Optional[int] = None
    tokens_cache_create: Optional[int] = None
    tool_calls: Optional[list] = None
    stop_reason: Optional[str] = None
    latency_ms: Optional[int] = None


def _call_with_retry(fn, max_retries=2, backoff=1.0):
    """Retry fn() up to max_retries times with exponential backoff.

    Rate limits (RateLimitError) are retried once with a short wait if the
    provider includes a Retry-After header (capped at 10s).  If no header
    is present (typical for quota exhaustion), the error propagates
    immediately — quota exhaustion is not transient and sleeping 30-90s
    just blocks the request without helping.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            return fn()
        except RateLimitError as e:
            # If the provider told us to wait a short time, honour it once.
            # Otherwise fail fast — quota exhaustion won't resolve itself.
            if e.retry_after and e.retry_after <= 10.0:
                logger.warning(
                    f"Rate limited by {e.provider or 'provider'} "
                    f"(Retry-After {e.retry_after:.0f}s). Retrying once..."
                )
                time.sleep(e.retry_after)
                try:
                    return fn()
                except RateLimitError:
                    raise  # Second rate limit — give up
            raise  # No Retry-After or > 10s — fail fast
        except NonRetryableError:
            raise
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1


def _parse_retry_after(exc) -> Optional[float]:
    """Extract the Retry-After header value from an HTTP exception, or return None."""
    if hasattr(exc, 'response') and exc.response is not None:
        ra = exc.response.headers.get('retry-after')
        if ra:
            try:
                return float(ra)
            except (ValueError, TypeError) as parse_err:
                logger.debug(f"[LLM] Could not parse Retry-After header value {ra!r}: {parse_err}")
    return None


def _is_thinking_rejection(exc, create_kwargs: dict) -> bool:
    """Return True when the provider rejected a reasoning_effort parameter."""
    if 'reasoning_effort' not in create_kwargs:
        return False
    err = str(exc).lower()
    return 'reasoning_effort' in err or 'unsupported' in err


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

        Rate limits from the primary now trigger a fallback attempt —
        the fallback is often a different provider (e.g., local Ollama)
        that cannot be rate-limited by the primary's quota.

        Args:
            system_prompt: Instruction context placed in the system role.
            user_message: The user-turn content to send to the model.
            stream: If ``True``, request streamed output (passed through to
                the underlying service; not yet supported by most providers).

        Returns:
            LLMResponse from whichever service successfully handled the request.

        Raises:
            Exception: If both the primary and fallback services raise errors.
        """
        try:
            return self._primary.send_message(system_prompt, user_message, stream=stream)
        except Exception as e:
            logger.warning(f"Primary LLM failed ({type(e).__name__}), using fallback: {e}")
            return self._fallback.send_message(system_prompt, user_message, stream=stream)

    def send_messages(self, system_prompt: str, messages: list, cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        try:
            return self._primary.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        except Exception as e:
            logger.warning(f"Primary LLM failed ({type(e).__name__}), using fallback: {e}")
            return self._fallback.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)

    def get_context_limit(self) -> int:
        return self._primary.get_context_limit()

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        return self._primary.count_tokens(messages, system_prompt, tools)


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
    elif platform == 'openai_compatible':
        api_key = config.get('api_key')
        host = config.get('host')
        if not api_key:
            raise ValueError(
                "openai_compatible provider requires 'api_key' field"
            )
        if not host:
            raise ValueError(
                "openai_compatible provider requires 'host' field "
                "(base URL, e.g. 'https://api.minimax.io/v1')"
            )
        return OpenAIService(config)
    raise ValueError(f"Unknown platform: {platform}")


class LoggingLLMService:
    """Thin wrapper that logs every LLM call to the persistent call log."""

    def __init__(self, service, job_name: str):
        self._service = service
        self._job_name = job_name

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        result = self._service.send_message(system_prompt, user_message, stream=stream)
        _log_llm_call(self._job_name, result)
        return result

    def send_messages(self, system_prompt: str, messages: list, cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        result = self._service.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        _log_llm_call(self._job_name, result)
        return result

    def get_context_limit(self) -> int:
        return self._service.get_context_limit()

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        return self._service.count_tokens(messages, system_prompt, tools)


def create_llm_service(config: dict):
    """
    Create an LLM service based on the platform field in config.

    If the config contains a '_job_name' key (injected by
    ConfigService.resolve_agent_config), the returned service is wrapped
    with LoggingLLMService so every call is logged to the persistent store.

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
                primary = FallbackLLMService(primary, fallback)
        except Exception as e:
            logger.warning(f"Failed to load fallback provider '{fallback_name}': {e}")

    job_name = config.get('_job_name')
    if job_name:
        return LoggingLLMService(primary, job_name)
    return primary


def _log_llm_call(job_name: str, response: LLMResponse):
    """Fire-and-forget logging of LLM call to persistent store."""
    try:
        from services.llm_call_log_service import log_call
        log_call(
            job_name=job_name,
            provider=response.provider or 'unknown',
            model=response.model or 'unknown',
            tokens_input=response.tokens_input or 0,
            tokens_output=response.tokens_output or 0,
            latency_ms=response.latency_ms or 0,
        )
    except Exception as e:
        logger.debug(f"[LLM] _log_llm_call: persistent log write failed (non-fatal): {e}")


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
                (e.g., ``'cognitive-drift'``, ``'episodic-memory'``).
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
        result = self._service.send_message(system_prompt, user_message, stream=stream)
        _log_llm_call(self._agent_name, result)
        return result

    def send_messages(self, system_prompt: str, messages: list, cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        self._ensure_fresh()
        result = self._service.send_messages(system_prompt, messages, cache_prefix, tools=tools, thinking_mode=thinking_mode)
        _log_llm_call(self._agent_name, result)
        return result

    def get_context_limit(self) -> int:
        self._ensure_fresh()
        return self._service.get_context_limit()

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        self._ensure_fresh()
        return self._service.count_tokens(messages, system_prompt, tools)


def create_refreshable_llm_service(agent_name: str) -> RefreshableLLMService:
    """
    Create an LLM service that auto-refreshes when provider configuration changes.

    Use this instead of create_llm_service() for long-lived services that store
    the LLM client as an instance variable. The underlying client is re-created
    automatically when the provider cache version changes (e.g., after a provider
    is added, updated, or reassigned via the Brain UI).

    Args:
        agent_name: Agent config name (e.g., 'cognitive-drift', 'episodic-memory')

    Returns:
        RefreshableLLMService that transparently re-creates its client on changes.
    """
    return RefreshableLLMService(agent_name)


_ANTHROPIC_THINKING_BUDGETS = {'medium': 4096, 'high': 16384}


def _anthropic_raise_rate_limit(exc) -> None:
    """Parse a Retry-After header from an Anthropic RateLimitError and raise RateLimitError."""
    retry_after = None
    if hasattr(exc, 'response') and exc.response is not None:
        ra = exc.response.headers.get('retry-after')
        if ra:
            try:
                retry_after = float(ra)
            except (ValueError, TypeError) as parse_err:
                logger.debug(f"[LLM] Could not parse Retry-After header value {ra!r}: {parse_err}")
    raise RateLimitError(str(exc), retry_after=retry_after, provider='anthropic') from exc


def _anthropic_build_thinking_kwargs(thinking_mode: str, model: str) -> dict:
    """Return extra kwargs for the thinking flag, or an empty dict."""
    if thinking_mode not in _ANTHROPIC_THINKING_BUDGETS:
        return {}
    budget = _ANTHROPIC_THINKING_BUDGETS[thinking_mode]
    logger.info(f"[THINKING] native flag passed: provider=anthropic mode={thinking_mode} model={model}")
    return {'thinking': {'type': 'enabled', 'budget_tokens': budget}}


def _anthropic_parse_content_blocks(content) -> tuple:
    """Extract (text, tool_calls) from an Anthropic response content list."""
    text_parts = []
    tool_calls = []
    for block in (content or []):
        block_type = getattr(block, 'type', None)
        if block_type == 'tool_use':
            tool_calls.append({'id': block.id, 'name': block.name, 'input': block.input})
        elif block_type == 'thinking':
            logger.debug("[AnthropicService] thinking block present; usage deferred to caller")
        elif hasattr(block, 'text') and block.text:
            text_parts.append(block.text)
    return '\n'.join(text_parts), tool_calls or None


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

    def _get_client(self):
        import anthropic
        api_key = _resolve_api_key(self._config)
        return anthropic.Anthropic(api_key=api_key, timeout=self.timeout)

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

        client = self._get_client()
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
                _anthropic_raise_rate_limit(e)

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

    def send_messages(self, system_prompt: str, messages: list, cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        import anthropic

        client = self._get_client()
        start_time = time.time()

        system = (
            [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            if cache_prefix
            else system_prompt
        )

        # Anthropic: input_schema must be valid JSON Schema. Standard types only
        # (string, number, integer, boolean, array, object, null). Custom types rejected.
        # Enums: {"type": "string", "enum": [...]}. No oneOf/allOf at top level.
        # Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
        create_kwargs = {
            'model': self.model,
            'max_tokens': self._MAX_TOKENS,
            'system': system,
            'messages': _anthropic_convert_messages(messages),
            **({'tools': tools} if tools else {}),
            **_anthropic_build_thinking_kwargs(thinking_mode, self.model),
        }

        def _call():
            try:
                return client.messages.create(**create_kwargs)
            except anthropic.RateLimitError as e:
                _anthropic_raise_rate_limit(e)
            except (anthropic.BadRequestError, anthropic.APIError) as e:
                if 'thinking' in str(e).lower() and 'thinking' in create_kwargs:
                    logger.info(
                        f"[THINKING] native flag rejected by provider=anthropic model={self.model} — retried without"
                    )
                    return client.messages.create(**{k: v for k, v in create_kwargs.items() if k != 'thinking'})
                raise

        response = _call_with_retry(_call)
        latency_ms = int((time.time() - start_time) * 1000)

        # Thinking blocks are intentionally NOT surfaced in `text` — internal reasoning only.
        # Token cost is folded into response.usage.output_tokens per Anthropic API docs.
        text, tool_calls = _anthropic_parse_content_blocks(response.content)
        stop_reason = response.stop_reason  # 'end_turn', 'tool_use', 'max_tokens'

        logger.info(
            f"[AnthropicService] model={response.model}, "
            f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}, "
            f"latency={latency_ms}ms, stop={stop_reason}"
            + (f", tools={len(tool_calls)}" if tool_calls else "")
        )

        return LLMResponse(
            text=text,
            model=response.model,
            provider='anthropic',
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            tokens_cache_read=getattr(response.usage, 'cache_read_input_tokens', None),
            tokens_cache_create=getattr(response.usage, 'cache_creation_input_tokens', None),
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )

    def get_context_limit(self) -> int:
        """All Claude 3+ models support 200k context."""
        return 200_000

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Count tokens via Anthropic's server-side API (free, exact)."""
        try:
            client = self._get_client()
            kwargs = {
                'model': self.model,
                'messages': _anthropic_convert_messages(messages),
            }
            if system_prompt:
                kwargs['system'] = system_prompt
            if tools:
                kwargs['tools'] = tools
            result = client.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception as e:
            logger.debug(f"[AnthropicService] count_tokens API failed, using estimate: {e}")
            parts = [system_prompt] + [m.get('content', '') or '' for m in messages]
            return estimate_tokens(' '.join(parts))


def _anthropic_convert_messages(messages: list) -> list:
    """Convert normalized messages to Anthropic's content block format.

    Handles three message types:
    - Regular: {"role": "user"|"assistant", "content": "text"}
    - Assistant with tool calls: {"role": "assistant", "content": "text", "tool_calls": [...]}
    - Tool result: {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            # Build content blocks: text (if any) + tool_use blocks
            content = []
            text = msg.get('content', '')
            if text:
                content.append({"type": "text", "text": text})
            for tc in msg['tool_calls']:
                content.append({
                    "type": "tool_use",
                    "id": tc['id'],
                    "name": tc['name'],
                    "input": tc['input'],
                })
            result.append({"role": "assistant", "content": content})

        elif msg['role'] == 'tool':
            # Anthropic expects tool results as user messages with tool_result content blocks.
            # Collect consecutive tool results into a single user message.
            tool_results = []
            while i < len(messages) and messages[i]['role'] == 'tool':
                tm = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tm['tool_call_id'],
                    "content": tm.get('content', ''),
                })
                i += 1
            result.append({"role": "user", "content": tool_results})
            continue  # Skip the i += 1 at the bottom

        else:
            # Regular message — pass through
            result.append(msg)

        i += 1
    return result


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

    def _get_client(self):
        from openai import OpenAI
        api_key = _resolve_api_key(self._config)
        kwargs = {'api_key': api_key, 'timeout': self.timeout}
        base_url = self._config.get('host')
        if base_url:
            kwargs['base_url'] = base_url
        return OpenAI(**kwargs)

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

        client = self._get_client()
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
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[LLM] Could not parse Retry-After header value {ra!r}: {e}")
                raise RateLimitError(str(e), retry_after=retry_after, provider='openai') from e

        response = _call_with_retry(_call)
        latency_ms = int((time.time() - start_time) * 1000)

        text = _strip_think_blocks(response.choices[0].message.content or "")
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

    @staticmethod
    def _build_openai_tools(tools: list) -> list:
        """Convert normalized tool schemas to the OpenAI function-calling format.

        OpenAI: parameters must be valid JSON Schema. Standard types only.
        strict=true enforces exact compliance; best-effort without it.
        Enums: {"type": "string", "enum": [...]}. Custom types rejected.
        Ref: https://platform.openai.com/docs/guides/function-calling
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t['name'],
                    "description": t.get('description', ''),
                    "parameters": t.get('input_schema', {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _apply_reasoning_effort(self, create_kwargs: dict, thinking_mode: Optional[str]) -> None:
        """Inject reasoning_effort into create_kwargs when thinking_mode is set."""
        _reasoning_efforts = {'medium': 'medium', 'high': 'high'}
        if thinking_mode in _reasoning_efforts:
            create_kwargs['reasoning_effort'] = _reasoning_efforts[thinking_mode]
            logger.info(
                f"[THINKING] native flag passed: provider=openai mode={thinking_mode} model={self.model}"
            )

    def _call_completions(self, client, create_kwargs: dict):
        """Execute the completions API call with rate-limit and thinking-fallback handling."""
        import openai as openai_mod

        def _call():
            try:
                return client.chat.completions.create(**create_kwargs)
            except openai_mod.RateLimitError as e:
                retry_after = _parse_retry_after(e)
                raise RateLimitError(str(e), retry_after=retry_after, provider='openai') from e
            except (openai_mod.BadRequestError, openai_mod.APIError) as e:
                if _is_thinking_rejection(e, create_kwargs):
                    logger.info(
                        f"[THINKING] native flag rejected by provider=openai model={self.model} — retried without"
                    )
                    fallback_kwargs = {k: v for k, v in create_kwargs.items() if k != 'reasoning_effort'}
                    return client.chat.completions.create(**fallback_kwargs)
                raise

        return _call_with_retry(_call)

    @staticmethod
    def _parse_openai_tool_calls(msg) -> Optional[list]:
        """Parse tool calls from an OpenAI response message. Returns None if absent."""
        import json as _json

        if not msg.tool_calls:
            return None
        tool_calls = []
        for tc in msg.tool_calls:
            try:
                parsed_args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
            except _json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append({
                'id': tc.id,
                'name': tc.function.name,
                'input': parsed_args,
            })
        return tool_calls

    def send_messages(self, system_prompt: str, messages: list, _cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        # Note: _cache_prefix is accepted for interface uniformity but OpenAI's
        # completions API has no prefix-caching mechanism — it is intentionally unused.
        # Note: send_messages is the native-tool-calling / multi-turn path.
        # Never set response_format: json_object here — the prompt may not
        # mention "json" (OpenAI requires it), and tool calling uses its own
        # structured output protocol. Legacy JSON output lives in send_message.
        client = self._get_client()
        start_time = time.time()

        api_messages = _openai_convert_messages(messages)
        create_kwargs = {
            'model': self.model,
            'messages': [{"role": "system", "content": system_prompt}] + api_messages,
        }
        if tools:
            create_kwargs['tools'] = self._build_openai_tools(tools)
        self._apply_reasoning_effort(create_kwargs, thinking_mode)

        response = self._call_completions(client, create_kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        msg = response.choices[0].message
        text = _strip_think_blocks(msg.content or "")
        finish_reason = response.choices[0].finish_reason
        tool_calls = self._parse_openai_tool_calls(msg)

        log_level = logger.info if (text and text.strip()) or tool_calls else logger.warning
        log_level(
            f"[OpenAIService] model={response.model}, "
            f"tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}, "
            f"latency={latency_ms}ms, finish={finish_reason}"
            + (f", tools={len(tool_calls)}" if tool_calls else "")
        )

        _completion_details = getattr(response.usage, 'completion_tokens_details', None)
        _reasoning_tokens = getattr(_completion_details, 'reasoning_tokens', None) if _completion_details else None

        return LLMResponse(
            text=text,
            model=response.model,
            provider='openai',
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            tokens_thinking=_reasoning_tokens,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
        )

    def get_context_limit(self) -> int:
        """Default 128k for GPT-4 class models."""
        return 128_000

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Count tokens using tiktoken if available, else estimate."""
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding('cl100k_base')

            import json as _json
            parts = []
            if system_prompt:
                parts.append(system_prompt)
            for msg in messages:
                parts.append(msg.get('content', '') or '')
                if msg.get('tool_calls'):
                    parts.append(_json.dumps(msg['tool_calls'], default=str))
            if tools:
                parts.append(_json.dumps(tools, default=str))
            text = '\n'.join(parts)
            overhead = (len(messages) + 1) * 4  # ~4 tokens per message for framing
            return len(enc.encode(text)) + overhead
        except ImportError:
            logger.debug("[LLM] tiktoken not installed; falling back to token estimate")
        except Exception as e:
            logger.debug(f"[OpenAIService] tiktoken counting failed: {e}")
        parts = [system_prompt] + [m.get('content', '') or '' for m in messages]
        return estimate_tokens(' '.join(parts))


def _openai_convert_messages(messages: list) -> list:
    """Convert normalized messages to OpenAI format."""
    import json as _json
    result = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            oai_msg = {
                "role": "assistant",
                "content": msg.get('content') or None,
                "tool_calls": [
                    {
                        "id": tc['id'],
                        "type": "function",
                        "function": {
                            "name": tc['name'],
                            "arguments": _json.dumps(tc['input']),
                        },
                    }
                    for tc in msg['tool_calls']
                ],
            }
            result.append(oai_msg)
        elif msg['role'] == 'tool':
            result.append({
                "role": "tool",
                "tool_call_id": msg['tool_call_id'],
                "content": msg.get('content', ''),
            })
        else:
            result.append(msg)
    return result


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
                # 5xx server errors are permanent — do not retry
                if 'ServerError' in ename or '500' in str(e) or '503' in str(e):
                    raise NonRetryableError(f"Gemini server error (no retry): {e}") from e
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

    def send_messages(self, system_prompt: str, messages: list, _cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai package is not installed. "
                "Run: pip install google-genai"
            )

        client = genai.Client(api_key=_resolve_api_key(self._config))
        start_time = time.time()
        gemini_contents = _gemini_convert_messages(messages)
        gen_config_kwargs = self._gemini_build_config(genai, system_prompt, tools, thinking_mode)
        response = _call_with_retry(
            lambda: self._gemini_generate(client, genai, gemini_contents, gen_config_kwargs)
        )
        latency_ms = int((time.time() - start_time) * 1000)
        text, tool_calls, finish_reason = self._gemini_parse_response(response)

        if not text and not tool_calls:
            logger.warning(f"[GeminiService] Empty response, finish_reason={finish_reason}")
            raise ValueError(f"Empty Gemini response (finish_reason={finish_reason})")

        usage = getattr(response, 'usage_metadata', None)
        tokens_input = getattr(usage, 'prompt_token_count', None) if usage else None
        tokens_output = getattr(usage, 'candidates_token_count', None) if usage else None
        tokens_thinking = getattr(usage, 'thoughts_token_count', None) if usage else None

        logger.info(
            f"[GeminiService] model={self.model}, "
            f"tokens={tokens_input}+{tokens_output}, "
            f"latency={latency_ms}ms"
            + (f", tools={len(tool_calls)}" if tool_calls else "")
        )

        return LLMResponse(
            text=text,
            model=self.model,
            provider='gemini',
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_thinking=tokens_thinking,
            latency_ms=latency_ms,
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=finish_reason,
        )

    def _gemini_build_config(self, genai, system_prompt: str, tools: list, thinking_mode: str) -> dict:
        """Build the GenerateContentConfig kwargs dict for a multi-turn request."""
        cfg: dict = {'system_instruction': system_prompt}
        if self.format == 'json' and not tools:
            cfg['response_mime_type'] = 'application/json'
        if tools:
            # Gemini: parameters follow OpenAPI schema subset (stricter than JSON Schema).
            # Standard types only. No default values in schema. Custom types rejected.
            # Enums: {"type": "string", "enum": [...]}. No oneOf/allOf/anyOf.
            # Ref: https://ai.google.dev/gemini-api/docs/function-calling
            cfg['tools'] = [
                genai.types.Tool(function_declarations=[
                    genai.types.FunctionDeclaration(
                        name=t['name'],
                        description=t.get('description', ''),
                        parameters=t.get('input_schema'),
                    )
                    for t in tools
                ])
            ]
        _thinking_budgets = {'medium': 4096, 'high': 16384}
        if thinking_mode in _thinking_budgets:
            cfg['thinking_config'] = genai.types.ThinkingConfig(
                thinking_budget=_thinking_budgets[thinking_mode]
            )
            logger.info(
                f"[THINKING] native flag passed: provider=gemini mode={thinking_mode} model={self.model}"
            )
        return cfg

    def _gemini_generate(self, client, genai, gemini_contents, gen_config_kwargs: dict):
        """Execute a single generate_content call, mapping provider errors to typed exceptions."""
        try:
            return client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=genai.types.GenerateContentConfig(**gen_config_kwargs),
            )
        except Exception as e:
            ename = type(e).__name__
            if 'ResourceExhausted' in ename or '429' in str(e):
                raise RateLimitError(str(e), retry_after=None, provider='gemini') from e
            if 'ServerError' in ename or '500' in str(e) or '503' in str(e):
                raise NonRetryableError(f"Gemini server error (no retry): {e}") from e
            if 'thinking_config' in gen_config_kwargs and (
                'thinking' in str(e).lower() or 'unsupported' in str(e).lower()
            ):
                logger.info(
                    f"[THINKING] native flag rejected by provider=gemini model={self.model} — retried without"
                )
                fallback_kwargs = {k: v for k, v in gen_config_kwargs.items() if k != 'thinking_config'}
                return client.models.generate_content(
                    model=self.model,
                    contents=gemini_contents,
                    config=genai.types.GenerateContentConfig(**fallback_kwargs),
                )
            raise

    def _gemini_parse_response(self, response) -> tuple:
        """Extract (text, tool_calls, finish_reason) from a Gemini response object."""
        text_parts = []
        tool_calls = []
        if response.candidates:
            for part in (response.candidates[0].content.parts or []):
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        'id': f"gemini_{fc.name}_{int(time.time()*1000)}",
                        'name': fc.name,
                        'input': dict(fc.args) if fc.args else {},
                    })
        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = str(response.candidates[0].finish_reason)
        return '\n'.join(text_parts), tool_calls, finish_reason

    def get_context_limit(self) -> int:
        """Query Gemini API for model's input token limit, cached."""
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit
        try:
            from google import genai
            api_key = _resolve_api_key(self._config)
            client = genai.Client(api_key=api_key)
            model_info = client.models.get(model=self.model)
            self._cached_context_limit = model_info.input_token_limit
            return self._cached_context_limit
        except Exception as e:
            logger.debug(f"[GeminiService] Failed to get context limit: {e}")
            self._cached_context_limit = 1_000_000  # Gemini models default
            return self._cached_context_limit

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Count tokens via Gemini's server-side API (free, exact)."""
        try:
            from google import genai
            api_key = _resolve_api_key(self._config)
            client = genai.Client(api_key=api_key)
            contents = _gemini_convert_messages(messages)
            result = client.models.count_tokens(
                model=self.model,
                contents=contents,
            )
            # System prompt counted separately since count_tokens may not accept it
            sys_tokens = estimate_tokens(system_prompt) if system_prompt else 0
            return result.total_tokens + sys_tokens
        except Exception as e:
            logger.debug(f"[GeminiService] count_tokens API failed, using estimate: {e}")
            parts = [system_prompt] + [m.get('content', '') or '' for m in messages]
            return estimate_tokens(' '.join(parts))


def _gemini_convert_messages(messages: list) -> list:
    """Convert normalized messages to Gemini format."""
    result = []
    for msg in messages:
        role = "model" if msg['role'] == 'assistant' else msg['role']
        if msg['role'] == 'tool':
            # Gemini expects function responses as user messages
            result.append({
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": msg.get('name', ''),
                        "response": {"content": msg.get('content', '')},
                    }
                }],
            })
        elif msg['role'] == 'assistant' and msg.get('tool_calls'):
            parts = []
            text = msg.get('content', '')
            if text:
                parts.append({"text": text})
            for tc in msg['tool_calls']:
                parts.append({
                    "function_call": {
                        "name": tc['name'],
                        "args": tc['input'],
                    }
                })
            result.append({"role": "model", "parts": parts})
        else:
            result.append({
                "role": role,
                "parts": [{"text": msg.get('content', '')}],
            })
    return result
