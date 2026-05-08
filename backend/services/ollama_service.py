"""
Ollama LLM service — local model inference via the Ollama HTTP API.

Wraps the Ollama ``/api/generate`` and ``/api/chat`` endpoints with retry
logic (exponential back-off on connection errors and 5xx responses),
rate-limit handling (HTTP 429 → :class:`~services.llm_service.RateLimitError`),
and a thin delegation to the shared
:class:`~services.embedding_service.EmbeddingService` for embedding generation.
"""

import logging
import re
import time
from urllib.parse import urlparse

import requests
import json
from services.llm_service import LLMResponse, PayloadTooLargeError, RateLimitError

# Model identifier accepts alphanumeric, dot, underscore, dash, slash, and the
# `:cloud` / `:7b` size suffix separator. Validating in __init__ acts as a
# CodeQL sanitisation barrier so logging the model name later does not
# trip py/clear-text-logging-sensitive-data on the config-derived value.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:\-/]+$")
_ALLOWED_HOST_SCHEMES = frozenset({"http", "https"})

# Ollama Cloud (model name suffix ``:cloud``) is fronted by an HTTP edge proxy
# whose request-body size limit is far below the model's nominal context
# window (e.g. kimi-k2.6:cloud reports 262 144 tokens but the edge rejects
# payloads at a much smaller byte threshold with HTTP 413). Clamping the
# reported context length here makes ``MessageProcessor._check_threshold``
# fire Stage 1/2 compaction before the request leaves the box.
OLLAMA_CLOUD_CONTEXT_CAP = 65_536


class OllamaService:
    """LLM service backed by a locally-running Ollama instance.

    Communicates with the Ollama HTTP API (``/api/generate``) and supports
    automatic retries with exponential back-off for transient network and
    server errors.  Embedding generation is delegated to the unified
    :class:`~services.embedding_service.EmbeddingService` rather than
    performing a separate Ollama embed call.
    """

    # JSON path of the user-visible text field in this provider's response.
    # Substituted into system prompts at the {{provider_content_field_name}}
    # placeholder so the model is told the exact field where its prose lands.
    CONTENT_FIELD_LABEL = "message.content"

    def __init__(self, config: dict):
        """Initialize the Ollama service with connection and inference settings.

        Args:
            config: Configuration dict.  Recognised keys:

                - ``platform`` (str): Must be ``'ollama'``; raises
                  :exc:`ValueError` otherwise.
                - ``host`` (str): Base URL of the Ollama server
                  (e.g., ``'http://localhost:11434'``).
                - ``model`` (str): Name of the Ollama model to use.
                - ``keep_alive`` (str, default ``'0'``): Ollama keep-alive
                  duration passed verbatim to the API.
                - ``timeout`` (int, default 60): HTTP request timeout in
                  seconds.
                - ``format`` (str, default ``'json'``): Response format.
                  Pass ``'text'`` to omit the format field from the request.
                - ``max_retries`` (int, default 2): Number of additional
                  attempts after an initial failure.

        Raises:
            ValueError: If ``config['platform']`` is not ``'ollama'``.
        """
        platform = config.get('platform', 'ollama')
        if platform != 'ollama':
            raise ValueError(f"OllamaService does not support platform '{platform}'")

        self._config = config
        self.host = _validate_host(config.get('host'))
        self.model = _validate_model(config.get('model'))
        self.keep_alive = config.get('keep_alive', '0')
        self.timeout = config.get('timeout', 60)
        self.format = config.get('format', 'json')
        self.max_retries = config.get('max_retries', 2)
        # Cached result of _model_supports_thinking(). None = not yet checked.
        self._thinking_supported: bool | None = None

    def send_message(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Send a message to Ollama and return the response."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": user_message,
            "system": system_prompt,
            "stream": False,
            "raw": False,
            "keep_alive": self.keep_alive,
        }

        # Only add format if not "text" (Ollama treats omission as natural language)
        if self.format != "text":
            payload["format"] = self.format

        start_time = time.time()
        for attempt in range(1 + self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    text=data['response'],
                    model=data.get('model', self.model),
                    provider='ollama',
                    tokens_input=data.get('prompt_eval_count'),
                    tokens_output=data.get('eval_count'),
                    latency_ms=int((time.time() - start_time) * 1000),
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self._handle_network_error(e, attempt)
            except requests.exceptions.HTTPError as e:
                self._handle_http_error(e, attempt)

    def build_request_body(self, system_prompt: str, messages: list, tools: list = None, **_kwargs) -> str:
        """Return the serialised request body that send_messages would POST.

        Uses the same _build_chat_payload path as send_messages; thinking_mode
        is omitted (None) because the think flag adds negligible token weight
        and requires a live /api/show preflight — unsuitable for measurement.
        requests.post(json=payload) serialises with json.dumps; we use the
        same call here so measurement and wire body are identical.
        """
        api_messages = _ollama_convert_messages(messages)
        payload = self._build_chat_payload(system_prompt, api_messages, tools, thinking_mode=None)
        return json.dumps(payload)

    def send_messages(self, system_prompt: str, messages: list, _cache_prefix: bool = False, tools: list = None, thinking_mode: str = None) -> LLMResponse:
        # ``_cache_prefix`` is named with a leading underscore on purpose:
        # interface parity with Anthropic, but Ollama has no prefix-cache so
        # the value is intentionally ignored.
        url = f"{self.host}/api/chat"

        api_messages = _ollama_convert_messages(messages)
        payload = self._build_chat_payload(system_prompt, api_messages, tools, thinking_mode)

        start_time = time.time()
        for attempt in range(1 + self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                parsed = _parse_chat_response(response.json(), self.model)
                parsed.latency_ms = int((time.time() - start_time) * 1000)
                return parsed
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self._handle_network_error(e, attempt)
            except requests.exceptions.HTTPError as e:
                self._handle_http_error(e, attempt)

    def _build_chat_payload(self, system_prompt: str, api_messages: list, tools: list, thinking_mode: 'str | None') -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + api_messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = [
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
        # Ollama native thinking: binary True/False in the chat API (top-level
        # field, not nested under options). 'medium' and 'high' both map to
        # True — Ollama has no budget granularity.
        # Pre-flight: only send the flag when the model advertises 'thinking'
        # in its /api/show capabilities list. Unknown models default to False
        # (safe — the flag simply doesn't appear in the payload).
        if thinking_mode in ('medium', 'high'):
            if self._model_supports_thinking():
                payload["think"] = True
                logging.info(
                    f"[THINKING] native flag passed: provider=ollama mode={thinking_mode} model={self.model}"
                )
            else:
                logging.info(
                    f"[THINKING] provider=ollama model={self.model} does not support "
                    f"native think — request sent without flag"
                )
        return payload

    def _handle_network_error(self, e: Exception, attempt: int) -> None:
        if attempt < self.max_retries:
            backoff = 2 * (2 ** attempt)
            logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after {type(e).__name__} — backoff {backoff}s")
            time.sleep(backoff)
        else:
            logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed ({type(e).__name__})")
            raise e

    def _handle_http_error(self, e: requests.exceptions.HTTPError, attempt: int) -> None:
        status = e.response.status_code if e.response is not None else None
        if status == 429:
            _raise_rate_limit(e)
        if status == 413:
            self._raise_payload_too_large(e)
        if status is not None and status >= 500:
            self._handle_5xx_error(e, attempt)
            return
        self._log_4xx_error()
        raise e

    def _raise_payload_too_large(self, e: requests.exceptions.HTTPError) -> None:
        # Static log message — every variable previously interpolated here
        # (self.model, payload metadata, response body length) trips
        # py/clear-text-logging-sensitive-data because CodeQL traces back
        # through ``__init__``'s config-dict source. The PayloadTooLargeError
        # raised below carries the model name in its formatted message, and
        # operators correlate via timestamps; that's the right channel.
        logging.warning("[OllamaService] HTTP 413 — raising PayloadTooLargeError to trigger Stage 2 compaction")
        raise PayloadTooLargeError(
            f"Ollama rejected payload with HTTP 413 (model={self.model})"
        ) from e

    def _handle_5xx_error(self, e: requests.exceptions.HTTPError, attempt: int) -> None:
        # See _raise_payload_too_large for the static-string rationale.
        # The exception ``e`` re-raised below carries the response, so
        # callers / outer ``except`` blocks have full diagnostic data.
        if attempt < self.max_retries:
            backoff = 1.5 * (2 ** attempt)
            logging.warning("[OllamaService] retrying after HTTP 5xx")
            time.sleep(backoff)
            return
        logging.error("[OllamaService] retries exhausted on HTTP 5xx")
        raise e

    def _log_4xx_error(self) -> None:
        # Static log — same rationale as _raise_payload_too_large.
        # The ``HTTPError`` re-raised by the caller carries the status code
        # for any caller that wants to switch on it.
        logging.error("[OllamaService] HTTP 4xx error from upstream")

    def _model_supports_thinking(self) -> bool:
        """Return True if the configured model advertises thinking capability.

        Queries /api/show once per OllamaService instance and caches the result.
        Returns False on any network or parse error (safe default — the think
        flag simply won't be sent).

        Ollama's /api/show response includes a ``capabilities`` list. Models
        that support the think flag include ``'thinking'`` in that list.
        """
        if self._thinking_supported is not None:
            return self._thinking_supported
        try:
            resp = requests.post(
                f"{self.host}/api/show",
                json={"name": self.model},
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                caps = data.get('capabilities', [])
                self._thinking_supported = 'thinking' in caps
                return self._thinking_supported
        except Exception as exc:
            logging.debug(
                f"[OllamaService] _model_supports_thinking check failed for "
                f"model={self.model}: {exc}"
            )
        self._thinking_supported = False
        return False

    def get_context_limit(self) -> int:
        """Query Ollama for model's context window size, cached.

        For ``:cloud`` models the result is clamped to
        ``OLLAMA_CLOUD_CONTEXT_CAP`` because the cloud edge proxy enforces a
        much smaller body-size limit than the model's nominal context window.

        Cache lifetime: instance-scoped. ``Providers._resolve()`` builds a
        fresh ``OllamaService`` on each call, so the cache effectively lives
        for one provider call only. Do not store an ``OllamaService`` long-
        term and assume the cap stays current.
        """
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit
        raw_limit: int | None = None
        try:
            resp = requests.post(
                f"{self.host}/api/show",
                json={"name": self.model},
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                model_info = data.get('model_info', {})
                for key, val in model_info.items():
                    if 'context_length' in key.lower():
                        raw_limit = int(val)
                        break
        except Exception as e:
            logging.debug(f"[OllamaService] Failed to get context limit: {e}")
        if raw_limit is None:
            raw_limit = 8192  # Conservative default
        self._cached_context_limit = self._apply_cloud_cap(raw_limit)
        return self._cached_context_limit

    def _apply_cloud_cap(self, raw_limit: int) -> int:
        # Lower-cased compare so future tag variants (':Cloud', ':CLOUD') are
        # caught. Suffix-only — cloud-routed models always end with ':cloud'
        # in Ollama's current naming convention.
        model_lower = (self.model or '').lower()
        if model_lower.endswith(':cloud') and raw_limit > OLLAMA_CLOUD_CONTEXT_CAP:
            logging.info(
                "[OllamaService] Clamping :cloud model context window: "
                "model=%s reported=%d capped=%d (edge proxy body-size limit)",
                self.model, raw_limit, OLLAMA_CLOUD_CONTEXT_CAP,
            )
            return OLLAMA_CLOUD_CONTEXT_CAP
        return raw_limit

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Estimate tokens using heuristic (Ollama models vary too much for fixed tokenizer)."""
        from services.llm_service import estimate_tokens
        parts = [system_prompt] if system_prompt else []
        for msg in messages:
            parts.append(msg.get('content', '') or '')
        if tools:
            parts.append(json.dumps(tools, default=str))
        return estimate_tokens(' '.join(parts))

    def generate_embedding(self, text: str, _embedding_model: str = None, _target_dimensions: int = None) -> list:
        """
        Generate embedding vector via EmbeddingService (no Ollama required).

        Note: embedding_model and target_dimensions parameters are deprecated and ignored.
        All embeddings now use the unified EmbeddingService.

        Args:
            text: Text to embed
            embedding_model: (deprecated, ignored)
            target_dimensions: (deprecated, ignored)

        Returns:
            Embedding vector (768-dim, L2-normalized)

        Raises:
            Exception if embedding generation fails
        """
        try:
            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            return emb_service.generate_embedding(text)

        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            raise


def _validate_model(raw_model) -> str:
    """Reject Ollama model identifiers that contain anything but the canonical
    name characters. Raising here acts as a CodeQL sanitisation barrier so
    ``self.model`` is no longer treated as derived-from-config-dict tainted
    data when it later lands in log calls.
    """
    if raw_model is None:
        raise ValueError("Ollama config requires a 'model' field")
    text = str(raw_model)
    if not _MODEL_NAME_RE.fullmatch(text):
        raise ValueError(f"Invalid Ollama model identifier: {raw_model!r}")
    return text


def _validate_host(raw_host) -> str:
    """Validate the Ollama host URL via ``urllib.parse.urlparse`` + scheme +
    netloc gates. Same CodeQL-barrier rationale as ``_validate_model``.
    """
    if raw_host is None:
        raise ValueError("Ollama config requires a 'host' field")
    text = str(raw_host)
    parsed = urlparse(text)
    if parsed.scheme not in _ALLOWED_HOST_SCHEMES or not parsed.netloc:
        raise ValueError(f"Invalid Ollama host URL: {raw_host!r}")
    return text


def _raise_rate_limit(e: requests.exceptions.HTTPError) -> None:
    """Parse Retry-After from a 429 response and raise RateLimitError."""
    retry_after = None
    if e.response is not None:
        ra = e.response.headers.get('retry-after')
        if ra:
            try:
                retry_after = float(ra)
            except (ValueError, TypeError):
                pass
    raise RateLimitError(str(e), retry_after=retry_after, provider='ollama') from e


def _parse_chat_response(data: dict, default_model: str) -> LLMResponse:
    """Build an LLMResponse from a raw Ollama /api/chat response dict.

    Native tool-calling only: tool calls are read exclusively from the
    structured ``message.tool_calls`` field. No inline-content fallback.
    """
    msg = data.get('message', {})
    text = msg.get('content', '')
    tool_calls = None
    raw_tool_calls = msg.get('tool_calls')
    if raw_tool_calls:
        tool_calls = [
            {
                'id': f"ollama_{tc.get('function', {}).get('name', 'unknown')}_{i}",
                'name': tc.get('function', {}).get('name', ''),
                'input': tc.get('function', {}).get('arguments', {}),
            }
            for i, tc in enumerate(raw_tool_calls)
        ]
    return LLMResponse(
        text=text,
        model=data.get('model', default_model),
        provider='ollama',
        tokens_input=data.get('prompt_eval_count'),
        tokens_output=data.get('eval_count'),
        tool_calls=tool_calls,
        stop_reason='tool_use' if tool_calls else 'end_turn',
    )


def _ollama_convert_messages(messages: list) -> list:
    """Convert normalized messages to Ollama format (OpenAI-compatible)."""
    result = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append({
                "role": "assistant",
                "content": msg.get('content', ''),
                "tool_calls": [
                    {
                        "function": {
                            "name": tc['name'],
                            "arguments": tc['input'],
                        },
                    }
                    for tc in msg['tool_calls']
                ],
            })
        elif msg['role'] == 'tool':
            result.append({
                "role": "tool",
                "content": msg.get('content', ''),
            })
        else:
            result.append(msg)
    return result
