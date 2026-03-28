"""
Ollama LLM service — local model inference via the Ollama HTTP API.

Wraps the Ollama ``/api/generate`` endpoint with retry logic (exponential
back-off on connection errors and 5xx responses), rate-limit handling
(HTTP 429 → :class:`~services.llm_service.RateLimitError`), and a thin
delegation to the shared :class:`~services.embedding_service.EmbeddingService`
for embedding generation.
"""

import logging
import time

import requests
import json
from services.llm_service import LLMResponse, RateLimitError


class OllamaService:
    """LLM service backed by a locally-running Ollama instance.

    Communicates with the Ollama HTTP API (``/api/generate``) and supports
    automatic retries with exponential back-off for transient network and
    server errors.  Embedding generation is delegated to the unified
    :class:`~services.embedding_service.EmbeddingService` rather than
    performing a separate Ollama embed call.
    """

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
                - ``temperature`` (float, default 0.5): Sampling temperature.
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
        self.host = config.get('host')
        self.model = config.get('model')
        self.keep_alive = config.get('keep_alive', '0')
        self.temperature = config.get('temperature', 0.5)
        self.timeout = config.get('timeout', 60)
        self.format = config.get('format', 'json')
        self.max_retries = config.get('max_retries', 2)

    def send_message(self, system_prompt: str, user_message: str, stream: bool = False) -> LLMResponse:
        """Send a message to Ollama and return the response."""
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": user_message,
            "system": system_prompt,
            "stream": False,
            "think": False,
            "raw": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            }
        }

        # Only add format if not "text" (Ollama treats omission as natural language)
        if self.format != "text":
            payload["format"] = self.format

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
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    backoff = 2 * (2 ** attempt)
                    logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after {type(e).__name__}: {e} — backoff {backoff}s")
                    time.sleep(backoff)
                else:
                    logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    retry_after = None
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                    raise RateLimitError(str(e), retry_after=retry_after, provider='ollama') from e
                elif e.response is not None and e.response.status_code >= 500:
                    if attempt < self.max_retries:
                        backoff = 1.5 * (2 ** attempt)
                        logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after HTTP {e.response.status_code} — backoff {backoff}s")
                        time.sleep(backoff)
                    else:
                        logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                        raise
                else:
                    raise

    def send_messages(self, system_prompt: str, messages: list, cache_prefix: bool = False, tools: list = None) -> LLMResponse:
        url = f"{self.host}/api/chat"

        # Convert normalized messages to Ollama format (OpenAI-compatible)
        api_messages = _ollama_convert_messages(messages)

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + api_messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
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

        for attempt in range(1 + self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                msg = data.get('message', {})
                text = msg.get('content', '')

                # Extract tool calls (Ollama uses OpenAI-compatible format)
                tool_calls = None
                raw_tool_calls = msg.get('tool_calls')
                if raw_tool_calls:
                    tool_calls = []
                    for i, tc in enumerate(raw_tool_calls):
                        fn = tc.get('function', {})
                        tool_calls.append({
                            'id': f"ollama_{fn.get('name', 'unknown')}_{i}",
                            'name': fn.get('name', ''),
                            'input': fn.get('arguments', {}),
                        })

                return LLMResponse(
                    text=text,
                    model=data.get('model', self.model),
                    provider='ollama',
                    tokens_input=data.get('prompt_eval_count'),
                    tokens_output=data.get('eval_count'),
                    tool_calls=tool_calls,
                    stop_reason='tool_use' if tool_calls else 'end_turn',
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    backoff = 2 * (2 ** attempt)
                    logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after {type(e).__name__}: {e} — backoff {backoff}s")
                    time.sleep(backoff)
                else:
                    logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    retry_after = None
                    ra = e.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                    raise RateLimitError(str(e), retry_after=retry_after, provider='ollama') from e
                elif e.response is not None and e.response.status_code >= 500:
                    if attempt < self.max_retries:
                        backoff = 1.5 * (2 ** attempt)
                        logging.warning(f"[OllamaService] Retry {attempt + 1}/{self.max_retries} after HTTP {e.response.status_code} — backoff {backoff}s")
                        time.sleep(backoff)
                    else:
                        logging.error(f"[OllamaService] All {1 + self.max_retries} attempts failed: {e}")
                        raise
                else:
                    raise

    def get_context_limit(self) -> int:
        """Query Ollama for model's context window size, cached."""
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit
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
                        self._cached_context_limit = int(val)
                        return self._cached_context_limit
        except Exception as e:
            logging.debug(f"[OllamaService] Failed to get context limit: {e}")
        self._cached_context_limit = 8192  # Conservative default
        return self._cached_context_limit

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Estimate tokens using heuristic (Ollama models vary too much for fixed tokenizer)."""
        from services.llm_service import estimate_tokens
        parts = [system_prompt] if system_prompt else []
        for msg in messages:
            parts.append(msg.get('content', '') or '')
        if tools:
            parts.append(json.dumps(tools, default=str))
        return estimate_tokens(' '.join(parts))

    def generate_embedding(self, text: str, embedding_model: str = None, target_dimensions: int = None) -> list:
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
