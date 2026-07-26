"""
Ollama thin client — wraps the Ollama /api/chat endpoint.

Native size-rejection signal: HTTP 413 → ContextLimit.
This client maps it to ContextLimit instead.

Known quirk (preserved): the `think` flag is gated on model capability
(via /api/show), NOT on ThinkingLevel. ThinkingLevel is effectively ignored
for the on/off decision; only MEDIUM/HIGH/MAX enable the think flag at all,
but whether it actually appears in the payload depends on the model.

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens,
_app_user_agent).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import ClassVar, Optional, cast
from urllib.parse import urlparse
from uuid import uuid4

import requests

from configs.enums.thinking_level import ThinkingLevel
from contracts.provider_client import ProviderClient
from services.llm_clients.thinking_map import OLLAMA_THINK
from exceptions import (
    ContextLimit,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from services.provider_api import (
    PROVIDER_CALL_TIMEOUT_S,
    ProviderApiRequest,
    ProviderApiResponse,
)

logger = logging.getLogger(__name__)

# Model identifier accepts alphanumeric, dot, underscore, dash, slash, and the
# `:cloud` / `:7b` size suffix separator. Validating in __init__ acts as a
# CodeQL sanitisation barrier so logging the model name later does not
# trip py/clear-text-logging-sensitive-data on the config-derived value.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:\-/]+$")
_ALLOWED_HOST_SCHEMES = frozenset({"http", "https"})


def _validate_model(raw_model: object) -> str:
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


def _validate_host(raw_host: object) -> str:
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


def _ollama_convert_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert normalised messages to Ollama format (OpenAI-compatible)."""
    result: list[dict[str, object]] = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append({
                "role": "assistant",
                "content": msg.get('content', ''),
                "tool_calls": [
                    {"function": {"name": cast(dict[str, object], tc)['name'], "arguments": cast(dict[str, object], tc)['input']}}
                    for tc in cast(list[object], msg['tool_calls'])
                ],
            })
        elif msg['role'] == 'tool':
            result.append({"role": "tool", "content": msg.get('content', '')})
        else:
            img = msg.get('image')
            if img:
                result.append({
                    "role": msg['role'],
                    "content": msg.get('content', ''),
                    "images": [cast(dict[str, object], img)['data']],
                })
            else:
                result.append(msg)
    return result


def _parse_chat_response(data: dict[str, object], default_model: str) -> ProviderApiResponse:
    """Build a ProviderApiResponse from a raw Ollama /api/chat response dict."""
    msg = cast(dict[str, object], data.get('message', {}))
    text = cast(str, msg.get('content', ''))
    tool_calls = None
    raw_tool_calls = msg.get('tool_calls')
    if raw_tool_calls:
        tool_calls = [
            {
                'id': f"ollama_{cast(dict[str, object], cast(dict[str, object], tc).get('function', {})).get('name', 'unknown')}_{uuid4().hex[:8]}",
                'name': cast(dict[str, object], cast(dict[str, object], tc).get('function', {})).get('name', ''),
                'input': cast(dict[str, object], cast(dict[str, object], tc).get('function', {})).get('arguments', {}),
            }
            for tc in cast(list[object], raw_tool_calls)
        ]
    return ProviderApiResponse(
        text=text,
        model=cast(str, data.get('model', default_model)),
        provider='ollama',
        tokens_input=cast(Optional[int], data.get('prompt_eval_count')),
        tokens_output=cast(Optional[int], data.get('eval_count')),
        tool_calls=tool_calls,
        stop_reason='tool_use' if tool_calls else 'end_turn',
        response_code=200,
    )


class OllamaClient(ProviderClient):
    CONTENT_FIELD_LABEL: ClassVar[str] = "message.content"

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self.host: str = _validate_host(config.get('host'))
        self.model: str = _validate_model(config.get('model'))
        self._keep_alive: str = cast(str, config.get('keep_alive', '0'))
        # Cached result of _model_supports_thinking(). None = not yet checked.
        self._thinking_supported: Optional[bool] = None

    def _user_agent(self) -> dict[str, str]:
        from services.llm_service import _app_user_agent  # noqa: PLC0415
        return {"User-Agent": _app_user_agent()}

    def _model_supports_thinking(self) -> bool:
        """Return True if the configured model advertises thinking capability.

        Queries /api/show once per instance and caches the result. Returns
        False on any error (safe default — think flag simply won't be sent).
        """
        if self._thinking_supported is not None:
            return self._thinking_supported
        try:
            resp = requests.post(
                f"{self.host}/api/show",
                json={"name": self.model},
                timeout=5,
                headers=self._user_agent(),
            )
            if resp.ok:
                caps = resp.json().get('capabilities', [])
                self._thinking_supported = 'thinking' in caps
                return self._thinking_supported
        except Exception as exc:
            logger.debug(
                "[OllamaClient] _model_supports_thinking check failed for model=%s: %s",
                self.model, exc,
            )
        self._thinking_supported = False
        return False

    def _build_payload(self, system: str, api_messages: list[dict[str, object]], tools: Optional[list[dict[str, object]]],
                       thinking_mode: ThinkingLevel) -> dict[str, object]:
        """Build the /api/chat payload dict."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + api_messages,
            "stream": False,
            "keep_alive": self._keep_alive,
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
        # Quirk preserved: think flag is binary and gated on model capability,
        # NOT graduated by level. NONE sends an explicit False — Ollama enables
        # thinking by default on capable models, so omission is not "off".
        think = OLLAMA_THINK.get(thinking_mode)
        if think is not None:
            if self._model_supports_thinking():
                payload["think"] = think
                logger.info(
                    "[THINKING] native flag passed: provider=ollama mode=%s model=%s think=%s",
                    thinking_mode.value, self.model, think,
                )
            else:
                logger.info(
                    "[THINKING] provider=ollama model=%s does not support native think"
                    " — request sent without flag", self.model,
                )
        return payload

    def _handle_http_error(self, exc: requests.exceptions.HTTPError) -> None:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            raise RateLimitError(str(exc), provider='ollama') from exc
        if status == 413:
            raise ContextLimit(
                f"Ollama rejected payload with HTTP 413 (model={self.model})",
                provider='ollama',
            ) from exc
        raise ProviderResponseError(str(exc), response_code=status or 0, provider='ollama') from exc

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → Ollama /api/chat → ProviderApiResponse."""
        url = f"{self.host}/api/chat"
        api_messages = _ollama_convert_messages(dto.messages)
        payload = self._build_payload(dto.system, api_messages, dto.tools, dto.thinking_mode)

        start = time.time()
        try:
            resp = requests.post(
                url, json=payload,
                headers=self._user_agent(),
                timeout=PROVIDER_CALL_TIMEOUT_S,
            )
            resp.raise_for_status()
            parsed = _parse_chat_response(resp.json(), self.model)
            parsed.latency_ms = int((time.time() - start) * 1000)
            return parsed
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError(str(exc), provider='ollama') from exc
        except requests.exceptions.ConnectionError as exc:
            raise ProviderResponseError(str(exc), response_code=0, provider='ollama') from exc
        except requests.exceptions.HTTPError as exc:
            self._handle_http_error(exc)
            raise  # pragma: no cover — _handle_http_error always raises

    def get_context_limit(self) -> int | None:
        """Query Ollama for model's context window size, cached per-instance.

        Returns None when the query fails or the key is absent — Ollama models
        vary too much for a safe conservative default.
        """
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit
        raw_limit: int | None = None
        try:
            resp = requests.post(
                f"{self.host}/api/show",
                json={"name": self.model},
                timeout=5,
                headers=self._user_agent(),
            )
            if resp.ok:
                model_info = resp.json().get('model_info', {})
                for key, val in model_info.items():
                    if 'context_length' in key.lower():
                        raw_limit = int(val)
                        break
        except Exception as exc:
            logger.debug("[OllamaClient] Failed to get context limit: %s", exc)

        self._cached_context_limit: int | None = raw_limit
        return self._cached_context_limit

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Heuristic estimate — Ollama models vary too much for a fixed tokeniser."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        api_messages = _ollama_convert_messages(dto.messages)
        payload = self._build_payload(dto.system, api_messages, dto.tools, ThinkingLevel.LOW)
        return estimate_tokens(json.dumps(payload))
