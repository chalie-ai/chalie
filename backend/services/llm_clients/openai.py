"""
OpenAI thin client — covers both 'openai' and 'openai_compatible' platforms.

The openai_compatible path differs only in that a 'host' (base_url) is set in
the config; the SDK call-path is identical.

Native size-rejection signal:
  HTTP 400 with error.code == 'context_length_exceeded' → ResponseOverLimitError.
  Confirmed from existing code: openai_mod.BadRequestError is caught in
  _call_completions; the 'context_length_exceeded' code is the canonical OpenAI
  signal (https://platform.openai.com/docs/guides/error-codes).

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens,
_app_user_agent, _resolve_api_key, _strip_think_blocks — utilities).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, cast

if TYPE_CHECKING:
    from typing import Protocol

    import openai as _openai_mod

    _Msg = dict[str, object]

    class _ToolFunction(Protocol):
        arguments: "str | None"
        name: str

    class _ToolCall(Protocol):
        id: str
        function: _ToolFunction

    class _ChatMessage(Protocol):
        content: "str | None"
        tool_calls: "list[_ToolCall] | None"

from services.llm_clients.base import ProviderClient
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
    RateLimitError,
    ResponseOverLimitError,
    ProviderResponseError,
    ThinkingLevel,
)

logger = logging.getLogger(__name__)

_APP_URL = "https://chalie.ai"
_APP_TITLE = "Chalie"

_REASONING_EFFORTS: dict[str, str] = {
    ThinkingLevel.MEDIUM.value: 'medium',
    ThinkingLevel.HIGH.value: 'high',
    ThinkingLevel.MAX.value: 'high',  # OpenAI has no 'max'; map to highest available
}


def _openai_convert_messages(messages: "list[_Msg]") -> "list[_Msg]":
    """Convert normalised messages to OpenAI format."""
    result: "list[_Msg]" = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append({
                "role": "assistant",
                "content": msg.get('content') or None,
                "tool_calls": [
                    {
                        "id": cast(str, tc['id']),
                        "type": "function",
                        "function": {
                            "name": cast(str, tc['name']),
                            "arguments": json.dumps(tc['input']),
                        },
                    }
                    for tc in cast("list[_Msg]", msg['tool_calls'])
                ],
            })
        elif msg['role'] == 'tool':
            result.append({
                "role": "tool",
                "tool_call_id": msg['tool_call_id'],
                "content": msg.get('content', ''),
            })
        else:
            img = cast("_Msg", msg.get('image'))
            if img:
                parts: "list[_Msg]" = []
                if msg.get('content'):
                    parts.append({"type": "text", "text": msg['content']})
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"},
                })
                result.append({"role": msg['role'], "content": parts})
            else:
                result.append(msg)
    return result


class OpenAIClient(ProviderClient):
    CONTENT_FIELD_LABEL: ClassVar[str] = "choices[].message.content"

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self.model: str = cast(str, config.get('model', 'gpt-4o-mini'))
        self._format: str = cast(str, config.get('format', 'text'))

    def _get_client(self) -> "_openai_mod.OpenAI":
        from openai import OpenAI  # noqa: PLC0415
        from services.llm_service import _resolve_api_key, _app_user_agent  # noqa: PLC0415
        from services.providers import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415
        kwargs: dict[str, object] = {
            'api_key': _resolve_api_key(self._config),
            'timeout': PROVIDER_CALL_TIMEOUT_S,
            'default_headers': {
                "HTTP-Referer": _APP_URL,
                "X-Title": _APP_TITLE,
                "User-Agent": _app_user_agent(),
            },
        }
        base_url = self._config.get('host')
        if base_url:
            kwargs['base_url'] = base_url
        return cast("Callable[..., _openai_mod.OpenAI]", OpenAI)(**kwargs)

    def _thinking_native(self, level: ThinkingLevel) -> Optional[str]:
        """Return the reasoning_effort string or None when thinking is off."""
        effort = _REASONING_EFFORTS.get(level.value)
        if effort:
            logger.info(
                "[THINKING] native flag passed: provider=openai mode=%s model=%s",
                level.value, self.model,
            )
        return effort

    def _build_openai_tools(self, tools: "list[_Msg]") -> "list[_Msg]":
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

    @staticmethod
    def _parse_tool_calls(msg: "_ChatMessage") -> "Optional[list[_Msg]]":
        if not msg.tool_calls:
            return None
        calls: "list[_Msg]" = []
        for tc in msg.tool_calls:
            try:
                parsed: object = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                parsed = {}
            calls.append({'id': tc.id, 'name': tc.function.name, 'input': parsed})
        return calls

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → OpenAI Chat Completions API → ProviderApiResponse."""
        import openai as openai_mod  # noqa: PLC0415
        from services.llm_service import _strip_think_blocks, _is_thinking_rejection  # noqa: PLC0415
        from services.provider_api import ProviderTimeoutError  # noqa: PLC0415

        client = self._get_client()
        start = time.time()

        api_messages = _openai_convert_messages(dto.messages)
        create_kwargs: dict[str, object] = {
            'model': self.model,
            'messages': [{"role": "system", "content": dto.system}] + api_messages,
        }
        if dto.tools:
            create_kwargs['tools'] = self._build_openai_tools(dto.tools)

        effort = self._thinking_native(dto.thinking_mode)
        if effort:
            create_kwargs['reasoning_effort'] = effort

        def _call() -> "_openai_mod.types.chat.ChatCompletion":
            try:
                return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**create_kwargs)
            except openai_mod.RateLimitError as exc:
                raise RateLimitError(str(exc), provider='openai') from exc
            except openai_mod.APITimeoutError as exc:
                raise ProviderTimeoutError(str(exc), provider='openai') from exc
            except (openai_mod.BadRequestError, openai_mod.APIError) as exc:
                # Confirmed: OpenAI sends HTTP 400 with error.code == 'context_length_exceeded'
                err_body = getattr(exc, 'body', None) or {}
                err_code = (err_body.get('error') or {}).get('code', '') if isinstance(err_body, dict) else ''
                if err_code == 'context_length_exceeded' or 'context_length_exceeded' in str(exc).lower():
                    status = getattr(exc, 'status_code', 400)
                    raise ResponseOverLimitError(
                        f"OpenAI rejected payload (context_length_exceeded): {exc}",
                        response_code=status, provider='openai',
                    ) from exc
                if _is_thinking_rejection(exc, create_kwargs):
                    logger.info(
                        "[THINKING] native flag rejected by provider=openai model=%s — retried without",
                        self.model,
                    )
                    fallback = {k: v for k, v in create_kwargs.items() if k != 'reasoning_effort'}
                    return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**fallback)
                status = getattr(exc, 'status_code', 0)
                raise ProviderResponseError(str(exc), response_code=status, provider='openai') from exc

        response = _call()
        latency_ms = int((time.time() - start) * 1000)

        msg = response.choices[0].message
        text = _strip_think_blocks(msg.content or "")
        finish_reason = response.choices[0].finish_reason
        tool_calls = self._parse_tool_calls(cast("_ChatMessage", msg))

        _completion_details = getattr(response.usage, 'completion_tokens_details', None)
        _reasoning = getattr(_completion_details, 'reasoning_tokens', None) if _completion_details else None

        log_fn = logger.info if (text and text.strip()) or tool_calls else logger.warning
        log_fn(
            "[OpenAIClient] model=%s tokens=%d+%d latency=%dms finish=%s%s",
            response.model,
            cast("_openai_mod.types.CompletionUsage", response.usage).prompt_tokens,
            cast("_openai_mod.types.CompletionUsage", response.usage).completion_tokens,
            latency_ms,
            finish_reason,
            f" tools={len(tool_calls)}" if tool_calls else "",
        )

        return ProviderApiResponse(
            text=text,
            model=response.model,
            provider='openai',
            tokens_input=cast("_openai_mod.types.CompletionUsage", response.usage).prompt_tokens,
            tokens_output=cast("_openai_mod.types.CompletionUsage", response.usage).completion_tokens,
            tokens_thinking=_reasoning,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            response_code=200,
        )

    def get_context_limit(self) -> int:
        """Default 128k for GPT-4 class models."""
        return 128_000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate tokens using tiktoken if available, else heuristic."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        try:
            import tiktoken  # noqa: PLC0415
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding('cl100k_base')
            parts: list[str] = []
            if dto.system:
                parts.append(dto.system)
            for msg in dto.messages:
                parts.append(cast(str, msg.get('content', '') or ''))
                if msg.get('tool_calls'):
                    parts.append(json.dumps(msg['tool_calls'], default=str))
            if dto.tools:
                parts.append(json.dumps(dto.tools, default=str))
            text = '\n'.join(parts)
            overhead = (len(dto.messages) + 1) * 4
            return len(enc.encode(text)) + overhead
        except ImportError:
            logger.debug("[OpenAIClient] tiktoken not installed; falling back to token estimate")
        except Exception as exc:
            logger.debug("[OpenAIClient] tiktoken counting failed: %s", exc)
        parts = [dto.system] + [cast(str, m.get('content', '') or '') for m in dto.messages]
        return estimate_tokens(' '.join(parts))
