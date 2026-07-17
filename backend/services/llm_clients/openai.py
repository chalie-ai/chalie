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
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, TypeAlias, cast

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

from configs.enums.thinking_level import ThinkingLevel
from contracts.provider_client import ProviderClient
from exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseOverLimitError,
)
from services.llm_clients.thinking_map import (
    OPENAI_COMPATIBLE_NONE_BODY,
    OPENAI_NONE_FALLBACK_EFFORT,
    OPENAI_REASONING_EFFORTS,
)
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
)

logger = logging.getLogger(__name__)

_APP_URL = "https://chalie.ai"
_APP_TITLE = "Chalie"

_COMPLETION_USAGE: TypeAlias = "_openai_mod.types.CompletionUsage"
_CHAT_COMPLETIONS_CREATE: TypeAlias = "Callable[..., _openai_mod.types.chat.ChatCompletion]"


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
        self.platform: str = cast(str, config.get('platform', 'openai'))
    def _get_client(self) -> "_openai_mod.OpenAI":
        from openai import OpenAI  # noqa: PLC0415
        from services.llm_service import _resolve_api_key, _app_user_agent  # noqa: PLC0415
        from services.provider_api import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415
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
        effort = OPENAI_REASONING_EFFORTS.get(level)
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

    def _retry_thinking_rejection(self, client: "_openai_mod.OpenAI", create_kwargs: dict[str, object]) -> "_openai_mod.types.chat.ChatCompletion":
        """Ladder: only when we sent reasoning_effort='none' do we try
        fallback steps; otherwise strip and retry once."""
        from services.llm_service import _is_thinking_rejection  # noqa: PLC0415
        if create_kwargs.get('reasoning_effort') == 'none':
            # Step 1: retry with minimal effort, no extra_body
            logger.info(
                "[THINKING] native flag rejected by provider=openai model=%s — retried with minimal",
                self.model,
            )
            fallback1 = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
            fallback1['reasoning_effort'] = OPENAI_NONE_FALLBACK_EFFORT
            try:
                return cast(_CHAT_COMPLETIONS_CREATE, client.chat.completions.create)(**fallback1)
            except Exception as retry_exc:
                if _is_thinking_rejection(retry_exc, fallback1):
                    # Step 2: retry bare — no reasoning_effort, no extra_body
                    logger.info(
                        "[THINKING] native flag rejected by provider=openai model=%s — retried without",
                        self.model,
                    )
                    fallback2 = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
                    return cast(_CHAT_COMPLETIONS_CREATE, client.chat.completions.create)(**fallback2)
                raise
        # Single bare-strip retry for other reasoning_effort values
        logger.info(
            "[THINKING] native flag rejected by provider=openai model=%s — retried without",
            self.model,
        )
        fallback = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
        return cast(_CHAT_COMPLETIONS_CREATE, client.chat.completions.create)(**fallback)

    def _handle_bad_request(self, client: "_openai_mod.OpenAI", create_kwargs: dict[str, object], exc: Exception) -> "_openai_mod.types.chat.ChatCompletion":
        """Map a BadRequestError/APIError: context-length rejection, thinking-retry ladder, or generic error."""
        from services.llm_service import _is_thinking_rejection  # noqa: PLC0415
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
            return self._retry_thinking_rejection(client, create_kwargs)
        status = getattr(exc, 'status_code', 0)
        raise ProviderResponseError(str(exc), response_code=status, provider='openai') from exc

    def _invoke_create(self, client: "_openai_mod.OpenAI", create_kwargs: dict[str, object]) -> "_openai_mod.types.chat.ChatCompletion":
        """Call chat.completions.create, mapping SDK errors with a thinking-retry fallback."""
        import openai as openai_mod  # noqa: PLC0415
        try:
            return cast(_CHAT_COMPLETIONS_CREATE, client.chat.completions.create)(**create_kwargs)
        except openai_mod.RateLimitError as exc:
            raise RateLimitError(str(exc), provider='openai') from exc
        except openai_mod.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider='openai') from exc
        except (openai_mod.BadRequestError, openai_mod.APIError) as exc:
            return self._handle_bad_request(client, create_kwargs, exc)

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → OpenAI Chat Completions API → ProviderApiResponse."""
        from services.llm_service import _strip_think_blocks  # noqa: PLC0415

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
            # Vendor-extension disable param for openai_compatible endpoints
            # whose reasoning toggle is a body field (vLLM/Z.ai style). Sent
            # only for NONE — other levels use reasoning_effort above.
            if self.platform == 'openai_compatible' and dto.thinking_mode is ThinkingLevel.NONE:
                create_kwargs['extra_body'] = OPENAI_COMPATIBLE_NONE_BODY

        response = self._invoke_create(client, create_kwargs)
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
            cast(_COMPLETION_USAGE, response.usage).prompt_tokens,
            cast(_COMPLETION_USAGE, response.usage).completion_tokens,
            latency_ms,
            finish_reason,
            f" tools={len(tool_calls)}" if tool_calls else "",
        )

        return ProviderApiResponse(
            text=text,
            model=response.model,
            provider='openai',
            tokens_input=cast(_COMPLETION_USAGE, response.usage).prompt_tokens,
            tokens_output=cast(_COMPLETION_USAGE, response.usage).completion_tokens,
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
