"""
OpenAI thin client — covers both 'openai' and 'openai_compatible' platforms.

The openai_compatible path differs only in that a 'host' (base_url) is set in
the config; the SDK call-path is identical.

Native size-rejection signal:
  HTTP 400 with error.code == 'context_length_exceeded' → ContextLimit.
  Confirmed from existing code: openai_mod.BadRequestError is caught in
  _call_completions; the 'context_length_exceeded' code is the canonical OpenAI
  signal (https://platform.openai.com/docs/guides/error-codes).
  That code is OpenAI's own convention, and the openai_compatible hosts sharing
  this client are under no obligation to use it, so the message prose is matched
  as well (``is_token_limit_message``) — otherwise a size rejection from a
  self-hosted server would surface as a generic 400 and never compact.

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
    ContextLimit,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from services.llm_clients.context_window import default_window_for_model
from services.llm_clients.thinking_map import (
    OPENAI_COMPATIBLE_NONE_BODY,
    OPENAI_NONE_FALLBACK_EFFORT,
    OPENAI_REASONING_EFFORTS,
)
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
    is_token_limit_message,
)

logger = logging.getLogger(__name__)

_APP_URL = "https://chalie.ai"
_APP_TITLE = "Chalie"

_COMPLETION_USAGE: TypeAlias = "_openai_mod.types.CompletionUsage"

# Published context windows for official OpenAI models. api.openai.com/v1/models
# returns no size information at all, so for platform='openai' the documented
# figure IS the measurement — there is nothing else to read. Matched longest
# prefix first, because 'gpt-4o'/'gpt-4.1' must not be caught by 'gpt-4'.
# Deliberately NOT used for platform='openai_compatible': a third-party or
# self-hosted host behind the same API serves whatever it likes under any name,
# so that path pings the host to confirm it answers and falls back to a
# family-based default (see _default_window_for_model).
_OPENAI_PUBLISHED_WINDOWS: tuple[tuple[str, int], ...] = (
    ('gpt-3.5-turbo', 16_385),
    ('gpt-4-turbo', 128_000),
    ('gpt-4.1', 1_047_576),
    ('gpt-4o', 128_000),
    ('gpt-4', 8_192),
    ('gpt-5', 400_000),
    ('o1', 200_000),
    ('o3', 200_000),
    ('o4', 200_000),
)

def _published_openai_window(model: str) -> int | None:
    """Documented window for an official OpenAI model slug, or None if unknown.

    Unknown means unknown: a new slug gets a loud failure upstream rather than a
    neighbouring family's number, which could be wrong by two orders of magnitude.
    """
    slug = (model or '').lower()
    for prefix, window in sorted(
        _OPENAI_PUBLISHED_WINDOWS, key=lambda pair: len(pair[0]), reverse=True,
    ):
        if slug.startswith(prefix):
            return window
    return None


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

    def _invoke_create(self, client: "_openai_mod.OpenAI", create_kwargs: dict[str, object]) -> "_openai_mod.types.chat.ChatCompletion":
        """Call chat.completions.create, mapping SDK errors with a thinking-retry fallback."""
        import openai as openai_mod  # noqa: PLC0415
        from services.llm_service import _is_thinking_rejection  # noqa: PLC0415
        try:
            return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**create_kwargs)
        except openai_mod.RateLimitError as exc:
            raise RateLimitError(str(exc), provider='openai') from exc
        except openai_mod.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider='openai') from exc
        except (openai_mod.BadRequestError, openai_mod.APIError) as exc:
            # Confirmed: OpenAI sends HTTP 400 with error.code == 'context_length_exceeded'.
            # That code is OpenAI's own — the openai_compatible hosts sharing this
            # client (llama.cpp, vLLM, third-party gateways) rarely send it, so the
            # message prose is checked too or their size rejections never compact.
            err_body = getattr(exc, 'body', None) or {}
            err_code = (err_body.get('error') or {}).get('code', '') if isinstance(err_body, dict) else ''
            if (
                err_code == 'context_length_exceeded'
                or 'context_length_exceeded' in str(exc).lower()
                or is_token_limit_message(str(exc))
            ):
                raise ContextLimit(
                    f"Provider rejected payload (context length): {exc}",
                    provider=cast(str, self._config.get('platform') or 'openai'),
                    model=self.model,
                ) from exc
            if _is_thinking_rejection(exc, create_kwargs):
                # Ladder: only when we sent reasoning_effort='none' do we try
                # fallback steps; otherwise strip and retry once.
                if create_kwargs.get('reasoning_effort') == 'none':
                    # Step 1: retry with minimal effort, no extra_body
                    logger.info(
                        "[THINKING] native flag rejected by provider=openai model=%s — retried with minimal",
                        self.model,
                    )
                    fallback1 = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
                    fallback1['reasoning_effort'] = OPENAI_NONE_FALLBACK_EFFORT
                    try:
                        return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**fallback1)
                    except Exception as retry_exc:
                        if _is_thinking_rejection(retry_exc, fallback1):
                            # Step 2: retry bare — no reasoning_effort, no extra_body
                            logger.info(
                                "[THINKING] native flag rejected by provider=openai model=%s — retried without",
                                self.model,
                            )
                            fallback2 = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
                            return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**fallback2)
                        raise
                else:
                    # Single bare-strip retry for other reasoning_effort values
                    logger.info(
                        "[THINKING] native flag rejected by provider=openai model=%s — retried without",
                        self.model,
                    )
                    fallback = {k: v for k, v in create_kwargs.items() if k not in ('reasoning_effort', 'extra_body')}
                    return cast("Callable[..., _openai_mod.types.chat.ChatCompletion]", client.chat.completions.create)(**fallback)
            status = getattr(exc, 'status_code', 0)
            raise ProviderResponseError(str(exc), response_code=status, provider='openai') from exc

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

        # Defensive reads for provider-reported telemetry that the SDK does not
        # expose through typed attributes.
        #
        # Real OpenAI: usage.prompt_tokens_details.cached_tokens (int | None).
        # llama.cpp OpenAI-compatible server: timings.cache_n (int | None).
        _prompt_details = getattr(response.usage, 'prompt_tokens_details', None)
        tokens_cache_read = getattr(_prompt_details, 'cached_tokens', None) if _prompt_details else None
        timings = getattr(response, 'timings', None)
        if isinstance(timings, dict):
            if tokens_cache_read is None:
                _cache_n = timings.get('cache_n')
                if isinstance(_cache_n, int):
                    tokens_cache_read = _cache_n
            prefill_ms = timings.get('prompt_ms')
            decode_ms = timings.get('predicted_ms')
        else:
            prefill_ms = None
            decode_ms = None

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
            tokens_cache_read=tokens_cache_read,
            latency_ms=latency_ms,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            response_code=200,
        )

    def get_context_limit(self) -> int | None:
        """Determine the model's context window, cached per-instance.

        A hosted (openai_compatible) provider is pinged: a tiny one-shot
        completion both proves the host answers and yields a window — read from
        a size field on the response if the server volunteers one, otherwise a
        family-based default. Official OpenAI (no host) falls to the published
        table, since api.openai.com serves no size information anywhere.

        Returns None only when a hosted provider cannot be reached at all, so
        the caller leaves the column unset and the next send re-probes. Never
        raises: a provider must stay creatable while its host is briefly down.
        """
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit

        # Official OpenAI publishes no window on any endpoint, so the documented
        # table is the authoritative answer where it has one. A slug it does not
        # know (a model newer than the table) still gets measured — it falls
        # through to the same ping every hosted provider uses.
        window = None if self._config.get('host') else _published_openai_window(self.model)
        if window is None:
            window = self._probe_via_ping()

        self._cached_context_limit: int | None = window
        return self._cached_context_limit

    def _probe_via_ping(self) -> int | None:
        """Confirm a hosted provider answers and derive its context window.

        Sends a 5-token one-shot completion. A reply with a readable
        ``usage.prompt_tokens`` proves the host and model are live; the window
        is then taken from a size field on the response if one is present, else
        a family-based default. Returns None (never raises) when the host cannot
        be reached, so the row is left unset for the next send to re-probe.
        """
        try:
            client = self._get_client()
            create = cast(
                "Callable[..., _openai_mod.types.chat.ChatCompletion]",
                client.chat.completions.create,
            )
            resp = create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": "Do NOT think, answer immediately with 'pong'.",
                }],
                max_tokens=5,
            )
        except Exception as exc:
            logger.warning(
                "[OpenAIClient] Context-window ping failed for model=%s: %s",
                self.model, exc,
            )
            return None

        usage = getattr(resp, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage else None
        if not isinstance(prompt_tokens, int):
            logger.warning(
                "[OpenAIClient] Context-window ping for model=%s returned no usage — "
                "treating the host as unusable, leaving the window unset",
                self.model,
            )
            return None

        window = self._window_from_response(resp)
        source = 'response' if window is not None else 'family-default'
        if window is None:
            window = default_window_for_model(self.model)
        logger.info(
            "[OpenAIClient] Context-window ping ok for model=%s (prompt_tokens=%d) — window=%d (%s)",
            self.model, prompt_tokens, window, source,
        )
        return window

    @staticmethod
    def _window_from_response(resp: object) -> int | None:
        """A window the server volunteered on the completion, or None.

        Standard OpenAI responses carry no size; some self-hosted servers attach
        one under ``model_info`` or a top-level field. Read defensively — the
        SDK never types these, so they arrive as untyped extras or not at all.
        """
        model_info = getattr(resp, 'model_info', None)
        if isinstance(model_info, dict):
            for key in ('context_length', 'n_ctx', 'max_context_length'):
                val = model_info.get(key)
                if isinstance(val, int) and val > 0:
                    return val
        for attr in ('context_length', 'n_ctx'):
            val = getattr(resp, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

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
