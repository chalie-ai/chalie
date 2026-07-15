"""
Anthropic thin client — transforms ProviderApiRequest → Anthropic Messages API
and parses back to ProviderApiResponse.

Native size-rejection signal: HTTP 413 → ResponseOverLimitError.
VERIFIED: anthropic._exceptions.RequestTooLargeError has class-level
status_code == 413 and is a subclass of anthropic.APIError.  The catch
``except (anthropic.BadRequestError, anthropic.APIError)`` with
``getattr(exc, 'status_code', None) == 413`` therefore correctly intercepts
it.  Confirmed via:
  python3 -c "import anthropic._exceptions as e, anthropic;
              print(e.RequestTooLargeError.status_code,
                    issubclass(e.RequestTooLargeError, anthropic.APIError))"
  → 413 True

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens,
_app_user_agent, _resolve_api_key — utilities kept in llm_service during migration).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, cast

if TYPE_CHECKING:
    from typing import Protocol

    import anthropic as _anthropic_mod

    _Msg = dict[str, object]

    class _Block(Protocol):
        id: object
        name: object
        input: object
        text: str

from configs.enums.thinking_level import ThinkingLevel
from contracts.provider_client import ProviderClient
from exceptions import (
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
    ResponseOverLimitError,
)
from services.llm_clients.thinking_map import ANTHROPIC_NONE_THINKING, ANTHROPIC_THINKING_BUDGETS
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
)

logger = logging.getLogger(__name__)


def _assistant_tool_block(msg: "_Msg") -> "_Msg":
    content: "list[_Msg]" = []
    text = msg.get('content', '')
    if text:
        content.append({"type": "text", "text": text})
    for tc in cast("list[_Msg]", msg['tool_calls']):
        content.append({
            "type": "tool_use",
            "id": tc['id'],
            "name": tc['name'],
            "input": tc['input'],
        })
    return {"role": "assistant", "content": content}


def _image_block(msg: "_Msg") -> "_Msg":
    img = cast("_Msg", msg.get('image'))
    blocks: "list[_Msg]" = []
    if msg.get('content'):
        blocks.append({"type": "text", "text": msg['content']})
    blocks.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": img['mime_type'],
            "data": img['data'],
        },
    })
    return {"role": msg['role'], "content": blocks}


def _anthropic_convert_messages(messages: "list[_Msg]") -> "list[_Msg]":
    """Convert normalised messages to Anthropic's content-block format.

    Handles regular messages, assistant+tool_calls, tool results, and
    inline images (base64).
    """
    result: "list[_Msg]" = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append(_assistant_tool_block(msg))
        elif msg['role'] == 'tool':
            tool_results: "list[_Msg]" = []
            while i < len(messages) and messages[i]['role'] == 'tool':
                tm = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tm['tool_call_id'],
                    "content": tm.get('content', ''),
                })
                i += 1
            result.append({"role": "user", "content": tool_results})
            continue
        elif msg.get('image'):
            result.append(_image_block(msg))
        else:
            result.append(msg)
        i += 1
    return result


def _parse_content_blocks(content: Sequence[object]) -> tuple[str, Optional[list[dict[str, object]]], Optional[str]]:
    """Extract (text, tool_calls, thinking_block) from an Anthropic response content list."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    thinking_block: Optional[str] = None
    for block in (content or []):
        block_type = getattr(block, 'type', None)
        if block_type == 'tool_use':
            tool_calls.append({'id': cast("_Block", block).id, 'name': cast("_Block", block).name, 'input': cast("_Block", block).input})
        elif block_type == 'thinking':
            thinking_block = getattr(block, 'thinking', None)
            logger.debug("[AnthropicClient] thinking block present")
        elif hasattr(block, 'text') and cast("_Block", block).text:
            text_parts.append(cast("_Block", block).text)
    return '\n'.join(text_parts), tool_calls or None, thinking_block


class AnthropicClient(ProviderClient):
    """Anthropic Claude API thin client."""

    CONTENT_FIELD_LABEL: ClassVar[str] = "content[].text"

    def __init__(self, config: dict[str, object]) -> None:
        """Initialise from provider config dict (platform, model, api_key)."""
        self._config = config
        self.model: str = cast(str, config.get('model', 'claude-haiku-4-5-20251001'))

    def _get_client(self) -> "_anthropic_mod.Anthropic":
        import anthropic
        from services.llm_service import _resolve_api_key, _app_user_agent  # noqa: PLC0415
        from services.provider_api import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415
        return anthropic.Anthropic(
            api_key=_resolve_api_key(self._config),
            timeout=PROVIDER_CALL_TIMEOUT_S,
            default_headers={"User-Agent": _app_user_agent()},
        )

    def _thinking_native(self, level: ThinkingLevel, max_tokens: int) -> dict[str, object]:
        """Return extra kwargs for the Anthropic thinking flag, or empty dict.

        NONE → {'type':'disabled'} (explicit off).
        MEDIUM/HIGH → enabled with budget from ANTHROPIC_THINKING_BUDGETS.
        MAX → enabled with budget = request max_tokens (special case).
        LOW → absent (no flag).
        """
        if level == ThinkingLevel.NONE:
            logger.info(
                "[THINKING] native flag passed: provider=anthropic mode=none model=%s",
                self.model,
            )
            return {'thinking': ANTHROPIC_NONE_THINKING}
        if level == ThinkingLevel.MAX:
            budget = max_tokens
            logger.info(
                "[THINKING] native flag passed: provider=anthropic mode=max model=%s budget=%d",
                self.model, budget,
            )
            return {'thinking': {'type': 'enabled', 'budget_tokens': budget}}
        if level in ANTHROPIC_THINKING_BUDGETS:
            budget = ANTHROPIC_THINKING_BUDGETS[level]
            logger.info(
                "[THINKING] native flag passed: provider=anthropic mode=%s model=%s",
                level.value, self.model,
            )
            return {'thinking': {'type': 'enabled', 'budget_tokens': budget}}
        return {}

    def _invoke_create(self, client: "_anthropic_mod.Anthropic", create_kwargs: dict[str, object]) -> "_anthropic_mod.types.Message":
        """Call messages.create, mapping SDK errors to provider errors with a thinking-retry fallback."""
        import anthropic  # noqa: PLC0415
        try:
            return cast("Callable[..., _anthropic_mod.types.Message]", client.messages.create)(**create_kwargs)
        except anthropic.RateLimitError as exc:
            raise RateLimitError(str(exc), provider='anthropic') from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc), provider='anthropic') from exc
        except (anthropic.BadRequestError, anthropic.APIError) as exc:
            # VERIFIED: anthropic._exceptions.RequestTooLargeError subclasses
            # anthropic.APIError with class-level status_code == 413.
            # getattr(exc, 'status_code', None) returns 413 for that class.
            status = getattr(exc, 'status_code', None)
            if status == 413:
                raise ResponseOverLimitError(
                    f"Anthropic rejected payload (HTTP 413): {exc}",
                    response_code=413, provider='anthropic',
                ) from exc
            if 'thinking' in str(exc).lower() and 'thinking' in create_kwargs:
                logger.info(
                    "[THINKING] native flag rejected by provider=anthropic model=%s — retried without",
                    self.model,
                )
                return cast("Callable[..., _anthropic_mod.types.Message]", client.messages.create)(
                    **{k: v for k, v in create_kwargs.items() if k != 'thinking'}
                )
            raise ProviderResponseError(str(exc), response_code=status or 0, provider='anthropic') from exc

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → Anthropic Messages API → ProviderApiResponse."""
        # Compute max_tokens using the window already known to ProviderService.send().
        # The window is not available here; use the DTO's explicit value if set,
        # else fall back to a safe ceiling matching the old _MAX_TOKENS.
        window = self.get_context_limit()
        max_out = dto.resolve_max_tokens(window)

        client = self._get_client()
        start = time.time()

        system = (
            [{"type": "text", "text": dto.system, "cache_control": {"type": "ephemeral"}}]
            if dto.cache_prefix
            else dto.system
        )

        create_kwargs: dict[str, object] = {
            'model': self.model,
            'max_tokens': max_out,
            'system': system,
            'messages': _anthropic_convert_messages(dto.messages),
            **({'tools': dto.tools} if dto.tools else {}),
            **self._thinking_native(dto.thinking_mode, max_out),
        }

        response = self._invoke_create(client, create_kwargs)
        latency_ms = int((time.time() - start) * 1000)
        text, tool_calls, thinking_block = _parse_content_blocks(response.content)
        stop_reason = response.stop_reason

        logger.info(
            "[AnthropicClient] model=%s tokens=%d+%d latency=%dms stop=%s%s",
            response.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            latency_ms,
            stop_reason,
            f" tools={len(tool_calls)}" if tool_calls else "",
        )

        return ProviderApiResponse(
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
            thinking_block=thinking_block,
            response_code=200,
        )

    def get_context_limit(self) -> int:
        """All Claude 3+ models support 200k context."""
        return 200_000

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate the token cost of a request using a local heuristic.

        Uses the same strategy as the pre-refactor _over_cap path: serialise
        the request body and apply estimate_tokens() (no live API call).  This
        matches the old behaviour — the old providers._over_cap called
        build_request_body() + estimate_tokens(), never count_tokens().

        The Anthropic count_tokens API (exact but a live round-trip) is NOT
        called here.  That would double the number of API calls per ACT
        iteration and introduce a new failure mode (rate-limit on count_tokens
        before the main send).  Pre-flight over-cap checks only need a safe
        heuristic, not an exact count.
        """
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        body = json.dumps({
            'model': self.model,
            'system': dto.system,
            'messages': _anthropic_convert_messages(dto.messages),
            **({'tools': dto.tools} if dto.tools else {}),
        }, default=str)
        return estimate_tokens(body)
