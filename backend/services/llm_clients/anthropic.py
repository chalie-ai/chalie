"""
Anthropic thin client — transforms ProviderApiRequest → Anthropic Messages API
and parses back to ProviderApiResponse.

Native size-rejection signal: HTTP 413 → ResponseOverLimitError.
UNVERIFIED: The Anthropic SDK may surface a 413 as anthropic.APIStatusError
with status_code 413.  The existing code does NOT have a 413 handler for
Anthropic (only RateLimitError is caught); this client adds the catch.
Flag: # UNVERIFIED: Anthropic 413 — SDK exception class not confirmed from
existing code.  anthropic.APIStatusError with status_code 413 is the most
likely surface; left as a best-effort catch on status_code attribute.

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens,
_app_user_agent, _resolve_api_key — utilities kept in llm_service during migration).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import time
from typing import ClassVar, Optional

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

_THINKING_BUDGETS: dict[str, int] = {
    ThinkingLevel.MEDIUM.value: 4096,
    ThinkingLevel.HIGH.value: 16384,
}


def _anthropic_convert_messages(messages: list) -> list:
    """Convert normalised messages to Anthropic's content-block format.

    Handles regular messages, assistant+tool_calls, tool results, and
    inline images (base64).
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
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
            continue
        else:
            img = msg.get('image')
            if img:
                blocks = []
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
                result.append({"role": msg['role'], "content": blocks})
            else:
                result.append(msg)
        i += 1
    return result


def _parse_content_blocks(content) -> tuple[str, Optional[list], Optional[str]]:
    """Extract (text, tool_calls, thinking_block) from an Anthropic response content list."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    thinking_block: Optional[str] = None
    for block in (content or []):
        block_type = getattr(block, 'type', None)
        if block_type == 'tool_use':
            tool_calls.append({'id': block.id, 'name': block.name, 'input': block.input})
        elif block_type == 'thinking':
            thinking_block = getattr(block, 'thinking', None)
            logger.debug("[AnthropicClient] thinking block present")
        elif hasattr(block, 'text') and block.text:
            text_parts.append(block.text)
    return '\n'.join(text_parts), tool_calls or None, thinking_block


class AnthropicClient(ProviderClient):
    """Anthropic Claude API thin client."""

    CONTENT_FIELD_LABEL: ClassVar[str] = "content[].text"

    def __init__(self, config: dict) -> None:
        """Initialise from provider config dict (platform, model, api_key, timeout)."""
        self._config = config
        self.model: str = config.get('model', 'claude-haiku-4-5-20251001')
        self._timeout: int = config.get('timeout', 120)

    def _get_client(self):
        import anthropic
        from services.llm_service import _resolve_api_key, _app_user_agent  # noqa: PLC0415
        return anthropic.Anthropic(
            api_key=_resolve_api_key(self._config),
            timeout=self._timeout,
            default_headers={"User-Agent": _app_user_agent()},
        )

    def _thinking_native(self, level: ThinkingLevel, max_tokens: int) -> dict:
        """Return extra kwargs for the Anthropic thinking flag, or empty dict."""
        value = level.value
        if level == ThinkingLevel.MAX:
            budget = max_tokens
            logger.info(
                "[THINKING] native flag passed: provider=anthropic mode=max model=%s budget=%d",
                self.model, budget,
            )
            return {'thinking': {'type': 'enabled', 'budget_tokens': budget}}
        if value in _THINKING_BUDGETS:
            budget = _THINKING_BUDGETS[value]
            logger.info(
                "[THINKING] native flag passed: provider=anthropic mode=%s model=%s",
                value, self.model,
            )
            return {'thinking': {'type': 'enabled', 'budget_tokens': budget}}
        return {}

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → Anthropic Messages API → ProviderApiResponse."""
        import anthropic  # noqa: PLC0415
        from services.llm_service import _call_with_retry  # noqa: PLC0415 -- retry helper

        # Compute max_tokens using the window already known to Providers.send().
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

        create_kwargs: dict = {
            'model': self.model,
            'max_tokens': max_out,
            'system': system,
            'messages': _anthropic_convert_messages(dto.messages),
            **({'tools': dto.tools} if dto.tools else {}),
            **self._thinking_native(dto.thinking_mode, max_out),
        }

        def _call():
            try:
                return client.messages.create(**create_kwargs)
            except anthropic.RateLimitError as exc:
                retry_after = None
                if hasattr(exc, 'response') and exc.response is not None:
                    ra = exc.response.headers.get('retry-after')
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
                raise RateLimitError(str(exc), retry_after=retry_after, provider='anthropic') from exc
            except (anthropic.BadRequestError, anthropic.APIError) as exc:
                # UNVERIFIED: Anthropic 413 surface — checking status_code attribute
                # (anthropic.APIStatusError subclass). This is best-effort.
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
                    return client.messages.create(
                        **{k: v for k, v in create_kwargs.items() if k != 'thinking'}
                    )
                raise ProviderResponseError(str(exc), response_code=status or 0, provider='anthropic') from exc

        response = _call_with_retry(_call)
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
        """Estimate tokens via Anthropic's server-side API, falling back to heuristic."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        try:
            client = self._get_client()
            kwargs: dict = {
                'model': self.model,
                'messages': _anthropic_convert_messages(dto.messages),
            }
            if dto.system:
                kwargs['system'] = dto.system
            if dto.tools:
                kwargs['tools'] = dto.tools
            result = client.messages.count_tokens(**kwargs)
            return result.input_tokens
        except Exception as exc:
            logger.debug("[AnthropicClient] count_tokens API failed, using estimate: %s", exc)
        body = json.dumps({
            'model': self.model,
            'system': dto.system,
            'messages': _anthropic_convert_messages(dto.messages),
            **({'tools': dto.tools} if dto.tools else {}),
        }, default=str)
        return estimate_tokens(body)
