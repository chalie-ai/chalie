"""
Gemini thin client — transforms ProviderApiRequest → Google Gemini API.

Native size-rejection signal — VERIFIED (google-genai 1.65.0):
  The google.genai SDK raises ClientError (a subclass of APIError) for all
  4xx responses, and ServerError for all 5xx responses.  There is no
  dedicated "token limit exceeded" exception subclass.

  HTTP codes relevant to Chalie:
    400 → ClientError; exc.code == 400; exc.status == 'INVALID_ARGUMENT'
    429 → ClientError; exc.code == 429; exc.status == 'RESOURCE_EXHAUSTED'
    500/503 → ServerError; exc.code == 500 or 503

  Token-limit rejections arrive as HTTP 400 with status 'INVALID_ARGUMENT'.
  The same HTTP 400 / INVALID_ARGUMENT status covers many unrelated error
  causes (wrong parameter type, unsupported region, etc.), so a bare
  ``exc.code == 400`` check would over-trigger.  The most reliable
  discriminator is ``exc.code == 400 AND exc.status == 'INVALID_ARGUMENT'
  AND any(s in (exc.message or '').lower() for s in _TOKEN_LIMIT_STRINGS)``.

  Residual gap: Gemini may produce a token-limit 400 with a message phrasing
  not in _TOKEN_LIMIT_STRINGS.  If the string-match misses, the exception
  falls through to the bare ``raise`` and propagates to the MessageProcessor,
  which resends — the turn dies without compaction only after retries exhaust.
  This is an improvement over the old GeminiService which had NO token-limit
  catch at all (confirmed: git show 88421fc0^:backend/services/llm_service.py
  shows no size-rejection handler in GeminiService.send_messages).
  The logger.warning on mismatch (see _generate_with_fallback) surfaces misses
  in production logs so the strings can be extended.

  Note: the old _generate_with_fallback checked ``'ResourceExhausted' in ename``
  for rate limits, but the SDK raises ``ClientError`` (not a class named
  ``ResourceExhausted``); ``'429' in str(exc)`` is the reliable catch and is kept.

Depends on: services.provider_api (contract), services.llm_service (estimate_tokens,
_app_user_agent, _resolve_api_key).
Consumed by: services.llm_clients.factory (platform dispatch).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, ClassVar, Optional, cast
from uuid import uuid4

if TYPE_CHECKING:
    from typing import Protocol, TypedDict

    class _GenCfg(TypedDict, total=False):
        system_instruction: str
        response_mime_type: str
        tools: list[object]
        thinking_config: object

    class _GenaiTypes(Protocol):
        def ThinkingConfig(self, **kwargs: object) -> object: ...
        def Tool(self, **kwargs: object) -> object: ...
        def GenerateContentConfig(self, **kwargs: object) -> object: ...
        def Schema(self, **kwargs: object) -> object: ...
        def FunctionDeclaration(self, **kwargs: object) -> object: ...

    class _Genai(Protocol):
        types: _GenaiTypes

        def Client(self, **kwargs: object) -> "_GenaiClient": ...

    class _GenaiModels(Protocol):
        def generate_content(self, **kwargs: object) -> "_GenResponse": ...
        def get(self, *, model: str) -> "_ModelInfo": ...

    class _GenaiClient(Protocol):
        models: _GenaiModels

    class _ModelInfo(Protocol):
        input_token_limit: "int | None"

    class _Candidate(Protocol):
        content: object
        finish_reason: object

    class _GenResponse(Protocol):
        candidates: "list[_Candidate] | None"

    class _FunctionCall(Protocol):
        name: str
        args: dict[str, object]

    class _Part(Protocol):
        text: str
        function_call: "_FunctionCall | None"

    class _Content(Protocol):
        parts: "list[_Part] | None"

from services.llm_clients.base import ProviderClient
from services.provider_api import (
    ProviderApiRequest,
    ProviderApiResponse,
    RateLimitError,
    ResponseOverLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ThinkingLevel,
)

logger = logging.getLogger(__name__)


_THINKING_BUDGETS: dict[str, int] = {
    ThinkingLevel.MEDIUM.value: 4096,
    ThinkingLevel.HIGH.value: 16384,
}

# Message substrings used as a fallback discriminator for Gemini token-limit
# errors.  All token-limit rejections arrive as HTTP 400 / INVALID_ARGUMENT,
# but that same code covers many unrelated causes.  These strings narrow it
# to size-related errors; mismatches fall through and are logged at WARNING
# so the set can be extended if needed.
_TOKEN_LIMIT_STRINGS = frozenset({
    'token limit',
    'context length',
    'too long',
    'maximum context',
    'input too large',
    'request payload size',
    'exceeds the limit',
})


def _gemini_tool_response(msg: dict[str, object]) -> dict[str, object]:
    return {
        "role": "user",
        "parts": [{
            "function_response": {
                "name": msg.get('name', ''),
                "response": {"content": msg.get('content', '')},
            }
        }],
    }


def _gemini_assistant_parts(msg: dict[str, object]) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    text = msg.get('content', '')
    if text:
        parts.append({"text": text})
    for tc in cast(list[dict[str, object]], msg['tool_calls']):
        parts.append({
            "function_call": {"name": tc['name'], "args": tc['input']},
        })
    return {"role": "model", "parts": parts}


def _gemini_plain_parts(msg: dict[str, object], role: str) -> dict[str, object]:
    parts: list[dict[str, object]] = [{"text": msg.get('content', '')}]
    img = msg.get('image')
    if img:
        parts.append({
            "inline_data": {"mime_type": cast(dict[str, object], img)['mime_type'], "data": cast(dict[str, object], img)['data']},
        })
    return {"role": role, "parts": parts}


def _gemini_convert_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert normalised messages to Gemini content format."""
    result: list[dict[str, object]] = []
    for msg in messages:
        role = "model" if msg['role'] == 'assistant' else cast(str, msg['role'])
        if msg['role'] == 'tool':
            result.append(_gemini_tool_response(msg))
        elif msg['role'] == 'assistant' and msg.get('tool_calls'):
            result.append(_gemini_assistant_parts(msg))
        else:
            result.append(_gemini_plain_parts(msg, role))
    return result


def _accumulate_part(part: object, text_parts: list[str], tool_calls: list[dict[str, object]]) -> None:
    """Append text or a tool-call dict from a single Gemini response part."""
    if getattr(part, 'text', None):
        text_parts.append(cast("_Part", part).text)
    fc = getattr(part, 'function_call', None)
    if fc:
        tool_calls.append({
            'id': f"gemini_{cast('_FunctionCall', fc).name}_{uuid4().hex[:8]}",
            'name': cast("_FunctionCall", fc).name,
            'input': dict(cast("_FunctionCall", fc).args) if cast("_FunctionCall", fc).args else {},
        })


class GeminiClient(ProviderClient):
    """Google Gemini API thin client."""

    CONTENT_FIELD_LABEL: ClassVar[str] = "candidates[].content.parts[].text"

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self.model: str = cast(str, config.get('model', 'gemini-2.5-flash'))
        self._format: str = cast(str, config.get('format', 'text'))

    def _get_sdk(self) -> "_Genai":
        try:
            from google import genai  # noqa: PLC0415
            return cast("_Genai", genai)
        except ImportError:
            raise RuntimeError(
                "google-genai package is not installed. Run: pip install google-genai"
            )

    def _get_client(self, genai: "_Genai") -> "_GenaiClient":
        from services.llm_service import _resolve_api_key, _app_user_agent  # noqa: PLC0415
        from services.providers import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415
        return genai.Client(
            api_key=_resolve_api_key(self._config),
            # HttpOptions.timeout is in milliseconds.
            http_options={
                "timeout": PROVIDER_CALL_TIMEOUT_S * 1000,
                "headers": {"User-Agent": _app_user_agent()},
            },
        )

    def _thinking_native(self, genai: "_Genai", level: ThinkingLevel, cfg: "_GenCfg") -> None:
        """Inject thinking_config into cfg for MEDIUM/HIGH/MAX."""
        value = level.value
        if level == ThinkingLevel.MAX:
            # Gemini has no explicit model-ceiling budget; use a large fixed value.
            budget = 32768
            cfg['thinking_config'] = genai.types.ThinkingConfig(thinking_budget=budget)
            logger.info(
                "[THINKING] native flag passed: provider=gemini mode=max model=%s budget=%d",
                self.model, budget,
            )
        elif value in _THINKING_BUDGETS:
            budget = _THINKING_BUDGETS[value]
            cfg['thinking_config'] = genai.types.ThinkingConfig(thinking_budget=budget)
            logger.info(
                "[THINKING] native flag passed: provider=gemini mode=%s model=%s",
                value, self.model,
            )

    def _build_gen_config(self, genai: "_Genai", system: str, tools: Optional[list[dict[str, object]]],
                          thinking_mode: ThinkingLevel) -> "_GenCfg":
        cfg: "_GenCfg" = {'system_instruction': system}
        if self._format == 'json' and not tools:
            cfg['response_mime_type'] = 'application/json'
        if tools:
            cfg['tools'] = [
                genai.types.Tool(function_declarations=[
                    genai.types.FunctionDeclaration(
                        name=cast(str, t['name']),
                        description=cast(str, t.get('description', '')),
                        parameters=t.get('input_schema'),
                    )
                    for t in tools
                ])
            ]
        self._thinking_native(genai, thinking_mode, cfg)
        return cfg

    def _parse_response(self, response: "_GenResponse") -> tuple[str, Optional[list[dict[str, object]]], Optional[str]]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, object]] = []
        candidate = response.candidates[0] if response.candidates else None
        if candidate is not None:
            for part in (cast("_Content", candidate.content).parts or []):
                _accumulate_part(part, text_parts, tool_calls)
        finish_reason = str(candidate.finish_reason) if candidate and candidate.finish_reason else None
        return '\n'.join(text_parts), tool_calls or None, finish_reason

    def _classify_and_raise(self, exc: Exception, gen_cfg: "_GenCfg") -> bool:
        """Map a google-genai exception to a provider error, or signal thinking-fallback.

        Returns ``True`` when the caller should perform the thinking-retry; raises
        a provider error for every other error class.
        """
        exc_code = getattr(exc, 'code', None)
        exc_status = getattr(exc, 'status', None) or ''
        exc_str = str(exc).lower()

        # HTTP 429 — rate limit.
        # SDK raises ClientError with code=429, status='RESOURCE_EXHAUSTED'.
        # Kept '429' in str(exc) as belt-and-suspenders for SDK version drift.
        if exc_code == 429 or exc_status == 'RESOURCE_EXHAUSTED' or '429' in str(exc):
            raise RateLimitError(str(exc), provider='gemini') from exc

        # HTTP 5xx — transient server error.
        if exc_code is not None and exc_code >= 500:
            raise ProviderResponseError(
                f"Gemini server error: {exc}", response_code=exc_code, provider='gemini',
            ) from exc

        # HTTP 400 / INVALID_ARGUMENT — may be a token-limit rejection.
        # Primary: structured code + status confirms this is a 400 INVALID_ARGUMENT.
        # Secondary: message string narrows to size-related errors; a bare
        # code==400 check would over-trigger (covers wrong params, regions, etc.).
        # If the string-match misses, log a WARNING so the set can be extended.
        if exc_code == 400 and exc_status == 'INVALID_ARGUMENT':
            exc_msg = (getattr(exc, 'message', None) or '').lower()
            if any(s in exc_msg for s in _TOKEN_LIMIT_STRINGS):
                raise ResponseOverLimitError(
                    f"Gemini rejected payload (token limit): {exc}",
                    response_code=400, provider='gemini',
                ) from exc
            logger.warning(
                "[GeminiClient] 400 INVALID_ARGUMENT not matched as token-limit "
                "(msg=%r); propagating. Add matching string to _TOKEN_LIMIT_STRINGS "
                "if this is a size rejection.",
                getattr(exc, 'message', str(exc))[:200],
            )
            raise ProviderResponseError(
                f"Gemini 400 INVALID_ARGUMENT: {exc}", response_code=400, provider='gemini',
            ) from exc

        # Thinking fallback: retry without thinking_config on rejection.
        if 'thinking_config' in gen_cfg and (
            'thinking' in exc_str or 'unsupported' in exc_str
        ):
            logger.info(
                "[THINKING] native flag rejected by provider=gemini model=%s — retried without",
                self.model,
            )
            return True
        raise

    def _generate_with_fallback(self, client: "_GenaiClient", genai: "_Genai", contents: object, gen_cfg: "_GenCfg") -> object:
        """Execute generate_content, handling errors and thinking fallback.

        Error discrimination is based on the structured fields of
        google.genai.errors.ClientError / ServerError (verified, google-genai
        1.65.0): exc.code is the HTTP status int; exc.status is the gRPC
        status string.  String-matching on exc.message is used only as a
        secondary discriminator for token-limit errors because HTTP 400 /
        INVALID_ARGUMENT covers many unrelated causes.
        """
        try:
            return client.models.generate_content(
                model=self.model,
                contents=contents,
                config=genai.types.GenerateContentConfig(**gen_cfg),
            )
        except Exception as exc:
            # google-genai does not wrap transport errors — an httpx read/connect
            # timeout propagates raw. Fail fast so the retry helper does not loop.
            import httpx  # noqa: PLC0415
            if isinstance(exc, httpx.TimeoutException):
                raise ProviderTimeoutError(f"Gemini request timed out: {exc}", provider='gemini') from exc
            if self._classify_and_raise(exc, gen_cfg):
                fallback = {k: v for k, v in gen_cfg.items() if k != 'thinking_config'}
                return client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**fallback),
                )
            raise

    def send(self, dto: ProviderApiRequest) -> ProviderApiResponse:
        """Transform DTO → Gemini generate_content API → ProviderApiResponse."""
        genai = self._get_sdk()
        client = self._get_client(genai)
        start = time.time()

        contents = _gemini_convert_messages(dto.messages)
        gen_cfg = self._build_gen_config(genai, dto.system, dto.tools, dto.thinking_mode)

        response = self._generate_with_fallback(client, genai, contents, gen_cfg)
        latency_ms = int((time.time() - start) * 1000)
        text, tool_calls, finish_reason = self._parse_response(cast("_GenResponse", response))

        if not text and not tool_calls:
            logger.warning("[GeminiClient] Empty response, finish_reason=%s", finish_reason)
            raise ProviderResponseError(
                f"Empty Gemini response (finish_reason={finish_reason})",
                response_code=200, provider='gemini',
            )

        usage = getattr(response, 'usage_metadata', None)
        tokens_input = getattr(usage, 'prompt_token_count', None) if usage else None
        tokens_output = getattr(usage, 'candidates_token_count', None) if usage else None
        tokens_thinking = getattr(usage, 'thoughts_token_count', None) if usage else None

        logger.info(
            "[GeminiClient] model=%s tokens=%s+%s latency=%dms%s",
            self.model, tokens_input, tokens_output, latency_ms,
            f" tools={len(tool_calls)}" if tool_calls else "",
        )

        return ProviderApiResponse(
            text=text,
            model=self.model,
            provider='gemini',
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_thinking=tokens_thinking,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            response_code=200,
        )

    def get_context_limit(self) -> int:
        """Query Gemini API for model's input token limit, cached."""
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit
        try:
            genai = self._get_sdk()
            from services.llm_service import _resolve_api_key  # noqa: PLC0415
            client = genai.Client(api_key=_resolve_api_key(self._config))
            model_info = client.models.get(model=self.model)
            self._cached_context_limit: int = cast(int, model_info.input_token_limit)
            return self._cached_context_limit
        except Exception as exc:
            logger.debug("[GeminiClient] Failed to get context limit: %s", exc)
            self._cached_context_limit = 1_000_000
            return self._cached_context_limit

    def estimate_request_tokens(self, dto: ProviderApiRequest) -> int:
        """Estimate using build_request_body + heuristic estimate_tokens."""
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        contents = _gemini_convert_messages(dto.messages)
        config_dict: dict[str, object] = {'system_instruction': dto.system}
        if dto.tools:
            config_dict['tools'] = [
                {
                    'function_declarations': [
                        {
                            'name': t['name'],
                            'description': t.get('description', ''),
                            'parameters': t.get('input_schema'),
                        }
                        for t in dto.tools
                    ]
                }
            ]
        body = json.dumps({
            'model': self.model,
            'contents': contents,
            'config': config_dict,
        }, default=str)
        return estimate_tokens(body)
