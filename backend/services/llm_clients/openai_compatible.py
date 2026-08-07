"""
Base client for every provider that speaks the OpenAI wire protocol.

This is the shared implementation, not a provider in its own right — though it
is also usable directly as platform ``openai_compatible``, the escape hatch for
a host with no dedicated subclass.

Subclasses live one per file and customise through class attributes and the
``_probe_context_window`` hook. Nothing here branches on which provider is
calling: a new vendor is added by writing its own module, never by editing this
one. Where a vendor's behaviour genuinely differs — a different window endpoint,
a payload quirk — it overrides the hook rather than growing a condition here.

Native size-rejection signal:
  HTTP 400 with error.code == 'context_length_exceeded' → ContextLimit.
  Confirmed from existing code: openai_mod.BadRequestError is caught in
  _call_completions; the 'context_length_exceeded' code is the canonical OpenAI
  signal (https://platform.openai.com/docs/guides/error-codes).
  That code is OpenAI's own convention, and the openai_compatible hosts sharing
  this client are under no obligation to use it, so the message prose is matched
  as well (``is_token_limit_message``) — otherwise a size rejection from a
  self-hosted server would surface as a generic 400 and never compact.

Depends on: services.provider_api (contract), services.llm_service
(LlmService.app_user_agent, LlmService.resolve_api_key, LlmService.strip_think_blocks — utilities).
Consumed by: services.llm_clients.registry (platform dispatch), and by every
OpenAI-protocol subclass in this package.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Sequence, TypeAlias, cast

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
from services.llm_clients.context_window import DEFAULT_WINDOW
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

#: Stand-in credential for a server that asks for none. The OpenAI SDK refuses
#: to construct without an api_key, so a keyless self-hosted host needs some
#: string here — this is the one vLLM's and llama.cpp's own OpenAI-client
#: examples use, so it is what those servers expect to be ignoring.
_UNAUTHENTICATED_PLACEHOLDER_KEY = "EMPTY"

_COMPLETION_USAGE: TypeAlias = "_openai_mod.types.CompletionUsage"


class OpenAICompatibleClient(ProviderClient):
    """Shared implementation for OpenAI-protocol providers.

    Subclasses declare what their vendor does through the class attributes
    below and, where a vendor publishes its window somewhere other than a
    ``/models`` entry, by overriding ``_probe_context_window``. Adding a
    provider must never require an edit to this class.
    """

    CONTENT_FIELD_LABEL: ClassVar[str] = "choices[].message.content"

    #: Value stored in ``providers.platform`` and the key this class is
    #: registered under. Every subclass sets its own.
    PLATFORM: ClassVar[str] = 'openai_compatible'

    #: Catalog display name, and the base URL pre-filled in the setup wizard.
    #: An empty URL means the user must supply the host themselves.
    LABEL: ClassVar[str] = 'OpenAI-compatible (custom host)'
    DEFAULT_BASE_URL: ClassVar[str] = ''

    #: Whether creating this provider requires a key / an explicit host.
    REQUIRES_KEY: ClassVar[bool] = True
    REQUIRES_HOST: ClassVar[bool] = True

    #: Whether the reasoning toggle for ThinkingLevel.NONE travels as a body
    #: field. Vendor extension: official OpenAI rejects unknown top-level keys.
    SENDS_THINKING_EXTRA_BODY: ClassVar[bool] = True

    #: Where this vendor publishes the context window on a ``/models`` entry,
    #: as dotted paths tried in order — so a subclass that has both a precise
    #: and an approximate spelling states which one wins simply by listing it
    #: first. The base carries the union of the documented spellings because it
    #: also serves unknown hosts; a subclass narrows this to what its vendor
    #: actually documents, so a coincidental key on another vendor's payload
    #: can never be read as a window.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = (
        'max_model_len',                # vLLM / NVIDIA NIM ModelCard
        'context_length',               # xAI, Moonshot, OpenRouter
        'max_context_length',           # Mistral
        'context_window',               # Groq
        'context_size',                 # Novita
        'max_input_tokens',             # Anthropic-compatible shims
        'top_provider.context_length',  # OpenRouter
        'meta.n_ctx',                   # llama.cpp
    )

    @staticmethod
    def _as_dict(entry: object) -> dict[str, object]:
        """A ``/models`` listing entry as a plain dict, whatever the SDK returned.

        The OpenAI SDK types only the fields OpenAI itself serves, so every vendor
        size field arrives as an untyped extra. Flattening once here keeps every
        lookup below an ordinary dict lookup instead of a getattr/model_extra dance
        repeated per key.
        """
        if isinstance(entry, dict):
            return dict(entry)
        data: dict[str, object] = {}
        dump = getattr(entry, 'model_dump', None)
        if callable(dump):
            try:
                dumped = dump()
            except Exception:  # noqa: BLE001 - a vendor extra must never break a probe
                dumped = None
            if isinstance(dumped, dict):
                data.update(dumped)
        extra = getattr(entry, 'model_extra', None)
        if isinstance(extra, dict):
            data.update(extra)
        return data

    @staticmethod
    def _positive_int(value: object) -> int | None:
        """*value* if it is a usable token count, else None (bools are not ints here)."""
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value > 0 else None

    @staticmethod
    def _resolve_path(data: dict[str, object], path: str) -> object | None:
        """The value at a dotted *path* in *data*, or None if any hop is missing.

        Keeps ``WINDOW_FIELDS`` a single ordered list: a vendor that publishes both
        a nested and a top-level figure declares which it trusts by ordering, with
        no second list and no precedence rule to remember.
        """
        current: object = data
        for key in path.split('.'):
            if not isinstance(current, dict):
                return None
            current = cast(dict[str, object], current).get(key)
        return current

    @staticmethod
    def _select_models_entry(entries: "Sequence[object]", target: str) -> object | None:
        """The listing entry describing *target*, or None when it is ambiguous.

        Matches on ``id``, then on ``root`` — a server started from a local file
        reports the path in one and the alias in the other, and which way round
        varies by server. A single-entry listing matches unconditionally: one served
        model is the norm for a self-hosted host, and there is nothing to confuse it
        with.
        """
        slug = (target or '').strip().lower()
        for entry in entries:
            data = OpenAICompatibleClient._as_dict(entry)
            for field in ('id', 'root'):
                value = data.get(field)
                if isinstance(value, str) and value.strip().lower() == slug:
                    return entry
        return entries[0] if len(entries) == 1 else None

    @staticmethod
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

    @classmethod
    def fetch_models(
        cls, host: str, api_key: str,
    ) -> tuple[list[dict[str, str | None]] | None, str | None]:
        """Every OpenAI-protocol host answers ``GET {base_url}/models``."""
        from services.provider_probe import fetch_openai_compatible_models  # noqa: PLC0415
        return fetch_openai_compatible_models(host or cls.DEFAULT_BASE_URL, api_key)

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config
        self.model: str = cast(str, config.get('model', ''))
        self._format: str = cast(str, config.get('format', 'text'))
        self.platform: str = cast(str, config.get('platform', self.PLATFORM))
    def _api_key(self) -> str:
        """The credential to present, honouring this platform's own key rule.

        A platform that declares ``REQUIRES_KEY = False`` is a server that
        serves openly unless its operator opted in to a token — vLLM and
        llama.cpp both start without one. The OpenAI SDK still insists on a
        non-empty string, so an unauthenticated host gets the placeholder both
        vendors' own quickstarts pass; an open server ignores it, and a server
        that does want a token answers 401 on its own terms.

        A configured key always wins, whatever the platform's default: opting
        in to authentication is the operator's call, not ours.
        """
        from services.llm_service import LlmService  # noqa: PLC0415

        if not self.REQUIRES_KEY and not self._config.get('api_key'):
            return _UNAUTHENTICATED_PLACEHOLDER_KEY
        return LlmService.resolve_api_key(self._config)

    def _get_client(self) -> "_openai_mod.OpenAI":
        from openai import OpenAI  # noqa: PLC0415
        from services.llm_service import LlmService  # noqa: PLC0415
        from services.provider_api import PROVIDER_CALL_TIMEOUT_S  # noqa: PLC0415
        kwargs: dict[str, object] = {
            'api_key': self._api_key(),
            'timeout': PROVIDER_CALL_TIMEOUT_S,
            'default_headers': {
                "HTTP-Referer": _APP_URL,
                "X-Title": _APP_TITLE,
                "User-Agent": LlmService.app_user_agent(),
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
        from services.llm_service import LlmService  # noqa: PLC0415
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
            if LlmService.is_thinking_rejection(exc, create_kwargs):
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
                        if LlmService.is_thinking_rejection(retry_exc, fallback1):
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
        from services.llm_service import LlmService  # noqa: PLC0415

        client = self._get_client()
        start = time.time()

        api_messages = OpenAICompatibleClient._openai_convert_messages(dto.messages)
        create_kwargs: dict[str, object] = {
            'model': self.model,
            'messages': [{"role": "system", "content": dto.system}] + api_messages,
        }
        if dto.tools:
            create_kwargs['tools'] = self._build_openai_tools(dto.tools)

        effort = self._thinking_native(dto.thinking_mode)
        if effort:
            create_kwargs['reasoning_effort'] = effort
            # Vendor-extension disable param for endpoints whose reasoning
            # toggle is a body field (vLLM/Z.ai style). Sent only for NONE —
            # other levels use reasoning_effort above.
            if self.SENDS_THINKING_EXTRA_BODY and dto.thinking_mode is ThinkingLevel.NONE:
                create_kwargs['extra_body'] = OPENAI_COMPATIBLE_NONE_BODY

        response = self._invoke_create(client, create_kwargs)
        latency_ms = int((time.time() - start) * 1000)

        msg = response.choices[0].message
        text, think_trace = LlmService.strip_think_blocks(msg.content or "")
        # Priority: explicit reasoning field on the message, else the extracted
        # <think> content.  Leave thinking_block None when neither source is
        # present.
        thinking_block: str | None = None
        for _field in ('reasoning', 'reasoning_content'):
            _val = getattr(msg, _field, None)
            if _val and isinstance(_val, str) and _val.strip():
                thinking_block = _val.strip()
                break
        if thinking_block is None and think_trace:
            thinking_block = think_trace
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
            "[%s] model=%s tokens=%d+%d latency=%dms finish=%s%s",
            type(self).__name__,
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
            thinking_block=thinking_block,
            response_code=200,
        )

    def get_context_limit(self) -> int | None:
        """The model's context window, measured once per instance.

        Caching and the never-raise contract live here; *how* a vendor is asked
        lives in ``_probe_context_window``, which subclasses override. Returns
        None only when the provider could not be reached at all, so the column
        stays unset and the next send re-probes rather than pinning a
        fabricated figure onto the row.
        """
        if hasattr(self, '_cached_context_limit'):
            return self._cached_context_limit

        self._cached_context_limit: int | None = self._probe_context_window()
        return self._cached_context_limit

    def _probe_context_window(self) -> int | None:
        """Ask this vendor what its window is. The subclass override point.

        The default suits any server that publishes the figure on ``/models``:
        read it, and fall back to a ping — which proves the host answers and so
        earns the default window — when the listing names no size.

        Override this, do not extend it with a condition, when a vendor
        publishes elsewhere (a different path, a different response shape) or
        publishes nothing at all.
        """
        window = self._window_from_models_endpoint()
        return window if window is not None else self._probe_via_ping()

    def _window_from_models_entry(self, entry: object) -> int | None:
        """The context window a ``/models`` entry advertises, or None.

        Walks this class's ``WINDOW_FIELDS`` in declaration order and returns
        the first path that resolves to a positive integer.

        ``meta.n_ctx_train`` is deliberately absent from every field list, and
        must not be added. That field is the underlying model's *training*
        context, which routinely exceeds what the server was actually started
        with; trusting it over-promises the window and gets payloads rejected
        mid-turn — the exact failure this probe exists to prevent.
        """
        data = OpenAICompatibleClient._as_dict(entry)
        for path in self.WINDOW_FIELDS:
            window = OpenAICompatibleClient._positive_int(OpenAICompatibleClient._resolve_path(data, path))
            if window is not None:
                return window
        return None

    def _window_from_models_endpoint(self) -> int | None:
        """The window this host publishes for ``self.model`` on ``/models``.

        The authoritative answer wherever it exists: a self-hosted or
        third-party server that states its own serving limit is not guessing.
        See ``_select_models_entry`` for how the entry is matched and
        ``_window_from_models_entry`` for which fields are read.

        Returns None — never raises — when the host is down, serves no matching
        entry, or names no size, leaving the caller to fall through to a ping.
        """
        try:
            # Materialised eagerly: the SDK returns a lazily-iterated page, and
            # a network failure part-way through must be caught here, not later.
            entries = list(self._get_client().models.list())
        except Exception as exc:
            logger.warning(
                "[%s] /models unavailable for window detection on model=%s: %s",
                type(self).__name__, self.model, exc,
            )
            return None

        entry = OpenAICompatibleClient._select_models_entry(entries, self.model)
        return self._window_from_models_entry(entry) if entry is not None else None

    def _probe_via_ping(self) -> int | None:
        """Confirm a hosted provider answers and derive its context window.

        Sends a 5-token one-shot completion. A reply with a readable
        ``usage.prompt_tokens`` proves the host and model are live; the window
        is then taken from a size field on the response if one is present, else
        a family-based default. Returns None (never raises) when the host cannot
        be reached, so the row is left unset for the next send to re-probe.

        A refusal is not a silence. Any HTTP status — a 400 rejecting a
        parameter, a 401, a 404, a 429 — came from a host that is up, so it
        sizes like any other live provider and lets the real fault surface at
        send time, where the message actually names it.
        """
        import openai as openai_mod  # noqa: PLC0415

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
        except openai_mod.APIStatusError as exc:
            # The host answered, and an answer is the liveness this probe exists
            # to establish. Reasoning models reject max_tokens outright (they
            # require max_completion_tokens), so a plain 400 here is routine —
            # returning None for it would make pin_context_window raise and kill
            # every turn for a provider whose only real problem is elsewhere.
            window = DEFAULT_WINDOW
            logger.warning(
                "[%s] Context-window ping for model=%s was refused with "
                "HTTP %s (%s) — the host is up, so sizing it at %d",
                type(self).__name__, self.model, exc.status_code, exc, window,
            )
            return window
        except Exception as exc:
            logger.warning(
                "[%s] Context-window ping failed for model=%s: %s",
                type(self).__name__, self.model, exc,
            )
            return None

        usage = getattr(resp, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', None) if usage else None
        if not isinstance(prompt_tokens, int):
            logger.warning(
                "[%s] Context-window ping for model=%s returned no usage — "
                "treating the host as unusable, leaving the window unset",
                type(self).__name__, self.model,
            )
            return None

        reported = self._window_from_response(resp)
        window = reported if reported is not None else DEFAULT_WINDOW
        source = 'response' if reported is not None else 'DEFAULT_WINDOW'
        logger.info(
            "[%s] Context-window ping ok for model=%s (prompt_tokens=%d) — window=%d (%s)",
            type(self).__name__, self.model, prompt_tokens, window, source,
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


# Backward-compat aliases for callers that import the module-level helpers directly.
_as_dict = OpenAICompatibleClient._as_dict
_positive_int = OpenAICompatibleClient._positive_int
_resolve_path = OpenAICompatibleClient._resolve_path
_select_models_entry = OpenAICompatibleClient._select_models_entry
_openai_convert_messages = OpenAICompatibleClient._openai_convert_messages
