"""LLM service utilities — shared helpers used by the thin provider clients.

This module retains shared utilities (_app_user_agent,
_resolve_api_key, _strip_think_blocks, _is_thinking_rejection) after the
main client classes — and the message converters — were moved to
``services/llm_clients/*``.

The main client classes and their callers were migrated to the new
homes in ``services/llm_clients/*`` and ``services.provider_api``;
no backward-compat re-export shims remain.
"""

import logging
import re
from typing import cast

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>((?:(?!</think>).)*)(?:</think>\s*)?", re.DOTALL | re.IGNORECASE)

_APP_URL = "https://chalie.ai"
_APP_TITLE = "Chalie"


def _read_version() -> str:
    from services.file_mapper_service import FileMapperService
    try:
        return FileMapperService.get_version_path().read_text().strip()
    except OSError:
        return "0.0.0"


def _app_user_agent() -> str:
    return f"Chalie/{_read_version()}"


def _strip_think_blocks(text: str) -> tuple[str, str | None]:
    """Remove <think> chain-of-thought blocks emitted by reasoning models.

    An unclosed block is stripped to the end of the text: some providers omit
    the closing tag and glue the answer straight onto the reasoning with no
    delimiter, so nothing after the opener is mechanically separable. Callers
    must treat an empty result as "no response", never as an empty answer.

    Returns (cleaned_text, reasoning_trace).  *reasoning_trace* is the content
    of every <think>…</think> block found in *text*, joined in order, or None
    when no block carried content.  The cleaned text is identical to the
    previous behaviour.
    """
    if not text or "<think>" not in text.lower():
        return text, None
    traces = [t for t in (m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(text)) if t]
    stripped = _THINK_BLOCK_RE.sub("", text).strip()
    return stripped, "\n\n".join(traces) or None


def _is_thinking_rejection(exc: BaseException, create_kwargs: dict[str, object]) -> bool:
    """Return True when the provider refused a request that carried thinking params.

    Two facts decide it, and neither is the vendor's prose: the request we sent
    carried a thinking parameter — ``reasoning_effort``, or an ``extra_body``
    ``thinking`` key — and the provider answered **400**, a rejection of the
    request's shape rather than a server, auth or rate fault.

    Reading the message text is what this deliberately stopped doing. The old
    rule looked for 'reasoning_effort' or 'unsupported' in the error, and some
    servers write neither: "Unexpected reasoning effort high. Supported types
    are xhigh (default), medium, and low." spells the parameter with a space and
    says "Unexpected"/"Supported". A correct strip-and-retry ladder sat unreached
    behind that match, so every high-effort turn died on a refusal it could have
    recovered from.

    Callers reach this only inside a BadRequestError/APIError handler and only
    after the context-length branch has claimed its own case, so what is left is
    a request-shape fault. An unrelated one — a malformed tool schema, say —
    costs one retry without the thinking params and then raises as before. That
    is the price of not making recovery depend on wording nobody controls.
    """
    sent_thinking = 'reasoning_effort' in create_kwargs
    if not sent_thinking:
        extra_body = create_kwargs.get('extra_body')
        sent_thinking = isinstance(extra_body, dict) and 'thinking' in extra_body
    return sent_thinking and getattr(exc, 'status_code', 0) == 400


def _resolve_api_key(config: dict[str, object]) -> str:
    """Raises ValueError when the API key is not present in config."""
    api_key = config.get('api_key')
    if not api_key:
        raise ValueError(
            "API key not found in provider configuration. "
            "Store the API key in the database via the providers API "
            "(create: POST /api/providers/-1, update: POST /api/providers/<id>)"
        )
    return cast(str, api_key)
