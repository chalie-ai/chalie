"""LLM service utilities — shared helpers used by the thin provider clients.

This module retains shared utilities (estimate_tokens, _app_user_agent,
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

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>\s*|\Z)", re.DOTALL | re.IGNORECASE)

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


def _strip_think_blocks(text: str) -> str:
    """Remove <think> chain-of-thought blocks emitted by reasoning models.

    An unclosed block is stripped to the end of the text: some providers omit
    the closing tag and glue the answer straight onto the reasoning with no
    delimiter, so nothing after the opener is mechanically separable. Callers
    must treat an empty result as "no response", never as an empty answer.
    """
    if not text or "<think>" not in text.lower():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def estimate_tokens(text: str) -> int:
    """Fast token estimate (~1.3 tokens per whitespace-delimited word).

    Used as a fallback when provider-specific counting is unavailable,
    and for quick budget checks where exact counts aren't critical.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def _is_thinking_rejection(exc: BaseException, create_kwargs: dict[str, object]) -> bool:
    """Return True when the provider rejected a thinking-related parameter.

    Two rejection shapes are recognized:

    1. reasoning_effort rejection (OpenAI native):
       create_kwargs carries 'reasoning_effort' and the error text mentions
       'reasoning_effort' or 'unsupported'.

    2. extra_body thinking rejection (OpenAI-compatible / vLLM style):
       create_kwargs carries an 'extra_body' dict containing a 'thinking' key
       (the vendor-extension disable param) and the error text mentions
       'thinking', 'extra_forbidden', 'extra inputs', or 'unsupported'.
       Strict servers reject unknown body fields with 400 'Extra inputs are
       not permitted'; the field must be dropped and retried.
    """
    if 'reasoning_effort' in create_kwargs:
        err = str(exc).lower()
        if 'reasoning_effort' in err or 'unsupported' in err:
            return True
    extra_body = create_kwargs.get('extra_body')
    if isinstance(extra_body, dict) and 'thinking' in extra_body:
        err = str(exc).lower()
        if any(term in err for term in ('thinking', 'extra_forbidden', 'extra inputs', 'unsupported')):
            return True
    return False


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
