"""LLM service utilities — shared helpers used by the thin provider clients.

This module retains shared utilities (estimate_tokens, _app_user_agent,
_resolve_api_key, _strip_think_blocks, _is_thinking_rejection) after the
main client classes — and the message converters — were moved to
``services/llm_clients/*``.

The main client classes and their callers were migrated to the new
homes in ``services/llm_clients/*`` and ``services.provider_api``;
no backward-compat re-export shims remain.
"""

import re
import logging
from typing import cast

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

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
    """Remove <think>...</think> chain-of-thought blocks emitted by reasoning models."""
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
    """Return True when the provider rejected a reasoning_effort parameter."""
    if 'reasoning_effort' not in create_kwargs:
        return False
    err = str(exc).lower()
    return 'reasoning_effort' in err or 'unsupported' in err


def _resolve_api_key(config: dict[str, object]) -> str:
    """Raises ValueError when the API key is not present in config."""
    api_key = config.get('api_key')
    if not api_key:
        raise ValueError(
            "API key not found in provider configuration. "
            "Store the API key in the database via POST /providers or update via PUT /providers/<id>"
        )
    return cast(str, api_key)
