"""
LLM service utilities — shared helpers used by the thin provider clients.

This module retains shared utilities (estimate_tokens, _call_with_retry,
_app_user_agent, _resolve_api_key, _strip_think_blocks, _parse_retry_after,
_is_thinking_rejection) after the main client classes — and the message
converters — were moved to services/llm_clients/*.

DELETED in the provider-client refactor:
  - AnthropicService, OpenAIService, GeminiService → llm_clients/anthropic.py etc.
  - FallbackLLMService — dead code (fallback_provider never set in DB/schema).
  - LoggingLLMService — absorbed by Providers._log_after_call chokepoint.
  - create_llm_service / _build_service — replaced by llm_clients/factory.build_client.
  - _log_llm_call — merged into Providers._log_after_call (resolution #4).
  - LLMResponse — replaced by ProviderApiResponse in services.provider_api.
  - PayloadTooLargeError — replaced by ResponseOverLimitError in services.provider_api.
  - NonRetryableError — replaced by ProviderResponseError hierarchy.
  - RateLimitError — moved to services.provider_api.

No backward-compat re-export shims remain: every caller (production and test)
imports these symbols from their new homes (services.provider_api,
services.llm_clients.*) directly.
"""

import re
import time
import logging
from typing import Optional

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


def _call_with_retry(fn, max_retries=2, backoff=1.0):
    """Retry fn() up to max_retries times with exponential backoff.

    Rate limits (RateLimitError) are retried once with a short wait if the
    provider includes a Retry-After header (capped at 10s). If no header is
    present, the error propagates immediately.
    """
    from services.provider_api import RateLimitError  # noqa: PLC0415
    attempt = 0
    while attempt <= max_retries:
        try:
            return fn()
        except RateLimitError as e:
            if e.retry_after and e.retry_after <= 10.0:
                logger.warning(
                    "Rate limited by %s (Retry-After %.0fs). Retrying once...",
                    e.provider or 'provider', e.retry_after,
                )
                time.sleep(e.retry_after)
                try:
                    return fn()
                except RateLimitError:
                    raise
            raise
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning("LLM call failed (attempt %d): %s. Retrying in %ss...", attempt + 1, e, wait)
            time.sleep(wait)
            attempt += 1


def _parse_retry_after(exc) -> Optional[float]:
    """Extract the Retry-After header value from an HTTP exception, or return None."""
    if hasattr(exc, 'response') and exc.response is not None:
        ra = exc.response.headers.get('retry-after')
        if ra:
            try:
                return float(ra)
            except (ValueError, TypeError) as parse_err:
                logger.debug("[LLM] Could not parse Retry-After header value %r: %s", ra, parse_err)
    return None


def _is_thinking_rejection(exc, create_kwargs: dict) -> bool:
    """Return True when the provider rejected a reasoning_effort parameter."""
    if 'reasoning_effort' not in create_kwargs:
        return False
    err = str(exc).lower()
    return 'reasoning_effort' in err or 'unsupported' in err


def _resolve_api_key(config: dict) -> str:
    """Resolve API key from provider config (from database).

    Raises:
        ValueError: if api_key is not found in config.
    """
    api_key = config.get('api_key')
    if not api_key:
        raise ValueError(
            "API key not found in provider configuration. "
            "Store the API key in the database via POST /providers or update via PUT /providers/<id>"
        )
    return api_key
