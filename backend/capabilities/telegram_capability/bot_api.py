"""Telegram Bot API calls — the one place a request to Telegram is made.

The bot token is part of every Bot API URL, so a ``requests`` error carries
it. No text leaves :func:`post` with the token in it.
"""

from __future__ import annotations

from typing import NamedTuple

import requests

from exceptions import TelegramApiError

_API_BASE = "https://api.telegram.org"
TIMEOUT = 15


class Outcome(NamedTuple):
    """One Bot API call: its ``result``, the HTTP status (0 when Telegram could
    not be reached) and, when it failed, why, as token-free text."""

    result: object
    status: int
    error: str


def post(
    token: str, method: str, payload: dict[str, str | int] | None = None, timeout: int = TIMEOUT,
) -> Outcome:
    """Call one Bot API *method*. A ``requests`` error is reduced to its type
    name, because its message holds the URL; Telegram's own description has
    the token cut out."""
    try:
        resp = requests.post(f"{_API_BASE}/bot{token}/{method}", json=payload or {}, timeout=timeout)
    except requests.RequestException as exc:
        return Outcome(None, 0, f"Could not reach Telegram ({type(exc).__name__}).")
    try:
        body = resp.json()
    except ValueError:
        return Outcome(
            None, resp.status_code,
            f"Telegram sent an unreadable reply to {method} (HTTP {resp.status_code}).",
        )
    if not isinstance(body, dict) or body.get("ok") is not True:
        description = body.get("description") if isinstance(body, dict) else None
        reason = str(description) if description else f"HTTP {resp.status_code}"
        return Outcome(
            None, resp.status_code, f"Telegram refused {method}: {reason.replace(token, '[token]')}.",
        )
    return Outcome(body.get("result"), resp.status_code, "")


def call(token: str, method: str, payload: dict[str, str | int] | None = None) -> object:
    """Call one Bot API *method* and return its ``result``. Raises
    :class:`TelegramApiError` when the call fails."""
    outcome = post(token, method, payload)
    if outcome.error:
        raise TelegramApiError(outcome.error)
    return outcome.result
