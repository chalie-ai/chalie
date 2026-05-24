"""Shared data-transformation utilities — chokepoints for patterns repeated across abilities and services."""
from __future__ import annotations

import json
from typing import Any


def parse_json_column(raw: Any, *, default: Any = None) -> Any:
    """Safely parse a JSON string from a database column.

    Returns *default* (``{}`` when omitted) if *raw* is falsy or
    unparseable.  Non-string values are returned as-is.
    """
    if default is None:
        default = {}
    if not raw:
        return default
    if not isinstance(raw, (str, bytes)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default
