"""Shared data-transformation utilities — chokepoints for patterns repeated across abilities and services."""
from __future__ import annotations

import json


def parse_json_column(raw: object, *, default: object = None) -> object:
    if default is None:
        default = {}
    if not raw:
        return default
    if not isinstance(raw, (str, bytes)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default
