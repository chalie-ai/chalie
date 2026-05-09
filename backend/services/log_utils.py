"""Log sanitization helpers — defeats CWE-117 log injection by stripping control chars from user-controlled values before they reach the logger."""
from __future__ import annotations

_CTRL_CHARS_TRANS = {c: None for c in range(32) if c not in (9,)}  # keep tab, drop newlines + other control chars
_CTRL_CHARS_TRANS[127] = None  # DEL


def safe(value) -> str:
    """Strip control characters (newlines, CR, etc.) from a value for safe inclusion in a log line.

    Returns a string. Non-string inputs are coerced via str(). Truncated to 512 chars to bound log spam.
    """
    if value is None:
        return ""
    s = str(value)
    s = s.translate(_CTRL_CHARS_TRANS)
    if len(s) > 512:
        s = s[:509] + "..."
    return s
