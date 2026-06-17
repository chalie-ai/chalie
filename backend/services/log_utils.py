"""Log sanitization helpers — defeats CWE-117 log injection by stripping control chars from user-controlled values before they reach the logger."""
from __future__ import annotations

_CTRL_CHARS_TRANS = {c: None for c in range(32) if c not in (9,)}  # keep tab, drop newlines + other control chars
_CTRL_CHARS_TRANS[127] = None  # DEL


def safe(value) -> str:
    """Non-string inputs are coerced via str()."""
    if value is None:
        return ""
    s = str(value)
    s = s.translate(_CTRL_CHARS_TRANS)
    return s
