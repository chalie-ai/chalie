"""Filename utilities — safe, canonical sanitization for uploaded files.

Wraps ``werkzeug.utils.secure_filename`` (battle-tested, ships with Flask) and
adds the one guarantee it omits: a 255-character cap that preserves the
extension. ``secure_filename`` may return an empty string for names that are
entirely unsafe (e.g. ``".."`` or non-ASCII-only names), so callers must supply
their own domain-specific fallback when the result is empty.

Consumers: api/upload.py and api/documents.py (chat attachment + document
uploads). Both previously hand-rolled near-identical sanitizers; this is the
single shared implementation.
"""

import os

from werkzeug.utils import secure_filename

_MAX_FILENAME_LEN = 255


def safe_filename(name: str) -> str:
    """Return a filesystem-safe filename, capped at 255 chars.

    Delegates to ``werkzeug.utils.secure_filename`` for the actual hardening
    (strips path separators, null bytes, control chars, leading dots; collapses
    spaces to underscores; reduces to ASCII), then enforces a 255-char limit
    while keeping the extension intact.

    Returns an empty string when the name reduces to nothing safe — the caller
    is responsible for substituting a fallback in that case.
    """
    name = secure_filename(name or '')
    if len(name) > _MAX_FILENAME_LEN:
        stem, ext = os.path.splitext(name)
        # Drop a pathologically long "extension" so the slice can't go negative
        # and re-overflow the cap (e.g. a 256-char ".xxxx…" tail).
        if len(ext) >= _MAX_FILENAME_LEN:
            stem, ext = name, ''
        name = stem[:_MAX_FILENAME_LEN - len(ext)] + ext
    return name
