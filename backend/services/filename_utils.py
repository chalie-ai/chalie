"""Filename utilities — safe, canonical sanitization for uploaded files.

Wraps ``werkzeug.utils.secure_filename`` (battle-tested, ships with Flask) and
adds the one guarantee it omits: a 255-character cap that preserves the
extension. ``secure_filename`` may return an empty string for names that are
entirely unsafe (e.g. ``".."`` or non-ASCII-only names), so callers must supply
their own domain-specific fallback when the result is empty.

Consumers: api/endpoints/threads.py (chat attachment uploads) and
services/file_parser_service.py (file ingest). Both call sites share this
single implementation instead of hand-rolling near-identical sanitizers.
"""

import os

from werkzeug.utils import secure_filename

_MAX_FILENAME_LEN = 255


def safe_filename(name: str) -> str:
    """Sanitize filename to be filesystem-safe, capped at 255 chars while preserving the extension."""
    name = secure_filename(name or '')
    if len(name) > _MAX_FILENAME_LEN:
        stem, ext = os.path.splitext(name)
        # Drop a pathologically long "extension" so the slice can't go negative
        # and re-overflow the cap (e.g. a 256-char ".xxxx…" tail).
        if len(ext) >= _MAX_FILENAME_LEN:
            stem, ext = name, ''
        name = stem[:_MAX_FILENAME_LEN - len(ext)] + ext
    return name
