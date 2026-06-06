# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared location + guard for Chalie's transient upload/attachment files.

Single source of truth so the sites that touch these files can never drift
apart:

  * write  — ``api/chat.py`` (chat attachments) and ``api/documents.py`` (the
             Documents library) save the raw uploaded file here
  * ingest — ``abilities/document.py`` copies it into the documents store by
             PATH (``document.upload`` never carries bytes — TKT-844)
  * sweep  — ``workers/tmp_cleanup_worker.py`` deletes stale ones

Files live under the OS temp directory (``tempfile.gettempdir()``) rather than a
hardcoded ``/tmp``.  This lands them in the platform-appropriate, per-user temp
location (e.g. ``$TMPDIR`` on macOS) instead of a world-writable path, which is
what SonarCloud's "publicly writable directory" rule (python:S5443) asks for.
On Linux servers ``gettempdir()`` resolves to ``/tmp``, so server behaviour is
unchanged.
"""

import os
import tempfile

# Canonical temp directory (symlinks resolved) holding Chalie's transient files.
# The OS temp dir does not change at runtime, so resolve it once at import.
TMP_DIR = os.path.realpath(tempfile.gettempdir())

# Filename prefix marking a path as Chalie-owned (and sweepable by the worker).
TMP_PREFIX = "chalie_"

# Absolute path prefix that new transient files are created under.
TMP_PATH_PREFIX = os.path.join(TMP_DIR, TMP_PREFIX)


def new_tmp_path(suffix: str = "") -> str:
    """Return an absolute path under the Chalie temp prefix for a new file.

    ``suffix`` is appended verbatim (typically ``f"{token}{ext}"``).
    """
    return f"{TMP_PATH_PREFIX}{suffix}"


def is_chalie_tmp_file(path: str) -> bool:
    """True if *path* resolves to an existing file under the Chalie temp prefix.

    Resolves symlinks first (``realpath``) so a symlinked path cannot escape the
    prefix guard.
    """
    real = os.path.realpath(path)
    return real.startswith(TMP_PATH_PREFIX) and os.path.isfile(real)
