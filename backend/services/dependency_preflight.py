"""Boot-time check that every dependency declared in ``pyproject.toml`` is installed.

Reads ``backend/pyproject.toml`` (resolved relative to this file, never
``cwd``) and verifies each entry in ``[project].dependencies`` is present on
the running interpreter's distribution index. Used by the startup guard to
refuse to run when a declared dependency is missing — prevents silent fallback
to degraded code paths (see the ``tiktoken`` regression that the boot
preflight was introduced to prevent).

Pure query: no logging, no printing, no side effects beyond the read of the
manifest file.
"""

from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path
from typing import Final

_BARE_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r'^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)'
)

_PYPROJECT_TOML: Final[Path] = (
    Path(__file__).resolve().parent.parent / 'pyproject.toml'
)


class DependencyPreflight:
    """Return the sorted list of declared-but-missing distribution names."""

    @classmethod
    def missing(cls) -> list[str]:
        """Return sorted bare distribution names declared but not installed."""
        with _PYPROJECT_TOML.open('rb') as _f:
            _data = tomllib.load(_f)

        _project = _data.get('project')
        if not isinstance(_project, dict):
            return []

        _raw = _project.get('dependencies', [])
        if not isinstance(_raw, list):
            return []

        _missing: list[str] = []
        for _spec in _raw:
            if not isinstance(_spec, str):
                continue
            _bare = DependencyPreflight._bare_name(_spec)
            if not _bare:
                continue
            try:
                importlib.metadata.distribution(_bare)
            except importlib.metadata.PackageNotFoundError:
                _missing.append(_bare)

        return sorted(_missing)

    @staticmethod
    def _bare_name(spec: str) -> str:
        """Strip version specifiers, extras, and env markers from a PEP 508 spec.

        Returns the empty string when the spec does not begin with a valid package
        name (should not happen for well-formed ``pyproject.toml``).
        """
        _match = _BARE_NAME_RE.match(spec.strip())
        return _match.group(1) if _match else ''
