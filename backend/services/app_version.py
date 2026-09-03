"""The one place that reads and compares Chalie's application version.

Every version read in the backend goes through :func:`get_version` — the
health endpoint (via ``consumer.APP_VERSION``), the provider user agent,
and the ``chalie_docs`` ability. No module may open the VERSION file on
its own. The file lives at the repo root (see
:meth:`services.file_mapper_service.FileMapperService.get_version_path`),
and a missing or empty file raises an exception naming the path: a silent
"0.0.0" / "unknown" fallback would let a broken install masquerade as a
release.

Version comparison goes through :func:`version_sort_key`: versions order
numerically per dotted part, and a pre-release suffix sorts before its
final release — ``1.2.0 < 1.10.0``, ``1.3.0-beta < 1.3.0 < 1.3.1``.
Hand-rolled on purpose: no ``packaging`` dependency.
"""

from __future__ import annotations

import re

from services.file_mapper_service import FileMapperService

#: Accepted version shape: dotted numeric parts plus an optional pre-release
#: suffix — ``1.3.0``, ``1.3.0-beta``, ``1.10.0-rc.2``.
_VERSION_RE = re.compile(r"^(?P<core>\d+(?:\.\d+)*)(?:-(?P<pre>[0-9A-Za-z.]+))?$")

#: Splits a pre-release suffix into comparable tokens (``beta.2`` →
#: ``beta``, ``.``, ``2``).
_PRE_TOKEN_RE = re.compile(r"\d+|[A-Za-z]+|\.")

#: Shape of a :func:`version_sort_key` result: the numeric core parts,
#: ``0`` when the version carries a pre-release suffix (else ``1``), and
#: the pre-release tokens (empty for a final release).
VersionKey = tuple[tuple[int, ...], int, tuple[tuple[int, str | int], ...]]


def get_version() -> str:
    """Return the running build's version, read from the repo-root VERSION file.

    The single read point for the version: ``consumer.APP_VERSION``, the LLM
    user agent, and the ``chalie_docs`` ability all resolve through here.
    There is deliberately no fallback value — a missing or empty file raises
    an exception naming the path, so a broken install cannot masquerade as a
    release.

    Raises:
        FileNotFoundError: The VERSION file is missing (the message names the path).
        ValueError: The VERSION file is empty or whitespace-only (the message names the path).
    """
    path = FileMapperService.get_version_path()
    try:
        raw = path.read_text().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"VERSION file is missing: {path}") from exc
    if not raw:
        raise ValueError(f"VERSION file is empty: {path}")
    return raw


def _pre_token(token: str) -> tuple[int, str | int]:
    """Map one pre-release token to ``(kind, value)``.

    Kind ``0`` = numeric (value is the int, so ``rc2 < rc10``); kind ``1`` =
    alphabetic (value is the string). The kind tag keeps a numeric token
    from ever being compared directly against a string.
    """
    return (0, int(token)) if token.isdigit() else (1, token)


def version_sort_key(version: str) -> VersionKey:
    """Return a sort key that orders versions numerically per dotted part.

    A pre-release suffix sorts before its final release: ``1.2.0 < 1.10.0``
    and ``1.3.0-beta < 1.3.0 < 1.3.1``. The key is:

    1. the numeric core parts,
    2. ``0`` when the version carries a pre-release suffix, else ``1``,
    3. the pre-release tokens (see :func:`_pre_token`), empty for a final.

    Raises:
        ValueError: *version* does not match the shape accepted by
            :data:`_VERSION_RE`.
    """
    match = _VERSION_RE.fullmatch(version.strip())
    if match is None:
        raise ValueError(f"unrecognised version format: {version!r}")
    core = tuple(int(part) for part in match.group("core").split("."))
    pre = match.group("pre")
    if pre is None:
        return (core, 1, ())
    return (core, 0, tuple(_pre_token(token) for token in _PRE_TOKEN_RE.findall(pre)))
