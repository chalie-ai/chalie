"""FilePermissionsAbility — chmod a single file or directory.

Safe alternative to invoking ``bash`` for ``chmod``. The success body carries the
post-chmod mode (``mode_octal`` + ``mode_symbolic``) and ``meta previous`` carries
the pre-chmod octal, so the audit trail (and the user) can see exactly what
changed.

The ``permissions`` value accepts three grammars so the model never has to do
octal arithmetic to follow the intent-phrased examples:

* Octal — 3 or 4 octal digits, leading 0 optional (``"755"``, ``"0644"``).
* chmod-style symbolic clauses, comma-separated — ``[ugoa]*[+-=][rwx]+``
  (``"+x"``, ``"u+rwx"``, ``"go-w"``, ``"a=r"``, ``"u+x,go-rwx"``); no target
  letters means ``a`` (all). Clauses resolve against the file's CURRENT mode.
* Intent keywords (case-insensitive) — ``"executable"`` → ``+x``,
  ``"readonly"``/``"read-only"`` → ``a=r``, ``"owner-only"``/``"private"`` →
  ``go-rwx``.
"""

import logging
import os
import re
from pathlib import Path
from typing import ClassVar, cast

from abilities._ability import Ability
from abilities._params import Keys
from abilities._result import ToolResult

logger = logging.getLogger(__name__)

_MAX_OCTAL = 0o7777

_PERM_HINT = (
    "pass octal ('755', '0644') or symbolic ('+x', 'u+rwx,go-w', 'readonly', "
    "'owner-only')."
)

# Intent keywords map to their symbolic equivalents (resolved against the file's
# current mode, exactly like a typed symbolic clause).
_INTENT: dict[str, str] = {
    "executable": "+x",
    "readonly": "a=r",
    "read-only": "a=r",
    "owner-only": "go-rwx",
    "private": "go-rwx",
}

# One chmod clause: optional [ugoa] targets, an op, and 1+ of [rwx].
_CLAUSE_RE = re.compile(r"^([ugoa]*)([+\-=])([rwx]+)$")

# Per-target permission bit masks (the rwx triplet for each who-class).
_WHO_BITS: dict[str, int] = {
    "u": 0o700,
    "g": 0o070,
    "o": 0o007,
    "a": 0o777,
}
_PERM_BITS: dict[str, int] = {"r": 0o4, "w": 0o2, "x": 0o1}


class FilePermissionsAbility(Ability):
    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank ``path`` or ``permissions`` with
    #: ``code=missing-params`` BEFORE run(). Blank strings are invalid here, so the
    #: truthiness-based pre-gate is exactly right; run() guards the whitespace-only
    #: residue the pre-gate lets through (``"  "`` is truthy).
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.path, Keys.permissions)}

    def get_name(self) -> str:
        return "file_permissions"

    def get_summary(self) -> str:
        return "Change Unix-style filesystem permissions (chmod) on a file or directory."

    def get_examples(self) -> list[str]:
        return [
            "make this script executable",
            "chmod 755 /home/user/scripts/deploy.sh",
            "make this file read-only",
            "give this file 644 permissions",
            "change permissions on a file",
            "make this binary executable so I can run it",
            "lock down this config file to owner-only",
        ]

    def get_search_tooltip(self) -> str:
        return "Change file permissions (chmod)"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.path: {
                "type": "string",
                "description": "Absolute path to the file or directory whose permissions you want to change.",
            },
            Keys.permissions: {
                "type": "string",
                "description": (
                    "How to set the permissions. Accepts three forms:"
                    " (1) Octal — 3 or 4 octal digits, leading 0 optional:"
                    " '755' (rwxr-xr-x), '0644' (rw-r--r--), '600' (rw-------)."
                    " (2) chmod-style symbolic clauses, comma-separated,"
                    " each '[ugoa]*[+-=][rwx]+', resolved against the file's"
                    " current mode: '+x', 'u+rwx', 'go-w', 'a=r', 'u+x,go-rwx'"
                    " (no target letters means all)."
                    " (3) Intent keywords (case-insensitive): 'executable' (+x),"
                    " 'readonly'/'read-only' (a=r), 'owner-only'/'private' (go-rwx)."
                ),
            },
        },
        "required": [Keys.path, Keys.permissions],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        path_str = cast(str, params.get(Keys.path) or "").strip()
        perm_str = cast(str, params.get(Keys.permissions) or "").strip()

        # The ACTION_REQUIRED pre-gate rejects a missing/blank path or
        # permissions; this guards the whitespace-only residue it lets through
        # (a present-but-blank-after-strip value is truthy to the pre-gate).
        if not path_str or not perm_str:
            return ToolResult.err(
                "path and permissions are both required.",
                code="missing-params",
                valid=("path", "permissions"),
            )

        target = Path(os.path.expanduser(path_str)).resolve()
        target_str = str(target)

        if not target.exists():
            return ToolResult.err(
                f"No such file or directory: {target_str}",
                code="path-not-found",
                hint="pass an absolute path to an existing file or directory.",
            )

        try:
            before_mode = target.stat().st_mode
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied reading {target_str}: {exc}",
                code="permission-denied",
                hint=f"you do not own {target_str} or cannot stat it.",
            )

        mode = _resolve_mode(perm_str, before_mode & 0o7777)
        if mode is None:
            return ToolResult.err(
                f"Cannot parse permissions: {perm_str!r}",
                code="invalid-mode",
                hint=_PERM_HINT,
            )

        try:
            os.chmod(target, mode)
            after_mode = target.stat().st_mode
        except PermissionError as exc:
            return ToolResult.err(
                f"Permission denied changing {target_str}: {exc}",
                code="permission-denied",
                hint=f"you must own {target_str} to change its permissions.",
            )
        except OSError as exc:
            logger.exception("[FILE_PERMISSIONS] chmod failed for path=%r: %s", target_str, exc)
            return ToolResult.err(
                f"Could not change permissions on {target_str}: {exc}",
                code="chmod-failed",
                hint="check the path exists and is a regular file or directory.",
            )

        return ToolResult.ok(
            {
                "path": target_str,
                "mode_octal": _format_octal(after_mode),
                "mode_symbolic": _format_symbolic(after_mode),
            },
            previous=_format_octal(before_mode),
        )


def _resolve_mode(text: str, current: int) -> int | None:
    """Resolve a permissions string to an absolute mode int against *current*.

    Tries octal first, then the intent-keyword vocabulary, then chmod-style
    symbolic clauses. Returns ``None`` for anything none of the three accept.
    """
    octal = _parse_octal(text)
    if octal is not None:
        return octal

    spec = _INTENT.get(text.lower(), text)
    return _apply_symbolic(spec, current)


def _apply_symbolic(spec: str, current: int) -> int | None:
    """Apply comma-separated chmod clauses to *current*, returning the new mode.

    Each clause is ``[ugoa]*[+-=][rwx]+`` (no target letters = ``a``). ``+`` adds
    bits, ``-`` clears them, ``=`` sets the targeted who-classes to exactly the
    given perms. Returns ``None`` if any clause is malformed (copy-forms like
    ``u=g`` and modifiers like ``X``/``s``/``t`` are unsupported → ``None``).
    """
    mode = current & 0o7777
    clauses = spec.split(",")
    if not all(clauses):
        return None
    for clause in clauses:
        m = _CLAUSE_RE.match(clause.strip())
        if m is None:
            return None
        who, op, perms = m.groups()
        who_letters = who or "a"

        perm_value = 0
        for p in perms:
            perm_value |= _PERM_BITS[p]

        mask = 0
        for w in who_letters:
            who_mask = _WHO_BITS[w]
            mask |= who_mask & (perm_value * 0o111)

        if op == "+":
            mode |= mask
        elif op == "-":
            mode &= ~mask
        else:  # "="
            clear = 0
            for w in who_letters:
                clear |= _WHO_BITS[w]
            mode = (mode & ~clear) | mask
    return mode


def _parse_octal(text: str) -> int | None:
    """Parse a 3- or 4-digit octal string into an int; reject anything else.

    Accepts ``"755"`` and ``"0755"`` alike. Rejects empty strings, non-octal
    characters, and any value outside ``0o000``--``0o7777``.
    """
    if len(text) not in (3, 4):
        return None
    if not all(c in "01234567" for c in text):
        return None
    value = int(text, 8)
    if value < 0 or value > _MAX_OCTAL:
        return None
    return value


def _format_octal(st_mode: int) -> str:
    return format(st_mode & 0o7777, "04o")


def _format_symbolic(st_mode: int) -> str:
    bits = st_mode & 0o777
    out = []
    for shift in (6, 3, 0):
        triplet = (bits >> shift) & 0o7
        out.append("r" if triplet & 0o4 else "-")
        out.append("w" if triplet & 0o2 else "-")
        out.append("x" if triplet & 0o1 else "-")
    return "".join(out)
