"""FilePermissionsAbility — chmod a single file or directory.

Safe alternative to invoking ``bash`` for ``chmod``. Returns both the
``permissions_before`` and ``permissions_after`` octal strings so the audit
trail (and the user) can see exactly what changed.
"""

import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from abilities._ability import Ability

logger = logging.getLogger(__name__)

_MAX_OCTAL = 0o7777


class FilePermissionsAbility(Ability):
    NAME = "file_permissions"
    SUMMARY = "Change Unix-style filesystem permissions (chmod) on a file or directory."
    SEARCH_TOOLTIP = "Change file permissions (chmod)"
    EXAMPLES: ClassVar[list[str]] = [
        "make this script executable",
        "chmod 755 /home/user/scripts/deploy.sh",
        "make this file read-only",
        "give this file 644 permissions",
        "set permissions to 600 on my ssh key",
        "make this binary executable so I can run it",
        "lock down this config file to owner-only",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file or directory whose permissions you want to change.",
            },
            "permissions": {
                "type": "string",
                "description": (
                    "Octal permissions string, e.g. '755' for rwxr-xr-x, '644' for"
                    " rw-r--r--, '600' for rw-------. Leading 0 optional. Must be"
                    " exactly 3 or 4 octal digits."
                ),
            },
        },
        "required": ["path", "permissions"],
    }

    def run(self, params: dict) -> dict:
        path_str = (params.get("path") or "").strip()
        perm_str = (params.get("permissions") or "").strip()

        if not path_str:
            return {"text": json.dumps({"error": "path-required"})}
        if not perm_str:
            return {"text": json.dumps({"error": "permissions-required"})}

        mode = _parse_octal(perm_str)
        if mode is None:
            return {"text": json.dumps({"error": "invalid-permissions", "permissions": perm_str})}

        target = Path(os.path.expanduser(path_str)).resolve()
        target_str = str(target)

        if not target.exists():
            return {"text": json.dumps({"error": "path-not-found", "path": path_str})}

        try:
            before = _format_octal(target.stat().st_mode)
            os.chmod(target, mode)
            after = _format_octal(target.stat().st_mode)
        except OSError as exc:
            logger.exception("[FILE_PERMISSIONS] chmod failed for path=%r: %s", target_str, exc)
            return {"text": json.dumps({"error": "chmod-failed", "detail": str(exc)})}

        return {"text": json.dumps({
            "status": "success",
            "path": target_str,
            "permissions_before": before,
            "permissions_after": after,
        })}


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
    """Return the low 4 octal digits of ``st_mode`` (suid/sgid/sticky + rwx)."""
    return format(st_mode & 0o7777, "04o")
