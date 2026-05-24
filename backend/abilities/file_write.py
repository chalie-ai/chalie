"""FileWriteAbility — write or append content to a file at a caller-supplied path.

Act-trail guard: requires a prior ``read`` call on the same resolved path in
the current transcript before any write is executed.
"""

import json
import logging
from pathlib import Path

from abilities._base import Ability

logger = logging.getLogger(__name__)


class FileWriteAbility(Ability):
    NAME = "file_write"
    SUMMARY = "Write or append content to a file on the filesystem."
    SEARCH_TOOLTIP = "File writing and creation"
    POLICY_CATEGORY = "Files"
    POLICY_LABELS = {"": "Write to files"}
    EXAMPLES = [
        "save this text to a file",
        "write this configuration to /etc/myapp/config.yaml",
        "create a new script file",
        "append this line to the log file",
        "save the output to a temporary file",
        "write this JSON to a file so I can use it later",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to write to.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append"],
                "description": "Write mode: 'write' overwrites, 'append' adds to end. Default: write.",
            },
        },
        "required": ["path", "content"],
    }
    TIMEOUT = 15

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        path_str = params.get("path", "")
        content = params.get("content", "")
        mode = params.get("mode", "write")

        if not path_str:
            return {"text": "Error: 'path' must not be empty."}
        if not content:
            return {"text": "Error: 'content' must not be empty."}
        if mode not in ("write", "append"):
            return {"text": f"Error: invalid mode '{mode}'. Must be 'write' or 'append'."}

        target = Path(path_str).resolve()

        if not target.parent.exists():
            return {"text": f"Error writing to {target}: parent directory does not exist."}

        from services.database_service import get_shared_db_service
        guard_error = self._check_read_guard(get_shared_db_service(), target)
        if guard_error:
            return {"text": guard_error}

        try:
            if mode == "append":
                with open(target, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content)

            stat = target.stat()
            size_kb = round(stat.st_size / 1024, 2)
            perms = oct(stat.st_mode)[-3:]
            return {"text": json.dumps({
                "status": "success",
                "path": str(target),
                "permission": perms,
                "size": f"{size_kb}kb",
            })}

        except OSError as exc:
            return {"text": f"Error writing to {target}: {exc}"}

    @staticmethod
    def _check_read_guard(db, target: Path) -> str | None:
        """Return an error message if ``read`` was not called on this path first."""
        from services.message_processor import current_processor

        proc = current_processor()
        if proc is None:
            return None

        transcript_id = getattr(proc, "_uid", None)
        if transcript_id is None:
            logger.warning("file_write read-guard: active processor has no _uid — guard bypassed")
            return None

        target_str = str(target)
        rows = db.fetch_all(
            "SELECT params FROM tool_calls "
            "WHERE transcript_id = ? AND tool_name = 'read'",
            (transcript_id,),
        )
        for row in rows:
            try:
                p = json.loads(row["params"])
            except (json.JSONDecodeError, TypeError):
                continue
            source = p.get("source") or p.get("path") or p.get("url", "")
            if not source:
                continue
            try:
                if Path(source).resolve() == target:
                    return None
            except (ValueError, OSError):
                if source == target_str:
                    return None

        return (
            f"You need to call the 'read' tool with parameter '{target}' "
            f"before using 'file_write'."
        )
