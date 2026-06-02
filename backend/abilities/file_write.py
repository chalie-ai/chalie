"""FileWriteAbility — write content to a file at a caller-supplied absolute path.

Act-trail guard: if the target file already exists, a prior ``read`` call on
the same resolved path in the current transcript is required before the write
is executed.
"""

import json
import logging
from pathlib import Path

from abilities._base import Ability

logger = logging.getLogger(__name__)


class FileWriteAbility(Ability):
    NAME = "file_write"
    SUMMARY = "Write content to a file. You MUST call the 'read' tool on the target path before writing."
    SEARCH_TOOLTIP = "File writing and creation"
    POLICY_CATEGORY = "Files"
    POLICY_LABELS = {"": "Write to files"}
    EXAMPLES = [
        "save this text to a file",
        "write this configuration to /etc/myapp/config.yaml",
        "create a new script file",
        "save the output to a temporary file",
        "write this JSON to a file so I can use it later",
        "overwrite the contents of that file",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to write to.",
            },
            "contents": {
                "type": "string",
                "description": "Content to write to the file.",
            },
        },
        "required": ["path", "contents"],
    }

    def run(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        path_str = params.get("path", "")
        contents = params.get("contents", "")

        if not path_str:
            return {"text": "Error: 'path' is required."}
        if not contents:
            return {"text": "Error: 'contents' is required."}

        target = Path(path_str).resolve()

        if target.exists():
            from services.database_service import get_shared_db_service
            guard_error = self._check_read_guard(get_shared_db_service(), target)
            if guard_error:
                return {"text": guard_error}

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(contents)
            return {"text": json.dumps({
                "success": True,
                "path": str(target),
                "file_size": target.stat().st_size,
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
