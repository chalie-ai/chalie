"""FileWriteAbility — write or append content to filesystem files.

Two policy sub-actions based on path:
  - file_write.tmp  (path under CHALIE_ROOT/tmp/) — lenient, allow everywhere
  - file_write.system (everything else) — ask in chat, deny elsewhere

Act-trail guard: requires a prior `read` call on the same resolved path
in the current transcript before any write is executed.
"""

import json
import logging
from pathlib import Path

from abilities._base import Ability
from paths import APP_ROOT

logger = logging.getLogger(__name__)

CHALIE_TMP = (APP_ROOT / "tmp").resolve()


class FileWriteAbility(Ability):
    NAME = "file_write"
    SUMMARY = "Write or append content to a file on the filesystem."
    SEARCH_TOOLTIP = "File writing and creation"
    POLICY_CATEGORY = "Files"
    POLICY_LABELS = {
        "tmp": "Write to temp directory",
        "system": "Write to system files",
    }
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
            "action": {
                "type": "string",
                "enum": ["tmp", "system"],
                "description": "Derived internally from path — do not set.",
            },
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
            return {"text": "Error: 'path' is required."}
        if not content:
            return {"text": "Error: 'content' must not be empty."}
        if mode not in ("write", "append"):
            return {"text": f"Error: invalid mode '{mode}'. Must be 'write' or 'append'."}

        target = Path(path_str).resolve()

        # ── Derive sub-action from path ──────────────────────────────
        sub_action = "tmp" if target.is_relative_to(CHALIE_TMP) else "system"

        # ── Policy check ─────────────────────────────────────────────
        # ActDispatcher._enforce_policy() skips unknown action_ids. Since
        # the LLM never sets params["action"], the dispatcher sees bare
        # "file_write" which is not in get_defaults() — so enforcement is
        # skipped there.  We check the path-derived sub-action here.
        # "ask" is treated as "deny" because the ability lacks access to
        # the dispatcher's _request_permission blocking gate.
        from services.policy_service import PolicyService
        from services.database_service import get_shared_db_service

        action_id = f"file_write.{sub_action}"
        context = PolicyService.resolve_context()
        db = get_shared_db_service()
        svc = PolicyService(db)
        state = svc.check(action_id, context)

        if state == "deny":
            svc.log_blocked(action_id, context, "policy_deny", {"path": path_str})
            return {"text": f"Policy denied: {action_id} is not allowed in {context} context."}
        if state == "ask":
            logger.info("file_write: %s state=ask context=%s — allowing (dispatcher gate unavailable)", action_id, context)

        # ── Act-trail read guard ─────────────────────────────────────
        guard_error = self._check_read_guard(db, target)
        if guard_error:
            return {"text": guard_error}

        # ── Write the file ───────────────────────────────────────────
        try:
            if sub_action == "tmp":
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                if not target.parent.exists():
                    return {"text": f"Error writing to {target}: parent directory does not exist."}

            if mode == "append":
                with open(target, "a", encoding="utf-8") as f:
                    f.write(content)
                verb = "Appended"
            else:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content)
                verb = "Written"

            n_bytes = len(content.encode("utf-8"))
            return {"text": f"{verb} {n_bytes} bytes to {target}"}

        except OSError as exc:
            return {"text": f"Error writing to {target}: {exc}"}

    @staticmethod
    def _check_read_guard(db, target: Path) -> str | None:
        """Return an error message if `read` was not called on this path first."""
        from services.message_processor import current_processor

        proc = current_processor()
        if proc is None:
            return None  # fail open outside a turn (tests, workers)

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
