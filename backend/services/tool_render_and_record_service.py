"""ToolRenderAndRecordService — renders tool output and records to tool_calls."""

import json
import logging

from services.database_service import get_shared_db_service
from services.time_utils import utc_now

logger = logging.getLogger(__name__)


class ToolRenderAndRecordService:

    def __init__(self, tool_name: str, params: dict, result: str,
                 ephemeral: bool, transcript_id: int | None):
        self._tool_name = tool_name
        self._params = params
        self._result = result
        self._ephemeral = ephemeral
        self._transcript_id = transcript_id

    def _render(self) -> str:
        parts = []
        for k, v in self._params.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f'{k}={v}')
        param_str = ','.join(parts)
        return f'[{self._tool_name}({param_str})] {self._result}'

    def _record(self) -> None:
        # Internal processors with SKIP_TRANSCRIPT_WRITE=True never write a
        # transcript row, so self._uid is None for the entire turn. The
        # tool_calls.transcript_id column is NOT NULL; inserting a row would
        # raise IntegrityError, which the broad except below would swallow
        # silently — leaving operators believing tool calls are recorded
        # when they are not. Guard early and log at DEBUG so the behaviour
        # is observable without spamming logs.
        if self._transcript_id is None:
            logger.debug(
                "[TOOL RENDER] Skipping record (no transcript_id): tool=%s",
                self._tool_name,
            )
            return
        db = get_shared_db_service()
        params_json = json.dumps(self._params)
        now = utc_now().isoformat()
        try:
            with db.connection() as conn:
                conn.execute(
                    "INSERT INTO tool_calls "
                    "(transcript_id, tool_name, params, result, "
                    "ephemeral, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (self._transcript_id, self._tool_name, params_json,
                     self._result, 1 if self._ephemeral else 0, now),
                )
                conn.commit()
        except Exception:
            logger.exception(
                "[TOOL RENDER] Failed to record: tool=%s transcript=%s",
                self._tool_name, self._transcript_id,
            )

    def render_and_record(self) -> str:
        self._record()
        return self._render()

    def renderAndRecord(self) -> str:  # noqa: N802
        """Backward-compat shim — use ``render_and_record()``."""
        return self.render_and_record()

    @staticmethod
    def render_static(tool_name: str, params: dict, result: str) -> str:
        """Render without recording — for replaying historical tool_calls."""
        parts = []
        for k, v in params.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f'{k}={v}')
        param_str = ','.join(parts)
        return f'[{tool_name}({param_str})] {result}'
