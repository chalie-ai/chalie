"""
SubagentProcessor — Focused subagent execution with per-type tool surfaces.

MessageProcessor v2 subclass. CHANNEL is flat 'subagent'. sub_id is stored
in metadata only, not embedded in the channel string.

Overflow strategy: SubagentProcessor overrides ``_handle_overflow()`` to call
``SubagentTrailCompactionProcessor`` instead of the UMP continuity path. The
subagent compresses its accumulated tool trail in-place and continues the
current iteration (no full ACT restart). An audit row is written to tool_calls
for traceability. Design choice: override (not strategy injection) because the
subagent override is simple, tightly coupled to SubagentProcessor state
(``_act_trail`` replacement), and a strategy object would add indirection
without reducing coupling.
"""

import json
import logging
import time

from abilities.subagent import SUBAGENT_TYPES
from services.message_processor import MessageProcessor
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_BLOCKED: frozenset[str] = frozenset({"subagent"})


class SubagentProcessor(MessageProcessor):
    """MessageProcessor subclass for focused subagent task execution.

    Extended safety caps:
      MAX_ITERATIONS = 50   (vs. base 30)
      ITERATION_TIMEOUT inherited from base (1800s per-iteration wall)

    ALWAYS_AVAILABLE is set per-instance from SUBAGENT_TYPES[agent_type]['native_tools'].
    DISCOVERABLE covers everything not in the blocklist or type's native list.
    Per-instance deadline (self._deadline) enforces the type's wall-clock budget.

    _BLOCKED names are filtered from DISCOVERABLE and any find_tools discovery.
    """

    CHANNEL = 'subagent'
    ROLE = 'subagent'
    LOG_LABEL = 'subagent'
    USAGE_CLASS = 'subagent'
    MAX_ITERATIONS = 50
    # Inherits DISCOVERABLE from MessageProcessor; _BLOCKED filters subagent.
    _BLOCKED: frozenset[str] = _BLOCKED

    def __init__(
        self,
        raw_input: str,
        metadata: dict | None = None,
        agent_type: str = "general_purpose",
        max_timeout_override: int | None = None,
    ) -> None:
        super().__init__(raw_input, metadata)

        self.agent_type = agent_type

        # Instance-level tool surface from the type registry.
        self.ALWAYS_AVAILABLE = list(SUBAGENT_TYPES[agent_type]["native_tools"])

        # Per-instance deadline: prefer caller override, else type default.
        timeout_seconds = (
            int(max_timeout_override)
            if max_timeout_override is not None
            else SUBAGENT_TYPES[agent_type]["default_timeout"]
        )
        self._deadline = time.time() + timeout_seconds

    def get_system_prompt(self) -> str:
        """Build the subagent system prompt from the per-type prompt + guardrails."""
        from abilities.subagent import _SHARED_GUARDRAILS
        type_prompt = SUBAGENT_TYPES[self.agent_type]["system_prompt"]
        body = f"{type_prompt}\n\n{_SHARED_GUARDRAILS}"
        return f"{self.get_user_definition()}\n\n{body}"

    def get_user_definition(self) -> str:
        return (
            "The user is 'subagent' — a focused subagent executing a task "
            "delegated by the parent agent."
        )

    def get_user_prompt(self) -> str:
        """Task prompt is the raw input; prepend role prefix."""
        trail = self.get_act_loop_trail()
        parts = [f"subagent: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def _handle_overflow(self) -> bool:
        """Subagent overflow: compress the tool trail in-place via SubagentTrailCompactionProcessor.

        Unlike the UMP path, the subagent does NOT restart the ACT loop and
        does NOT run a full channel compaction. Instead:
        - Feeds the current ``_act_trail`` to SubagentTrailCompactionProcessor.
        - Replaces ``_act_trail`` in-place with the compacted block.
        - Writes an ephemeral=1 audit row (tool_name='subagent_trail_compaction').
        - Returns True so the caller re-renders ``user_body`` and continues
          the current iteration with the compressed trail.

        If compaction fails or produces empty output:
        - Writes a failure audit row.
        - Returns False so the loop breaks to cap exit.
        """
        from services.compaction_message_processor import SubagentTrailCompactionProcessor

        trail_text = '\n'.join(self._act_trail)
        in_entries = len(self._act_trail)

        if not trail_text.strip():
            logger.warning("[COMPACTION] subagent: _handle_overflow called with empty trail")
            return False

        trail_input = f"## Trail\n{trail_text}"
        proc = SubagentTrailCompactionProcessor(raw_input=trail_input)
        compacted_block = ''
        status = 'failure'
        reason = ''

        try:
            compacted_block = (proc.send() or '').strip()
            if compacted_block:
                status = 'success'
            else:
                reason = "compaction LLM returned empty output"
        except Exception as exc:
            reason = str(exc)
            logger.exception(
                "[COMPACTION] subagent: trail compaction failure — reason=%s",
                reason,
            )
        finally:
            self._metrics.merge(proc._metrics)

        out_chars = len(compacted_block)
        self._write_subagent_trail_audit_row(
            in_entries=in_entries, out_chars=out_chars, status=status,
            compacted_block=compacted_block, reason=reason,
        )

        if status == 'success':
            self._act_trail = [compacted_block]
            logger.info(
                "[COMPACTION] subagent: trail compaction success — "
                "in_entries=%d, out_chars=%d",
                in_entries, out_chars,
            )
            return True

        logger.warning(
            "[COMPACTION] subagent: trail compaction failure — reason=%s",
            reason or "unknown",
        )
        return False

    def _write_subagent_trail_audit_row(
        self,
        *,
        in_entries: int,
        out_chars: int,
        status: str,
        compacted_block: str,
        reason: str,
    ) -> None:
        """Write an ephemeral=1 audit row for a subagent trail compaction attempt.

        Never raises — persistence failure is logged and swallowed.
        """
        try:
            from services.database_service import get_shared_db_service
            params: dict = {
                'in_entries': in_entries,
                'out_chars': out_chars,
                'status': status,
            }
            if reason:
                params['reason'] = reason
            db = get_shared_db_service()
            with db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_calls
                        (transcript_id, tool_name, params, result, ephemeral, created_at)
                    VALUES (?, 'subagent_trail_compaction', ?, ?, 1, ?)
                    """,
                    (
                        self._uid,
                        json.dumps(params),
                        compacted_block,
                        utc_now().isoformat(),
                    ),
                )
        except Exception as exc:
            logger.warning(
                "[COMPACTION] subagent: failed to write trail audit row: %s", exc
            )

    def post_turn(self) -> None:
        """Subagent post-turn: metrics only."""
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('subagent_turns_total')
        except Exception as e:
            logger.debug("[Subagent.post_turn] Metrics failed: %s", e)
