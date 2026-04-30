"""
SubagentProcessor — Focused subagent execution with per-type tool surfaces.

MessageProcessor v2 subclass. CHANNEL is flat 'subagent'. sub_id is stored
in metadata only, not embedded in the channel string.

North star: /Volumes/llm/chalie-plans/message-processing.md
"""

import logging
import time

from abilities.subagent import SUBAGENT_TYPES
from services.message_processor import MessageProcessor

logger = logging.getLogger(__name__)

_BLOCKED: frozenset[str] = frozenset({"subagent", "save_graph", "detect_pattern"})


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
    MAX_ITERATIONS = 50
    DISCOVERABLE: list[str] = [
        "browser",
        "code_eval",
        "document",
        "list",
        "memory",
        "news",
        "programming_docs_search",
        "read",
        "review_tool_calls",
        "schedule",
        "search",
        "weather",
    ]

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

    def getSystemPrompt(self) -> str:
        """Build the subagent system prompt from the per-type prompt + guardrails."""
        from abilities.subagent import _SHARED_GUARDRAILS
        type_prompt = SUBAGENT_TYPES[self.agent_type]["system_prompt"]
        body = f"{type_prompt}\n\n{_SHARED_GUARDRAILS}"
        return f"{self.getUserDefinition()}\n\n{body}"

    def getDynamicTools(self) -> list[dict]:
        """Filter blocklist from runtime-discovered tools."""
        return [
            t for t in self._discovered_tools
            if t.get('name') not in _BLOCKED
        ]

    def getUserDefinition(self) -> str:
        return (
            "The user is 'subagent' — a focused subagent executing a task "
            "delegated by the parent agent."
        )

    def getUserPrompt(self) -> str:
        """Task prompt is the raw input; prepend role prefix."""
        trail = self.getActLoopTrail()
        parts = [f"subagent: {self._raw_input}"]
        if trail:
            parts.append(trail)
        return '\n'.join(parts)

    def postTurn(self) -> None:
        """Subagent post-turn: metrics only."""
        try:
            from services.metrics_service import MetricsService
            m = MetricsService()
            m.record_counter('requests_total')
            m.record_counter('subagent_turns_total')
        except Exception as e:
            logger.debug("[Subagent.postTurn] Metrics failed: %s", e, exc_info=True)
