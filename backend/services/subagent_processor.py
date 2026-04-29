"""
SubagentProcessor — Focused subagent execution with extended limits.

MessageProcessor v2 subclass. CHANNEL is flat 'subagent'. sub_id is stored
in metadata only, not embedded in the channel string.

North star: /Volumes/llm/chalie-plans/message-processing.md
"""

import logging

from services.message_processor import MessageProcessor
from services.system_message_prompt import SubagentSystemMessagePrompt

logger = logging.getLogger(__name__)

_BLOCKED: frozenset[str] = frozenset({"subagent", "save_graph", "detect_pattern"})


class SubagentProcessor(MessageProcessor):
    """MessageProcessor subclass for focused subagent task execution.

    Extended safety caps for long-running tasks:
      MAX_ITERATIONS = 50   (vs. base 30)
      MAX_TIMEOUT    = 7200 (2 hours, vs. base 900)

    ALWAYS_AVAILABLE is minimal — only find_tools. Everything else is
    pre-seeded at construction via find_tools_query() against DISCOVERABLE,
    or discovered at runtime via find_tools.

    _BLOCKED names are filtered from pre-seed results and from any future
    find_tools discovery accumulated in self._discovered_tools.
    """

    CHANNEL = 'subagent'
    ROLE = 'subagent'
    MAX_ITERATIONS = 50
    MAX_TIMEOUT = 7200  # hard ceiling; per-instance override clamps to this
    SYSTEM_PROMPT_CLASS = SubagentSystemMessagePrompt
    ALWAYS_AVAILABLE: list[str] = ["find_tools"]
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
        max_timeout_override: int | None = None,
    ) -> None:
        super().__init__(raw_input, metadata)

        # Clamp override to [0, MAX_TIMEOUT]; None means use class default.
        if max_timeout_override is not None:
            self._max_timeout_override: int | None = min(
                int(max_timeout_override), self.MAX_TIMEOUT
            )
        else:
            self._max_timeout_override = None

        # Pre-seed: run find_tools_query once at construction time against
        # DISCOVERABLE (minus blocklist). Cached on the instance; injected
        # into the first iteration via getDynamicTools().
        allow = [n for n in self.DISCOVERABLE if n not in _BLOCKED]
        self._preseeded_tools: list[dict] = self._build_preseed(raw_input, allow)

    @property
    def _max_timeout_seconds(self) -> int:
        return self._max_timeout_override if self._max_timeout_override is not None else self.MAX_TIMEOUT

    def _build_preseed(self, query: str, allow: list[str]) -> list[dict]:
        """Run find_tools_query and resolve top-3 names to ability schemas."""
        try:
            from abilities.find_tools import find_tools_query
            from abilities._registry import AbilityRegistry

            names = find_tools_query(query, allow=allow, limit=3)
            schemas: list[dict] = []
            for name in names:
                if name in _BLOCKED:
                    continue
                try:
                    a = AbilityRegistry.get(name)
                    schemas.append({
                        'name': a.NAME,
                        'description': a.SUMMARY,
                        'input_schema': a.INPUT_SCHEMA,
                    })
                except KeyError:
                    logger.debug("[SubagentProcessor] Pre-seed: no ability for '%s'", name)
            return schemas
        except Exception as exc:
            logger.debug("[SubagentProcessor] Pre-seed failed: %s", exc)
            return []

    def getDynamicTools(self) -> list[dict]:
        """Merge pre-seeded tools with runtime-discovered tools, filtering blocklist."""
        filtered_discovered = [
            t for t in self._discovered_tools
            if t.get('name') not in _BLOCKED
        ]
        # Pre-seeded tools are injected at the start; runtime-discovered tools
        # accumulate as the subagent calls find_tools during execution.
        # Deduplication in getTools() (first-seen wins) handles overlaps.
        return self._preseeded_tools + filtered_discovered

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
