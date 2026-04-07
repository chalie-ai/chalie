"""
Context Assembly Service - Unified multi-memory context retrieval and ranking.

Orchestrates retrieval from working memory (compaction + transcript) and
moments, with budget-constrained context payload.
"""

import logging
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from services.topic_context import TopicContext

try:
    from services.telemetry_service import (
        get_telemetry_collector,
        CONTEXT_ASSEMBLY,
    )
    _TELEMETRY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _TELEMETRY_AVAILABLE = False
    logging.debug(f"Telemetry service import unavailable: {e}")

class ContextAssemblyService:
    """Orchestrates context retrieval from all memory systems."""

    # Default weights for each memory type (higher = more important)
    DEFAULT_WEIGHTS = {
        'working_memory': 1.0,
        'moments': 0.95,
    }

    def __init__(self, config: dict):
        """
        Initialize context assembly service.

        Args:
            config: Configuration dict with:
                - context_weights: dict of memory type weights
                - max_context_tokens: approximate token budget
                - max_working_memory_turns: turns for working memory
        """
        self.config = config
        self.weights = config.get('context_weights', self.DEFAULT_WEIGHTS)
        self.max_context_tokens = config.get('max_context_tokens', 4000)

    def assemble(
        self,
        prompt: str,
        topic: str,
        act_history: str = "",
        thread_id: str = None,
        recent_visible_context: list = None,
        message_embedding=None,
        context: 'TopicContext' = None,
    ) -> Dict[str, str]:
        """
        Assemble context from all memory types.

        Args:
            prompt: User's current prompt
            topic: Current conversation topic
            act_history: Previous ACT loop history
            thread_id: Optional thread ID for working memory retrieval
            recent_visible_context: Optional last exchanges from expired thread

        Returns:
            Dict with context sections:
            {
                'working_memory': str,
                'episodes': str,
                'concepts': str,
                'previous_session': str,
                'total_tokens_est': int
            }
        """
        sections = {}

        # Use TopicContext when available, fall back to loose params
        if context is not None:
            wm_identifier = context.wm_identifier
        else:
            wm_identifier = thread_id if thread_id else topic

        sections['working_memory'] = self._get_working_memory(wm_identifier, topic, context=context)
        sections['moments'] = self._get_moments(prompt, context=context)

        # Inject recent visible context from previous session (visual continuity bridge)
        if recent_visible_context:
            lines = ["## Recent conversation (previous session):"]
            for ex in recent_visible_context[-2:]:
                lines.append(f"User: {ex.get('prompt', '')}")
                lines.append(f"Assistant: {ex.get('response', '')}")
            sections['previous_session'] = "\n".join(lines)
        else:
            sections['previous_session'] = ""

        # Self-awareness (interoception — only when noteworthy)
        try:
            from services.self_model_service import SelfModelService
            service = SelfModelService()
            if service.has_noteworthy_state():
                sections['self_awareness'] = service.format_for_prompt()
            else:
                sections['self_awareness'] = ""
        except Exception as e:
            logging.debug(f"[CONTEXT] Self-awareness retrieval failed: {e}")
            if context is not None:
                context.record_failure('self_awareness', e)
            sections['self_awareness'] = ""

        # Estimate total tokens
        total_tokens = sum(self._estimate_tokens(s) for s in sections.values() if isinstance(s, str))
        sections['total_tokens_est'] = total_tokens

        # Apply budget constraints if needed
        if total_tokens > self.max_context_tokens:
            sections = self._apply_budget(sections)

        if _TELEMETRY_AVAILABLE:
            try:
                get_telemetry_collector().record(
                    CONTEXT_ASSEMBLY,
                    {
                        "total_tokens_est": sections.get("total_tokens_est", 0),
                        "has_working_memory": bool(sections.get("working_memory")),
                        "has_moments": bool(sections.get("moments")),
                    },
                )
            except Exception as _tel_err:
                logging.debug(
                    "[CONTEXT] CONTEXT_ASSEMBLY telemetry emit failed (non-fatal): %s",
                    _tel_err,
                )

        return sections

    def _get_working_memory(self, identifier: str, topic: str = None, context: 'TopicContext' = None) -> str:
        """Retrieve working memory from compaction + transcript.

        Uses stored compaction (summarized older turns) plus recent transcript
        entries since the compaction watermark. Falls back to legacy MemoryStore
        FIFO when no transcript data exists yet.

        Args:
            identifier: Thread ID or topic string (used for legacy fallback).
            topic: Conversation topic for transcript/compaction lookup.

        Returns:
            Formatted working memory string, or empty string on error.
        """
        effective_topic = topic or identifier

        try:
            from services import compaction_service, transcript_service
            from services.llm_service import estimate_tokens

            compaction = compaction_service.get_compaction(effective_topic)
            watermark = compaction['compacted_up_to_id'] if compaction else 0

            entries = transcript_service.get_recent(
                effective_topic, limit=50, since_id=watermark,
            )

            if not compaction and not entries:
                return self._get_working_memory_legacy(identifier)

            parts = []

            if compaction and compaction.get('compacted_text'):
                parts.append(
                    f"## Conversation History Summary\n{compaction['compacted_text']}"
                )

            if entries:
                turn_budget = self.max_context_tokens // 2
                turn_lines = ["## Recent Conversation"]
                used_tokens = 0

                selected = []
                for entry in reversed(entries):
                    content = entry.get('content', '')
                    entry_tokens = estimate_tokens(content)
                    if used_tokens + entry_tokens > turn_budget:
                        break
                    selected.append(entry)
                    used_tokens += entry_tokens

                selected.reverse()
                for entry in selected:
                    role = entry.get('role', 'unknown').capitalize()
                    content = entry.get('content', '')
                    tool_name = entry.get('tool_name')
                    if tool_name:
                        turn_lines.append(f"{role} ({tool_name}): {content}")
                    else:
                        turn_lines.append(f"{role}: {content}")

                if len(turn_lines) > 1:
                    parts.append("\n".join(turn_lines))

            return "\n\n".join(parts) if parts else ""

        except Exception as e:
            logging.debug(f"[CONTEXT] Transcript-based working memory failed, using legacy: {e}")
            if context is not None:
                context.record_failure('working_memory', e)
            return self._get_working_memory_legacy(identifier, context=context)

    def _get_working_memory_legacy(self, identifier: str, context: 'TopicContext' = None) -> str:
        """Legacy working memory from MemoryStore FIFO buffer.

        Args:
            identifier: Thread ID or topic string.

        Returns:
            Formatted working memory string, or empty string on error.
        """
        try:
            from services.working_memory_service import WorkingMemoryService
            max_turns = self.config.get('max_working_memory_turns', 10)
            wm = WorkingMemoryService(max_turns=max_turns)
            return wm.get_formatted_context(identifier)
        except Exception as e:
            logging.debug(f"[CONTEXT] Working memory unavailable: {e}")
            if context is not None:
                context.record_failure('working_memory_legacy', e)
            return ""

    def _get_moments(self, prompt: str, context: 'TopicContext' = None) -> str:
        """Retrieve relevant pinned moments via semantic search.

        Args:
            prompt: Current user prompt used as the semantic search query.

        Returns:
            Formatted moments string with header, or empty string if none found.
        """
        try:
            from services.moment_service import MomentService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            service = MomentService(db_service)
            moments = service.search_moments(prompt, limit=2)

            if not moments:
                return ""

            # Only include moments above similarity threshold (distance < 0.6)
            relevant = [m for m in moments if m.get("distance", 1.0) < 0.6]
            if not relevant:
                return ""

            lines = ["## Pinned Moments"]
            for m in relevant:
                summary = m.get("summary") or m.get("message_text", "")[:100]
                pinned_at = m.get("pinned_at")
                date_str = ""
                if pinned_at:
                    try:
                        if hasattr(pinned_at, 'strftime'):
                            date_str = pinned_at.strftime("%d %b %Y")
                        else:
                            date_str = str(pinned_at)[:10]
                    except Exception as e:
                        logging.debug(f"[CONTEXT] Moment date formatting failed: {e}")
                        date_str = ""
                lines.append(f"- {summary} (pinned {date_str})")

            return "\n".join(lines)
        except Exception as e:
            logging.debug(f"[CONTEXT] Moments unavailable: {e}")
            if context is not None:
                context.record_failure('moments', e)
            return ""

    def _estimate_tokens(self, text: str) -> int:
        """Produce a rough token estimate using 4 characters per token.

        Args:
            text: Text to estimate token count for.

        Returns:
            Estimated token count as an integer.
        """
        if not text:
            return 0
        return len(text) // 4

    def _apply_budget(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trim context sections to fit within token budget.
        Trims lowest-weight sections first.

        Args:
            sections: Dict of context sections

        Returns:
            Budget-constrained sections
        """
        memory_types = ['working_memory', 'moments', 'previous_session']

        # Sort by weight ascending (trim lowest weight first)
        sorted_types = sorted(memory_types, key=lambda t: self.weights.get(t, 0.5))

        current_tokens = sections.get('total_tokens_est', 0)
        budget = self.max_context_tokens

        for mem_type in sorted_types:
            if current_tokens <= budget:
                break

            section_text = sections.get(mem_type, '')
            section_tokens = self._estimate_tokens(section_text)

            if section_tokens > 0:
                # Truncate this section proportionally
                excess = current_tokens - budget
                if section_tokens <= excess:
                    # Remove entire section
                    sections[mem_type] = ""
                    current_tokens -= section_tokens
                else:
                    # Truncate section
                    keep_ratio = 1.0 - (excess / section_tokens)
                    keep_chars = int(len(section_text) * keep_ratio)
                    sections[mem_type] = section_text[:keep_chars] + "\n[truncated]"
                    current_tokens = budget

        sections['total_tokens_est'] = current_tokens
        return sections
