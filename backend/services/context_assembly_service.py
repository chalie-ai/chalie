"""
Context Assembly Service - Unified multi-memory context retrieval and ranking.

Orchestrates retrieval from all memory types (working memory, facts, gists,
episodes, concepts), ranks with unified scoring, and produces budget-constrained
context payload.
"""

import logging
from typing import Dict, Any, Optional, List


class ContextAssemblyService:
    """Orchestrates context retrieval from all memory systems."""

    # Default weights for each memory type (higher = more important)
    DEFAULT_WEIGHTS = {
        'working_memory': 1.0,
        'facts': 0.9,
        'gists': 0.8,
        'episodes': 0.7,
        'concepts': 0.6
    }

    def __init__(self, config: dict):
        """
        Initialize context assembly service.

        Args:
            config: Configuration dict with:
                - context_weights: dict of memory type weights
                - max_context_tokens: approximate token budget
                - max_working_memory_turns: turns for working memory
                - max_gists: max gists to retrieve
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
                'facts': str,
                'gists': str,
                'episodes': str,
                'concepts': str,
                'previous_session': str,
                'total_tokens_est': int
            }
        """
        sections = {}

        # Use thread_id for working memory if available
        wm_identifier = thread_id if thread_id else topic
        sections['working_memory'] = self._get_working_memory(wm_identifier)
        sections['facts'] = self._get_facts(topic)
        sections['gists'] = self._get_gists(topic)
        sections['episodes'] = self._get_episodes(prompt, topic, act_history)
        sections['concepts'] = self._get_concepts(prompt, topic, act_history)

        # Inject recent visible context from previous session (visual continuity bridge)
        if recent_visible_context:
            lines = ["## Recent conversation (previous session):"]
            for ex in recent_visible_context[-2:]:
                lines.append(f"User: {ex.get('prompt', '')}")
                lines.append(f"Assistant: {ex.get('response', '')}")
            sections['previous_session'] = "\n".join(lines)
        else:
            sections['previous_session'] = ""

        # Estimate total tokens
        total_tokens = sum(self._estimate_tokens(s) for s in sections.values() if isinstance(s, str))
        sections['total_tokens_est'] = total_tokens

        # Apply budget constraints if needed
        if total_tokens > self.max_context_tokens:
            sections = self._apply_budget(sections)

        return sections

    def _get_working_memory(self, identifier: str) -> str:
        """Retrieve working memory context. Accepts thread_id or topic."""
        try:
            from services.working_memory_service import WorkingMemoryService
            max_turns = self.config.get('max_working_memory_turns', 10)
            wm = WorkingMemoryService(max_turns=max_turns)
            return wm.get_formatted_context(identifier)
        except Exception as e:
            logging.debug(f"[CONTEXT] Working memory unavailable: {e}")
            return ""

    def _get_facts(self, topic: str) -> str:
        """Retrieve facts context."""
        try:
            from services.fact_store_service import FactStoreService
            fs = FactStoreService()
            return fs.get_facts_formatted(topic)
        except Exception as e:
            logging.debug(f"[CONTEXT] Fact store unavailable: {e}")
            return ""

    def _get_gists(self, topic: str) -> str:
        """Retrieve gist context."""
        try:
            from services.gist_storage_service import GistStorageService
            min_confidence = self.config.get('min_gist_confidence', 7)
            max_gists = self.config.get('max_gists', 8)
            gs = GistStorageService(min_confidence=min_confidence, max_gists=max_gists)

            gists = gs.get_latest_gists(topic)
            if gists:
                # Filter out cold_start gists — internal metadata, not conversation context
                real_gists = [g for g in gists if g.get('type') != 'cold_start']
                if real_gists:
                    lines = ["## Recent Conversation Gists"]
                    for gist in real_gists:
                        lines.append(
                            f"- [{gist['type']}] {gist['content']} "
                            f"(confidence: {gist['confidence']})"
                        )
                    return "\n".join(lines)

            # Fallback to last message
            last_message = gs.get_last_message(topic)
            if last_message:
                return (
                    f"## Last Exchange\n"
                    f"User: {last_message['prompt']}\n"
                    f"Assistant: {last_message['response']}"
                )

            return "No previous conversation context available"
        except Exception as e:
            logging.debug(f"[CONTEXT] Gist store unavailable: {e}")
            return "No previous conversation context available"

    def _get_episodes(self, prompt: str, topic: str, act_history: str = "") -> str:
        """Retrieve episodic memory context."""
        try:
            from services.episodic_retrieval_service import EpisodicRetrievalService
            from services.database_service import DatabaseService, get_merged_db_config
            from services.config_service import ConfigService

            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            retrieval = EpisodicRetrievalService(db_service, episodic_config)

            # Extract semantic concepts from act_history for boost
            semantic_concepts = self._extract_semantic_from_history(act_history)

            episodes = retrieval.retrieve_episodes(
                query_text=prompt,
                topic=topic,
                limit=3,
                semantic_concepts=semantic_concepts if semantic_concepts else None
            )

            if not episodes:
                return ""

            lines = ["\n## Relevant Past Experiences"]
            for i, ep in enumerate(episodes, 1):
                lines.append(f"{i}. {ep['gist']}")
                lines.append(f"   - Outcome: {ep['outcome']}")
                lines.append(f"   - Salience: {ep['salience']}/10")

            return "\n".join(lines)
        except Exception as e:
            logging.debug(f"[CONTEXT] Episodic memory unavailable: {e}")
            return ""

    def _get_concepts(self, prompt: str, topic: str, act_history: str = "") -> str:
        """Retrieve semantic concept context from act_history if available."""
        # Concepts are currently injected via act_history semantic_query results
        # This is a placeholder for when dedicated concept injection is needed
        return ""

    def _extract_semantic_from_history(self, act_history: str) -> List[Dict]:
        """Parse semantic_query results from act_history string."""
        import re
        concepts = []
        pattern = r'-\s+([^:]+):\s+([^(]+)\s+\(strength:'
        matches = re.findall(pattern, act_history)
        for name, definition in matches:
            concepts.append({
                'name': name.strip(),
                'definition': definition.strip()
            })
        return concepts

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
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
        memory_types = ['working_memory', 'facts', 'gists', 'episodes', 'concepts', 'previous_session']

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
