# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import time
import json
from typing import List, Dict
from services.gist_storage_service import GistStorageService
import logging


class ChatHistoryProcessor:
    """Processes chat history for context injection into prompts."""

    def __init__(self, max_exchanges: int = None, max_tokens: int = None):
        self.max_exchanges = max_exchanges
        self.max_tokens = max_tokens

    def process(self, chat_history: list) -> str:
        if not chat_history:
            return "No previous conversation"

        limited_history = self._apply_limits(chat_history)

        lines = []
        for exchange in limited_history:
            if 'prompt' in exchange and 'message' in exchange['prompt']:
                lines.append(f"User: {exchange['prompt']['message']}")
            if 'response' in exchange:
                if isinstance(exchange['response'], dict):
                    if 'message' in exchange['response']:
                        lines.append(f"Assistant: {exchange['response']['message']}")
                    elif 'error' in exchange['response']:
                        lines.append(f"Assistant: [Error: {exchange['response']['error']}]")
                else:
                    lines.append(f"Assistant: {exchange['response']}")

        return "\n".join(lines) if lines else "No previous conversation"

    def _apply_limits(self, chat_history: list) -> list:
        if self.max_exchanges and len(chat_history) > self.max_exchanges:
            chat_history = chat_history[-self.max_exchanges:]
        return chat_history


class FrontalCortexService:
    """Service for generating contextual responses using LLM."""

    def __init__(self, config: dict):
        """Initialize with configuration for LLM."""
        from services.llm_service import create_llm_service
        from services.world_state_service import WorldStateService
        from services.config_service import ConfigService
        from services.database_service import DatabaseService
        from services.episodic_retrieval_service import EpisodicRetrievalService

        self.config = config
        self.llm = create_llm_service(config)
        self.world_state_service = WorldStateService()

        # Initialize gist storage service
        attention_span_minutes = config.get('attention_span_minutes', 30)
        min_confidence = config.get('min_gist_confidence', 7)
        max_gists = config.get('max_gists', 8)
        self.gist_storage = GistStorageService(
            attention_span_minutes=attention_span_minutes,
            min_confidence=min_confidence,
            max_gists=max_gists
        )

        # Initialize episodic memory retrieval
        try:
            from services.database_service import get_merged_db_config

            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            db_config = get_merged_db_config()
            database_service = DatabaseService(db_config)
            self.episodic_retrieval = EpisodicRetrievalService(database_service, episodic_config)
            logging.info("Episodic memory retrieval initialized")
        except Exception as e:
            logging.warning(f"Episodic memory service not available: {e}")
            self.episodic_retrieval = None

    def generate_response(
        self,
        system_prompt_template: str,
        original_prompt: str,
        classification: dict,
        chat_history: list,
        act_history: str = "",
        assembled_context: dict = None,
        relevant_tools: list = None,
        thread_id: str = None,
    ) -> dict:
        """
        Generate a response based on the prompt, classification, and history.

        Args:
            system_prompt_template: Template with {{variable}} placeholders
            original_prompt: The user's original message
            classification: Dict containing topic, confidence, etc.
            chat_history: List of previous exchanges
            act_history: Formatted ACT loop history (defaults to empty)

        Returns:
            dict: {
                'mode': str,
                'modifiers': list,
                'response': str,
                'generation_time': float,
                'actions': list|None
            }
        """
        start_time = time.time()

        # Inject parameters into template
        system_prompt = self._inject_parameters(
            system_prompt_template,
            original_prompt,
            classification,
            chat_history,
            act_history,
            assembled_context=assembled_context,
            relevant_tools=relevant_tools,
            thread_id=thread_id,
        )

        # Generate response from LLM
        try:
            response_text = self.llm.send_message(system_prompt, original_prompt).text
        except Exception as e:
            # Re-raise with context for upstream handling
            raise Exception(f"LLM generation failed: {str(e)}") from e

        generation_time = time.time() - start_time

        # Parse JSON response (format: "json" is set in config)
        # Strip markdown code fences if the model wrapped the JSON
        try:
            stripped = response_text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1]  # remove opening fence line
                stripped = stripped.rsplit("```", 1)[0]  # remove closing fence
            response_data = json.loads(stripped)

            # Extract fields (mode-specific prompts produce simpler output)
            mode = response_data.get('mode', 'RESPOND')
            modifiers = response_data.get('modifiers', [])
            user_response = response_data.get('response', '')
            actions = response_data.get('actions')
            confidence = response_data.get('confidence', 0.5)
            alternative_paths = response_data.get('alternative_paths', [])

            # Infer ACT mode if actions are present but mode wasn't explicit
            # (ACT prompt output contract returns actions without a mode field)
            if actions and isinstance(actions, list) and len(actions) > 0 and mode != 'ACT':
                mode = 'ACT'

            # Validate mode
            valid_modes = ['ACT', 'RESPOND', 'CLARIFY', 'ACKNOWLEDGE', 'IGNORE']
            if mode not in valid_modes:
                logging.warning(f"Invalid mode '{mode}', defaulting to RESPOND")
                mode = 'RESPOND'

            # Validate confidence range
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            # Validate alternative paths structure (needed by act_loop decision gate)
            validated_alternatives = []
            for i, path in enumerate(alternative_paths):
                if not isinstance(path, dict) or 'mode' not in path:
                    continue
                if 'expected_confidence' not in path:
                    path['expected_confidence'] = 0.5
                try:
                    path['expected_confidence'] = float(path['expected_confidence'])
                    path['expected_confidence'] = max(0.0, min(1.0, path['expected_confidence']))
                except (ValueError, TypeError):
                    path['expected_confidence'] = 0.5
                # Validate downstream_mode for ACT paths
                if mode == 'ACT' and path.get('mode') == 'ACT':
                    valid_terminal = ['RESPOND', 'CLARIFY', 'ACKNOWLEDGE', 'IGNORE']
                    if path.get('downstream_mode') not in valid_terminal:
                        path['downstream_mode'] = 'RESPOND'
                validated_alternatives.append(path)
            alternative_paths = validated_alternatives

            # Validate actions for ACT mode
            if mode == 'ACT' and actions and isinstance(actions, list):
                actions = [a for a in actions if isinstance(a, dict) and 'type' in a]
                if not actions:
                    actions = None
            elif mode != 'ACT':
                actions = None

            # Normalize empty actions to None
            if not actions:
                actions = None

            logging.info(f"[MODE:{mode}] Cortex response: mode={mode}, confidence={confidence:.2f}, "
                        f"actions={len(actions) if actions else 0}, alternatives={len(alternative_paths)}")

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}\nRaw response: {response_text[:200]}") from e

        return {
            'mode': mode,
            'modifiers': modifiers,
            'response': user_response,
            'generation_time': generation_time,
            'actions': actions,
            'confidence': confidence,
            'alternative_paths': alternative_paths,
            'downstream_mode': response_data.get('downstream_mode', 'RESPOND')
        }

    def _inject_parameters(
        self,
        template: str,
        original_prompt: str,
        classification: dict,
        chat_history: list,
        act_history: str = "",
        assembled_context: dict = None,
        relevant_tools: list = None,
        thread_id: str = None,
    ) -> str:
        """
        Replace {{variable}} placeholders in template with actual values.

        Args:
            template: System prompt template
            original_prompt: User's message
            classification: Classification result
            chat_history: Previous exchanges (unused - kept for compatibility)
            act_history: ACT loop history (defaults to empty)
            assembled_context: Pre-assembled context dict from ContextAssemblyService

        Returns:
            str: Processed system prompt
        """
        # Determine the topic (classifier may return similar_topic, topic_update, or topic)
        topic = classification.get('similar_topic') or \
                classification.get('topic_update') or \
                classification.get('topic', 'unknown')

        confidence = classification.get('confidence', 0)

        # Use pre-assembled context if available, otherwise build internally
        if assembled_context:
            formatted_context = assembled_context.get('gists', self._get_gist_context(topic))
            episodic_context = assembled_context.get('episodes', '')
            facts_context = assembled_context.get('facts', '')
            working_memory_context = assembled_context.get('working_memory', '')
            world_state = self.world_state_service.get_world_state(topic, thread_id=thread_id)
        else:
            # Parallelize independent I/O calls for faster context assembly
            from concurrent.futures import ThreadPoolExecutor, as_completed

            context_results = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._get_gist_context, topic): 'gists',
                    executor.submit(self.world_state_service.get_world_state, topic, thread_id): 'world_state',
                    executor.submit(self._get_episodic_context, original_prompt, topic, act_history): 'episodes',
                    executor.submit(self._get_facts_context, topic): 'facts',
                    executor.submit(self._get_working_memory_context, topic, thread_id): 'working_memory',
                }
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        context_results[key] = future.result()
                    except Exception as e:
                        logging.warning(f"[FRONTAL CORTEX] Parallel context failed for {key}: {e}")
                        context_results[key] = ''

            formatted_context = context_results.get('gists', '')
            world_state = context_results.get('world_state', '')
            episodic_context = context_results.get('episodes', '')
            facts_context = context_results.get('facts', '')
            working_memory_context = context_results.get('working_memory', '')

        # Replace placeholders
        result = template.replace('{{original_prompt}}', original_prompt)
        result = result.replace('{{topic}}', str(topic))
        result = result.replace('{{confidence}}', str(confidence))
        result = result.replace('{{chat_history}}', formatted_context)
        result = result.replace('{{world_state}}', world_state)
        result = result.replace('{{episodic_memory}}', episodic_context)
        result = result.replace('{{act_history}}', act_history)
        result = result.replace('{{facts}}', facts_context)
        result = result.replace('{{working_memory}}', working_memory_context)

        # Get available skills
        available_skills = self._get_available_skills()
        result = result.replace('{{available_skills}}', available_skills)

        # Available tools (dynamic, from tool registry — filtered when relevant_tools provided)
        available_tools = self._get_available_tools(relevant_tools=relevant_tools)
        result = result.replace('{{available_tools}}', available_tools)

        # Available specialists for delegate skill
        available_specialists = self._get_available_specialists()
        result = result.replace('{{available_specialists}}', available_specialists)

        # Identity modulation (voice mapper)
        identity_modulation = self._get_identity_modulation()
        result = result.replace('{{identity_modulation}}', identity_modulation)

        # User traits (known facts about the user)
        user_traits = self._get_user_traits(original_prompt, topic)
        result = result.replace('{{user_traits}}', user_traits)

        # Client context (timezone, location, locale from frontend heartbeat)
        client_context = self._get_client_context()
        result = result.replace('{{client_context}}', client_context)

        return result

    def _get_identity_modulation(self) -> str:
        """Get identity modulation text from voice mapper."""
        try:
            from services.identity_service import IdentityService
            from services.voice_mapper_service import VoiceMapperService
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                identity = IdentityService(db_service)
                mapper = VoiceMapperService()

                vectors = identity.get_vectors()
                identity.check_coherence()
                modulation = mapper.generate_modulation(vectors)
                return modulation if modulation else "Engage naturally as a peer."
            finally:
                db_service.close_pool()
        except Exception as e:
            logging.warning(f"Identity modulation unavailable: {e}")
            return "Engage naturally as a peer."

    def _get_user_traits(self, prompt: str, topic: str) -> str:
        """
        Get user traits formatted for prompt injection.

        Retrieves core traits (always) and contextually relevant traits
        (by embedding similarity), capped at 6 total.

        Args:
            prompt: Current user message (for contextual retrieval)
            topic: Current topic

        Returns:
            str: Formatted user traits section or empty string
        """
        try:
            from services.user_trait_service import UserTraitService
            from services.database_service import DatabaseService, get_merged_db_config

            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                trait_service = UserTraitService(db_service)
                return trait_service.get_traits_for_prompt(prompt)
            finally:
                db_service.close_pool()
        except ImportError:
            return ""
        except Exception as e:
            logging.debug(f"User traits not available: {e}")
            return ""

    def _get_client_context(self) -> str:
        """
        Get client context (timezone, location, locale) from Redis heartbeat.

        Returns:
            str: Formatted client context or empty string if not available
        """
        try:
            from services.client_context_service import ClientContextService
            return ClientContextService().format_for_prompt()
        except Exception as e:
            logging.debug(f"Client context not available: {e}")
            return ""

    def _get_gist_context(self, topic: str) -> str:
        """
        Get gists from Redis or fallback to last message.

        Args:
            topic: Topic name

        Returns:
            str: Formatted gist context or fallback message
        """
        # Try to get gists from Redis
        gists = self.gist_storage.get_latest_gists(topic)

        if gists:
            # Filter out cold_start gists — they're internal metadata, not conversation context
            real_gists = [g for g in gists if g.get('type') != 'cold_start']
            if real_gists:
                lines = ["## Recent Conversation Gists"]
                for gist in real_gists:
                    content = gist['content']
                    gist_type = gist['type']
                    confidence = gist['confidence']
                    lines.append(f"- [{gist_type}] {content} (confidence: {confidence})")
                return "\n".join(lines)

        # Fallback to last message if no gists available
        last_message = self.gist_storage.get_last_message(topic)
        if last_message:
            return f"## Last Exchange\nUser: {last_message['prompt']}\nAssistant: {last_message['response']}"

        return "No previous conversation context available"

    def _get_available_skills(self) -> str:
        """
        Get available innate skills from the dispatcher for prompt injection.

        Returns innate skills list only. Dynamic tools are injected separately
        via {{available_tools}} placeholder to keep concerns separated.

        Returns:
            Formatted available skills string or empty string
        """
        try:
            from services.act_dispatcher_service import ActDispatcherService
            dispatcher = ActDispatcherService()
            innate = ["recall", "memorize", "introspect", "associate"]
            available = [s for s in innate if s in dispatcher.handlers]
            if available:
                return "Available skills: " + ", ".join(available)
            return ""
        except Exception as e:
            logging.debug(f"Skill registry not available: {e}")
            return ""

    def _get_available_specialists(self) -> str:
        """Specialists have been removed. Returns empty string."""
        return ""

    def _get_available_tools(self, relevant_tools: list = None) -> str:
        """
        Get tool summaries from the tool registry for ACT prompt injection.

        When relevant_tools is provided (from ToolRelevanceService), only injects
        those tools (max 5) instead of all registered tools.

        Returns:
            Formatted tool list or "(no tools loaded)"
        """
        max_injection = self.config.get('act_max_tool_injection', 5)

        try:
            from services.tool_registry_service import ToolRegistryService
            registry = ToolRegistryService()

            if relevant_tools:
                # Filter to only relevant tool names (exclude innate skills)
                tool_names = [
                    item['name'] for item in relevant_tools
                    if item.get('type') == 'tool' and item['name'] in registry.tools
                ][:max_injection]

                if tool_names:
                    lines = []
                    for name in tool_names:
                        manifest = registry.tools[name]['manifest']
                        desc = manifest.get('description', name)
                        params = manifest.get('parameters', {})
                        param_names = list(params.keys())
                        param_str = f" ({', '.join(param_names)})" if param_names else ""
                        lines.append(f"- {name}{param_str}: {desc}")
                    return "\n".join(lines)

            # Fallback: all tools
            summaries = registry.get_tool_prompt_summaries()
            if summaries:
                return summaries
            return "(no tools loaded)"
        except Exception as e:
            logging.debug(f"Tool registry not available for prompt: {e}")
            return "(no tools loaded)"

    def _get_facts_context(self, topic: str) -> str:
        """
        Get facts from Redis formatted for prompt injection.

        Args:
            topic: Topic name

        Returns:
            Formatted facts context or empty string
        """
        try:
            from services.fact_store_service import FactStoreService
            fact_store = FactStoreService()
            return fact_store.get_facts_formatted(topic)
        except Exception as e:
            logging.warning(f"Fact store not available: {e}")
            return ""

    def _get_working_memory_context(self, topic: str, thread_id: str = None) -> str:
        """
        Get working memory formatted for prompt injection.

        Args:
            topic: Topic name (legacy fallback key)
            thread_id: Thread ID (preferred key, overrides topic)

        Returns:
            Formatted working memory context or empty string
        """
        try:
            from services.working_memory_service import WorkingMemoryService
            max_turns = self.config.get('max_working_memory_turns', 10)
            working_memory = WorkingMemoryService(max_turns=max_turns)
            identifier = thread_id if thread_id else topic
            return working_memory.get_formatted_context(identifier)
        except Exception as e:
            logging.warning(f"Working memory not available: {e}")
            return ""

    def _extract_semantic_from_history(self, act_history: str) -> List[Dict]:
        """
        Parse semantic_query results from act_history string.

        Returns list of {"name": "...", "definition": "..."} dicts.
        """
        import re

        concepts = []
        # Match both old format "(strength: 0.XX)" and innate skills format "(confidence=X, strength=Y)"
        pattern = r'-\s+([^:]+):\s+([^(]+)\s+\((?:strength[:=]|confidence=)'
        matches = re.findall(pattern, act_history)

        for name, definition in matches:
            concepts.append({
                'name': name.strip(),
                'definition': definition.strip()
            })

        return concepts

    def _get_episodic_context(self, prompt: str, topic: str, act_history: str = "") -> str:
        """
        Retrieve relevant episodes and format as context.

        Args:
            prompt: User's current prompt
            topic: Current topic
            act_history: ACT history to extract semantic concepts from

        Returns:
            str: Formatted episodic memory context or empty string
        """
        if not self.episodic_retrieval:
            return ""

        try:
            # Extract semantic concepts from act_history if present
            semantic_concepts = self._extract_semantic_from_history(act_history)

            # Retrieve episodes with semantic boost
            episodes = self.episodic_retrieval.retrieve_episodes(
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
            logging.error(f"Failed to retrieve episodic memories: {e}")
            return ""

