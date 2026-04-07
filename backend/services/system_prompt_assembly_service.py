# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
System Prompt Assembly Service — builds the stable, cacheable system prompt.

Contains identity, skills, constraints, directives, and other context that
rarely changes between turns. Designed for provider-side prompt caching.
"""

import logging

from services.prompt_assembly_contract import PromptAssemblyContract

logger = logging.getLogger(__name__)


class SystemPromptAssemblyService(PromptAssemblyContract):
    """Builds the stable system prompt from a template + slow-changing context.

    Usage:
        svc = SystemPromptAssemblyService()
        svc.build(type='unified')
        system_prompt = svc.to_provider()
    """

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._result = ''

    def build(self, type: str = 'unified', thread_id: str = None,
              classification: dict = None, original_prompt: str = '',
              **kwargs) -> 'SystemPromptAssemblyService':
        """Build system prompt for the given type. Returns self for chaining."""
        builders = {
            'unified': self._build_unified,
        }
        builder = builders.get(type)
        if not builder:
            raise ValueError(f"Unknown system prompt type: {type}")
        self._result = builder(thread_id=thread_id, classification=classification,
                               original_prompt=original_prompt, **kwargs)
        return self

    def to_provider(self) -> str:
        return self._result

    def _build_unified(self, thread_id=None, classification=None,
                       original_prompt='', **kwargs):
        """Identity + frontal-cortex-unified + placeholder injection."""
        from workers.digest_singletons import load_configs
        configs = load_configs()
        template = configs['cortex']['prompt_map']['UNIFIED']

        # Inject identity_modulation
        identity_modulation = self._get_identity_modulation()
        template = template.replace('{{identity_modulation}}', identity_modulation)

        # Inject adaptive_directives
        adaptive = self._get_adaptive_directives(
            original_prompt=original_prompt, thread_id=thread_id,
        )
        template = template.replace('{{adaptive_directives}}', adaptive)

        return template

    def _get_identity_modulation(self) -> str:
        """Retrieve identity modulation text from the voice mapper for prompt injection."""
        try:
            from services.identity_service import IdentityService
            from services.voice_mapper_service import VoiceMapperService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            identity = IdentityService(db_service)
            mapper = VoiceMapperService()

            vectors = identity.get_vectors()
            identity.check_coherence()
            modulation = mapper.generate_modulation(vectors)
            return modulation if modulation else "Engage naturally as a peer."
        except Exception as e:
            logger.warning(f"[SYSTEM PROMPT] Identity modulation unavailable: {e}")
            return "Engage naturally as a peer."

    def _get_adaptive_directives(self, original_prompt: str = '', thread_id=None) -> str:
        """Get adaptive response directives based on detected user interaction style."""
        try:
            from services.adaptive_layer_service import AdaptiveLayerService
            from services.database_service import get_shared_db_service
            from services.working_memory_service import WorkingMemoryService

            db_service = get_shared_db_service()
            service = AdaptiveLayerService(db_service)

            # Get raw working memory turns for cognitive load estimation
            working_memory_turns = []
            if thread_id:
                try:
                    wm_service = WorkingMemoryService()
                    working_memory_turns = wm_service.get_recent_turns(thread_id) or []
                except Exception:
                    pass

            # Build current signals from prompt length
            current_signals = {
                'prompt_token_count': len(original_prompt.split()) if original_prompt else 0,
            }
            if thread_id:
                try:
                    from services.memory_client import MemoryClientService
                    import json as _json
                    _store = MemoryClientService.create_connection()
                    _snapshot_raw = _store.get(f"adaptive_signals:{thread_id}")
                    if _snapshot_raw:
                        _snapshot = _json.loads(_snapshot_raw)
                        current_signals.update(_snapshot)
                except Exception:
                    pass

            return service.generate_directives(
                thread_id=thread_id,
                working_memory_turns=working_memory_turns,
                current_signals=current_signals,
                current_message=original_prompt,
            ) or ''
        except Exception as e:
            logger.warning(f"[SYSTEM PROMPT] Adaptive directives unavailable: {e}")
            return ''
