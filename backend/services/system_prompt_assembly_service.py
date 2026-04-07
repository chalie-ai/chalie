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

Migration status: shell — methods migrate here from PromptAssemblyService
one at a time. Once all methods are here, PromptAssemblyService is deleted.
"""

import logging

from services.prompt_assembly_contract import PromptAssemblyContract

logger = logging.getLogger(__name__)


class SystemPromptAssemblyService(PromptAssemblyContract):
    """Builds the stable system prompt from a template + slow-changing context.

    Migration target for all system-prompt concerns currently in
    PromptAssemblyService. Each method migrated here gets deleted from the
    old service immediately — no duplicates.

    Usage:
        svc = SystemPromptAssemblyService(config)
        svc.build(template, classification, ...)
        system_prompt = svc.to_provider()
    """

    def __init__(self, config: dict):
        self.config = config
        self._result = ''

    def build(self, template: str, classification: dict, thread_id: str = None,
              selected_skills: list = None, returning_from_silence: bool = False,
              inclusion_map: dict = None) -> 'SystemPromptAssemblyService':
        """Assemble the system prompt by injecting stable placeholders.

        Returns self for chaining: `svc.build(...).to_provider()`
        """
        # Delegate to legacy PromptAssemblyService for now.
        # As methods are migrated here, the delegation shrinks.
        from services.prompt_assembly_service import PromptAssemblyService
        legacy = PromptAssemblyService(self.config)
        self._result = legacy.build_system_prompt(
            system_prompt_template=template,
            original_prompt='',
            classification=classification,
            chat_history=[],
            selected_skills=selected_skills,
            thread_id=thread_id,
            returning_from_silence=returning_from_silence,
            inclusion_map=inclusion_map,
        )
        return self

    def to_provider(self) -> str:
        return self._result
