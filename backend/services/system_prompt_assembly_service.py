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


class SystemPromptAssemblyService:
    """Builds the stable system prompt from a template + slow-changing context.

    Migration target for all system-prompt concerns currently in
    PromptAssemblyService. Each method migrated here gets deleted from the
    old service immediately — no duplicates.
    """

    def __init__(self, config: dict):
        self.config = config
