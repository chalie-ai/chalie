# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Digest Singletons — Lazy singleton accessors used by digest_worker and friends.

Extracted from digest_worker.py to reduce file size. All getters preserve
their original semantics: create-on-first-access, reuse for process lifetime.

.. deprecated::
    This module is scheduled for removal alongside ``digest_worker`` itself.
    The new message-processing model (see
    ``/Volumes/llm/chalie-plans/message-processing.md``) does not use
    cross-channel singletons: ``load_configs()`` logic moves into the
    relevant ``SystemMessagePrompt`` subclass, and the orchestrator /
    context-relevance singletons die with their owning services.
    Do not add new callers.
"""

import logging

from services import ConfigService, OrchestratorService
from services.context_relevance_service import ContextRelevanceService

logger = logging.getLogger(__name__)

# ── Module-level singleton slots ──────────────────────────────────────────────

_context_relevance_service = None
_orchestrator = None


# ── Lazy getters ──────────────────────────────────────────────────────────────

def get_context_relevance_service():
    """Get or create global ContextRelevanceService instance."""
    global _context_relevance_service
    if _context_relevance_service is None:
        _context_relevance_service = ContextRelevanceService()
    return _context_relevance_service


def get_orchestrator():
    """Get or create global OrchestratorService instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator


def load_configs():
    """Load frontal cortex mode-specific prompts and configurations.

    DEPRECATED: identity-core.md deleted 2026-04-11 (folded into _UNIFIED_PROMPT in
    system_message_prompt.py). frontal-cortex-unified.md was already deleted before
    this change. All callers are in the deprecated legacy stack (digest_worker x2,
    SystemPromptAssemblyService) and will be removed with the digest_worker rip.
    Do not add new callers.
    """
    cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    # DEPRECATED: identity-core.md and frontal-cortex-unified.md both deleted 2026-04-11.
    # Legacy callers (digest_worker, SystemPromptAssemblyService) will be removed
    # with the digest_worker rip. Returning empty UNIFIED prompt is correct for now.
    unified_prompt = ''

    return {
        'cortex': {
            'config': cortex_config,
            'prompt_map': {
                'UNIFIED': unified_prompt,
            }
        },
    }
