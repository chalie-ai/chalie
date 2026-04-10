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
    mode-router / context-relevance singletons die with their owning
    services. Do not add new callers.
"""

import json
import logging

from services import ConfigService, OrchestratorService
from services.mode_router_service import ModeRouterService
from services.context_relevance_service import ContextRelevanceService

logger = logging.getLogger(__name__)

# ── Module-level singleton slots ──────────────────────────────────────────────

_context_relevance_service = None
_orchestrator = None
_mode_router = None


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


def get_mode_router():
    """Get or create global mode router instance."""
    global _mode_router
    if _mode_router is None:
        import os
        # Prefer generated config (from stability regulator) over base config
        generated_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "generated", "mode_router_config.json"
        )
        if os.path.exists(generated_path):
            try:
                with open(generated_path, 'r') as f:
                    router_config = json.load(f)
                logging.info("[DIGEST] Loaded generated mode router config")
            except Exception as e:
                logger.debug(f"[DIGEST] Failed to load generated mode router config, using default: {e}")
                router_config = ConfigService.get_agent_config("mode-router")
        else:
            router_config = ConfigService.get_agent_config("mode-router")
        _mode_router = ModeRouterService(router_config)
    return _mode_router


def load_configs():
    """Load frontal cortex mode-specific prompts and configurations.

    The only live cortex prompt is ``UNIFIED`` — the legacy ``ACT`` mode was
    folded into the unified path (see ``project_act_synthesis_removal``) and
    its prompt file has been deleted. ``prompt_map`` keeps its dict shape so
    downstream callers (``digest_worker.py``, ``system_prompt_assembly_service``)
    keep working without conditional key handling.
    """
    identity_prompt = ConfigService.get_agent_prompt("identity-core")
    cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    # Identity + mode prompt (values + voice + behavioral contract)
    unified_prompt = identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-unified")

    return {
        'cortex': {
            'config': cortex_config,
            'prompt_map': {
                'UNIFIED': unified_prompt,
            }
        },
    }
