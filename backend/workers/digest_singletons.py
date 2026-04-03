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
"""

import json
import logging

from services import ConfigService, OrchestratorService, SessionService
from services.mode_router_service import ModeRouterService
from services.thread_conversation_service import ThreadConversationService
from services.context_relevance_service import ContextRelevanceService
from services.context_assembly_service import ContextAssemblyService

logger = logging.getLogger(__name__)

# ── Module-level singleton slots ──────────────────────────────────────────────

_context_relevance_service = None
_context_assembly_service = None
_orchestrator = None
_thread_conv_service = None
_session_service = None
_mode_router = None


# ── Lazy getters ──────────────────────────────────────────────────────────────

def get_context_relevance_service():
    """Get or create global ContextRelevanceService instance."""
    global _context_relevance_service
    if _context_relevance_service is None:
        _context_relevance_service = ContextRelevanceService()
    return _context_relevance_service


def get_context_assembly_service():
    """Return the module-level singleton ``ContextAssemblyService`` instance.

    The service is created lazily on first access and reused for the lifetime
    of the worker process, avoiding repeated initialisation overhead across
    queue items.

    Returns:
        ContextAssemblyService: Shared context assembly service instance
            initialised with an empty configuration override dict.
    """
    global _context_assembly_service
    if _context_assembly_service is None:
        _context_assembly_service = ContextAssemblyService({})
    return _context_assembly_service


def get_orchestrator():
    """Get or create global OrchestratorService instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorService()
    return _orchestrator


def get_thread_conv_service() -> ThreadConversationService:
    """Get or create global ThreadConversationService instance."""
    global _thread_conv_service
    if _thread_conv_service is None:
        _thread_conv_service = ThreadConversationService()
    return _thread_conv_service


def get_session_service():
    """Get or create global session service instance."""
    global _session_service
    if _session_service is None:
        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        inactivity_timeout = episodic_config.get('inactivity_timeout', 600)
        _session_service = SessionService(inactivity_timeout=inactivity_timeout)
    return _session_service


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
    """Load frontal cortex mode-specific prompts and configurations."""
    soul_prompt = ConfigService.get_agent_prompt("soul")
    identity_prompt = ConfigService.get_agent_prompt("identity-core")
    cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    # Mode-specific prompts: soul -> identity -> mode prompt (instincts + context + contract)
    # Ordering: values first, then voice, then behavioral nudges closest to generation
    # ACT does NOT get identity -- reasoning stays pure
    act_prompt = ConfigService.get_agent_prompt("frontal-cortex-act")
    unified_prompt = soul_prompt + "\n\n" + identity_prompt + "\n\n" + ConfigService.get_agent_prompt("frontal-cortex-unified")

    return {
        'cortex': {
            'config': cortex_config,
            'prompt_map': {
                'ACT': act_prompt,
                'UNIFIED': unified_prompt,
            }
        },
    }
