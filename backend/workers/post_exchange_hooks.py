# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Post-Exchange Hooks — Deterministic hooks that run during or after each exchange.

These are lightweight, non-LLM functions that detect patterns in user input
and write signals to the knowledge store or MemoryStore.
"""

import logging

logger = logging.getLogger(__name__)


_FORK_RESPONSE_PATTERNS = {
    'prefers_concise': [r'\b(quick|short|brief|summary|tldr|just.{0,10}(main|key|quick))\b'],
    'prefers_depth': [r'\b(deep|deeper|detail|more|elaborate|explore|full|thorough)\b'],
    'enjoys_challenge': [r'\b(challenge|push back|harder|stress.test|poke holes|counterpoint|disagree)\b'],
}


def _detect_fork_response(text: str):
    """
    Detect if the user's message is a response to a previously offered fork.

    If a fork was pending (adaptive_fork_pending MemoryStore key exists),
    pattern-match the user's reply and store the corresponding micro-preference.
    """
    import re as _re
    try:
        from services.memory_client import MemoryClientService
        from services.database_service import get_shared_db_service
        from services.knowledge_service import KnowledgeService

        store = MemoryClientService.create_connection()
        fork_type = store.get('adaptive_fork_pending')
        if not fork_type:
            return

        # Match user response to a micro-preference
        text_lower = text.lower()
        for pref_key, patterns in _FORK_RESPONSE_PATTERNS.items():
            if any(_re.search(p, text_lower) for p in patterns):
                db_service = get_shared_db_service()
                ks = KnowledgeService(db_service)
                ks.store(
                    kind='trait', entity='user', key=pref_key, value='true',
                    data={'category': 'preference'},
                    decay_class='standard', confidence=0.75,
                    source='fork_response',
                )
                logging.info(f"[DIGEST] Fork response detected → stored micro-preference: {pref_key}")
                # Clear the pending key
                store.delete('adaptive_fork_pending')
                break
    except Exception as e:
        logging.debug(f"[DIGEST] Fork response detection failed: {e}")


def _store_adaptive_signals(text: str, signals: dict = None):
    """
    Store a minimal snapshot of current exchange signals to MemoryStore for use by
    AdaptiveLayerService (energy mirroring, cognitive load).

    Key: adaptive_signals, TTL: 300s
    """
    import json as _json
    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        snapshot = {
            'prompt_token_count': len(text.split()) if text else 0,
            'explicit_feedback': signals.get('explicit_feedback') if signals else None,
        }
        store.setex(
            'adaptive_signals',
            300,
            _json.dumps(snapshot),
        )
    except Exception as e:
        logging.debug(f"[DIGEST] Adaptive signal storage failed: {e}")
