# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Post-Exchange Hooks — Deterministic hooks that run during or after each exchange.

Extracted from digest_worker.py. These are lightweight, non-LLM functions that
detect patterns in user input (name statements, belief corrections, engagement
classification) and write signals to the knowledge store or MemoryStore.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Immediate Identity Promotion (IIP)
#
# Deterministic regex patterns for detecting explicit name statements.
# Written synchronously (before any LLM call) to MemoryStore + SQLite so the name
# is available within the same request cycle. Target: <5ms. No LLM, no embeddings.
# ─────────────────────────────────────────────────────────────────────────────

# Capture group: one or two tokens, each allowing Unicode letters, apostrophes, hyphens.
# [^\W\d_] = any Unicode letter (standard re module — no external packages).
# Accepts any case — casing is normalised on write.
_IIP_NAME_CAPTURE = (
    r"([^\W\d_](?:[^\W\d_]|['\-]){0,39}"
    r"(?:\s+[^\W\d_](?:[^\W\d_]|['\-]){0,39})?)"
)

_IIP_PATTERNS = [
    re.compile(r"\bcall me\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bmy name is\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bi go by\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\byou can call me\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\bi'?m known as\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
    re.compile(r"\brefer to me as\s+" + _IIP_NAME_CAPTURE, re.IGNORECASE),
]

_IIP_STOPWORDS = frozenset([
    'a', 'an', 'the', 'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she',
    'they', 'it', 'this', 'that', 'here', 'there', 'done', 'fine', 'good',
    'okay', 'ok', 'sure', 'yes', 'no', 'maybe', 'later', 'anything',
    'something', 'nothing', 'everything',
])


def _run_iip_hook(text: str, database_service) -> None:
    """
    Detect explicit name statements and write to MemoryStore + SQLite synchronously.

    Deterministic regex only — no LLM, no embedding. Target: <5ms. Never raises.
    Preserves user's mixed-case input (McDonald, O'Brien); only title-cases
    when input is all-lowercase.
    """
    try:
        matched_name = None
        for pattern in _IIP_PATTERNS:
            m = pattern.search(text)
            if m:
                candidate = m.group(1).strip()
                # Reject stopwords (case-insensitive) and single-char matches
                if candidate.lower() not in _IIP_STOPWORDS and len(candidate) >= 2:
                    matched_name = candidate.title() if candidate.islower() else candidate
                    break

        if not matched_name:
            return

        from services.identity_state_service import IdentityStateService
        IdentityStateService().set_field(
            'name', matched_name, confidence=0.95, provisional=False
        )

        from services.knowledge_service import KnowledgeService
        KnowledgeService(database_service).store(
            kind='trait', entity='user', key='name', value=matched_name,
            data={'category': 'core'},
            decay_class='permanent', confidence=0.95, source='iip_hook',
        )
        logging.info(f"[IIP] Promoted name='{matched_name}' → MemoryStore + SQLite")

    except Exception as e:
        logging.warning(f"[IIP] Hook failed (non-fatal): {e}")


# Belief correction patterns — detect explicit trait corrections/negations
_BELIEF_CORRECTION_PATTERNS = [
    # Direct negation: "I don't like X", "I'm not a Y"
    re.compile(r"\b(?:I\s+(?:don'?t|do\s+not|never)\s+(?:like|enjoy|want|eat|drink|use|have|prefer|need))\s+(.+)", re.IGNORECASE),
    re.compile(r"\b(?:I'?m\s+not\s+(?:a\s+)?|I\s+am\s+not\s+(?:a\s+)?)(.+)", re.IGNORECASE),

    # Explicit correction: "actually my X is Y", "my name is actually Y"
    re.compile(r"\b(?:actually,?\s+)?my\s+(\w+(?:\s+\w+)?)\s+is\s+(?:actually\s+)?(.+)", re.IGNORECASE),

    # Belief correction: "that's wrong about me", "you're wrong about"
    re.compile(r"\b(?:that'?s\s+(?:wrong|incorrect|not\s+(?:true|right|correct))\s+(?:about\s+me|about\s+that))", re.IGNORECASE),
    re.compile(r"\b(?:you(?:'re|\s+are)\s+wrong\s+about)", re.IGNORECASE),

    # Retraction: "I never said I liked X", "I didn't say"
    re.compile(r"\b(?:I\s+never\s+said|I\s+didn'?t\s+say|I\s+didn'?t\s+tell\s+you)", re.IGNORECASE),

    # Stop assuming: "stop assuming", "don't assume"
    re.compile(r"\b(?:(?:stop|don'?t)\s+(?:assuming|thinking)\s+(?:I|that\s+I))", re.IGNORECASE),
]


def _run_belief_correction_hook(text: str, thread_id: str = None):
    """
    Detect explicit belief corrections and update/delete traits.
    Runs synchronously in Phase A before LLM trait injection.
    Precision-first: better to miss a correction than delete the wrong trait.
    """
    if not any(p.search(text) for p in _BELIEF_CORRECTION_PATTERNS):
        return

    text_lower = text.lower()

    # GUARDRAIL 1: Require explicit self-reference before any mutation
    # Prevents "sushi is terrible" from deleting a food preference
    if not re.search(r"\b(i|me|my|about me)\b", text_lower):
        return

    try:
        from services.knowledge_service import KnowledgeService
        from services.database_service import get_shared_db_service
        ks = KnowledgeService(get_shared_db_service())

        traits = ks.get_by_kind('trait', entity='user', limit=100)
        if not traits:
            return

        for trait in traits:
            key = trait.get('key', '')
            value = trait.get('value', '')
            confidence = trait.get('confidence', 0)

            # GUARDRAIL 2: Skip low-confidence traits — don't churn noisy data
            if confidence < 0.4:
                continue

            # GUARDRAIL 3: Skip empty trait values — "" is substring of everything
            if not value or not value.strip():
                continue

            # Check if the user's message negates this specific trait value
            escaped_value = re.escape(value.lower())
            if value.lower() in text_lower:
                negation_near_value = re.search(
                    rf"\b(?:not|don'?t|never|no longer|isn'?t|aren'?t|wasn'?t|wrong)\b.{{0,30}}\b{escaped_value}\b|"
                    rf"\b{escaped_value}\b.{{0,30}}\b(?:is wrong|is incorrect|is not right|isn'?t right)\b",
                    text_lower
                )
                if negation_near_value:
                    ks.forget('user', key)
                    logging.info(f"[BELIEF CORRECTION] Deleted trait '{key}={value}' — user negated it")
                    continue

            # Check for "actually my X is Y" pattern (value replacement)
            # Cap capture at 3 words to avoid trailing clauses
            replacement_match = re.search(
                rf"(?:actually,?\s+)?my\s+{re.escape(key.replace('_', ' '))}\s+is\s+(?:actually\s+)?(.+?)(?:\.|,|!|\?|$)",
                text_lower
            )
            if replacement_match:
                raw_value = replacement_match.group(1).strip()
                # Cap at 3 words to avoid trailing clause capture
                new_value = " ".join(raw_value.split()[:3])
                if new_value and new_value.lower() != value.lower():
                    ks.update('user', key, value=new_value)
                    logging.info(f"[BELIEF CORRECTION] Corrected trait '{key}': '{value}' → '{new_value}'")

    except Exception as e:
        logging.warning(f"[BELIEF CORRECTION] Hook failed (non-fatal): {e}")


def _classify_engagement(text: str) -> str:
    """
    Classify user engagement with a proactive message.
    Deterministic, no LLM. Pattern-based classification.
    Returns: engaged|acknowledged|rejected|ignored
    """
    import re
    text_lower = text.strip().lower()

    if not text_lower:
        return 'ignored'

    # Acknowledgment patterns (short, non-substantive) — check first
    ack = re.compile(
        r'^(ok|okay|sure|thanks|cool|got it|noted|yep|yeah|alright|'
        r'fine|roger|k|ty|thx|ack)\s*[.!]*$', re.IGNORECASE
    )
    if ack.match(text_lower):
        return 'acknowledged'

    if len(text_lower) < 3:
        return 'ignored'

    # For substantive messages (>20 chars), check engagement BEFORE rejection
    # This handles "I don't think that's right but tell me more" correctly
    engagement_pattern = re.compile(
        r'\b(yes|please|tell me|show|how|what|why|when|do it|go ahead|'
        r'more|explain|help|interesting|continue|elaborate)\b', re.IGNORECASE
    )

    rejection_pattern = re.compile(
        r'\b(stop|don.t|no thanks|not interested|shut up|leave me|go away|'
        r'not now|quit|enough|annoying)\b', re.IGNORECASE
    )

    has_engagement = engagement_pattern.search(text_lower)
    has_rejection = rejection_pattern.search(text_lower)

    # For longer messages with both signals, engagement wins
    # (user is still engaging even if they disagree)
    if len(text_lower) > 20 and has_engagement and has_rejection:
        # Check if rejection is followed by a "but" clause with engagement
        if re.search(r'\b(but|however|though|although)\b', text_lower):
            return 'engaged'
        # For longer messages, engagement signal wins by default
        return 'engaged'

    # Pure rejection (short or unambiguous)
    if has_rejection:
        return 'rejected'

    # Engaged: longer response, question, or action words
    if len(text_lower) > 20 or '?' in text or has_engagement:
        return 'engaged'

    return 'acknowledged'


_FORK_RESPONSE_PATTERNS = {
    'prefers_concise': [r'\b(quick|short|brief|summary|tldr|just.{0,10}(main|key|quick))\b'],
    'prefers_depth': [r'\b(deep|deeper|detail|more|elaborate|explore|full|thorough)\b'],
    'enjoys_challenge': [r'\b(challenge|push back|harder|stress.test|poke holes|counterpoint|disagree)\b'],
}


def _detect_fork_response(text: str, thread_id: str):
    """
    Detect if the user's message is a response to a previously offered fork.

    If a fork was pending (adaptive_fork_pending:{thread_id} MemoryStore key exists),
    pattern-match the user's reply and store the corresponding micro-preference.
    """
    import re as _re
    try:
        from services.memory_client import MemoryClientService
        from services.database_service import get_shared_db_service
        from services.knowledge_service import KnowledgeService

        store = MemoryClientService.create_connection()
        fork_type = store.get(f"adaptive_fork_pending:{thread_id}")
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
                store.delete(f"adaptive_fork_pending:{thread_id}")
                break
    except Exception as e:
        logging.debug(f"[DIGEST] Fork response detection failed: {e}")


def _store_adaptive_signals(thread_id: str, text: str, signals: dict = None):
    """
    Store a minimal snapshot of current exchange signals to MemoryStore for use by
    AdaptiveLayerService (energy mirroring, cognitive load).

    Key: adaptive_signals:{thread_id}, TTL: 300s
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
            f"adaptive_signals:{thread_id}",
            300,
            _json.dumps(snapshot),
        )
    except Exception as e:
        logging.debug(f"[DIGEST] Adaptive signal storage failed: {e}")
