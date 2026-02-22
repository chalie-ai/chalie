"""
Intent Classifier Service — Fast, deterministic intent classification.

Runs BEFORE mode routing to produce structured intent metadata. No LLM call.
Uses NLP patterns + memory signals to classify the user's intent.

This layer feeds INTO the mode router as additional signals and drives
the fast-path/slow-path decision for ACT mode.
"""

import re
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ── Pattern banks ──────────────────────────────────────────────

INTERROGATIVE_PATTERNS = re.compile(
    r'\b(what|where|when|who|why|how|which|whose|whom)\b',
    re.IGNORECASE
)

COMMAND_PATTERNS = re.compile(
    r'\b(search\w*|find\w*|look\s*up|check\w*|research\w*|browse\w*|fetch\w*|google\w*|investigate\w*|'
    r'look\s+into|get\s+me|show\s+me|tell\s+me\s+about|explain|summarize|'
    r'analyze|compare|calculate|list|describe|translate|convert)\b',
    re.IGNORECASE
)

FEEDBACK_PATTERNS = re.compile(
    r'\b(thanks|thank\s+you|great|perfect|awesome|exactly|that\s+works|correct|'
    r'good|nice|helpful|got\s+it|understood|wrong|incorrect|no\s+that|doesn\'?t\s+work|'
    r'confused|not\s+what\s+i|try\s+again|not\s+helpful)\b',
    re.IGNORECASE
)

GREETING_PATTERNS = re.compile(
    r'^(hey|hi|hello|yo|sup|what\'?s\s*up|howdy|hiya|heya|greetings|'
    r'good\s*(morning|afternoon|evening))\b',
    re.IGNORECASE
)

CANCEL_PATTERNS = [
    re.compile(r'\bnever\s*mind\b', re.IGNORECASE),
    re.compile(r'\bignore\s+that\b', re.IGNORECASE),
    re.compile(r'\bstop\s+(searching|looking|checking)\b', re.IGNORECASE),
    re.compile(r'\bforget\s+(it|about\s+it|that)\b', re.IGNORECASE),
    re.compile(r'\bcancel\b', re.IGNORECASE),
    re.compile(r'\bdon\'?t\s+bother\b', re.IGNORECASE),
]

SELF_RESOLVED_PATTERNS = [
    re.compile(r'\b(found|figured|sorted|solved|got)\s+(it|that|this)\s*(out|now|myself)?\b', re.IGNORECASE),
    re.compile(r'\b(all\s+good|no\s+worries|no\s+need)\b', re.IGNORECASE),
    re.compile(r'\bnever\s*mind\s+.*(found|figured|got)\b', re.IGNORECASE),
]

COMPLEXITY_LONG_THRESHOLD = 30  # tokens
COMPLEXITY_MULTI_CLAUSE = re.compile(r'\b(and|also|plus|additionally|furthermore|then)\b', re.IGNORECASE)

# ── Casual / formal register ───────────────────────────────────

CASUAL_MARKERS = {'hey', 'u', 'pls', 'thx', 'lol', 'yo', 'sup', 'nah', 'gonna', 'wanna', 'kinda', 'sorta', 'idk', 'nope', 'yep', 'yup', 'bruh', 'dude', 'omg', 'btw'}
FORMAL_MARKERS = {'please', 'kindly', 'inquire', 'regarding', 'sincerely', 'respectfully', 'appreciate'}
FORMAL_PHRASES = [
    re.compile(r'\bcould you\b', re.IGNORECASE),
    re.compile(r'\bwould you\b', re.IGNORECASE),
    re.compile(r'\bi would like\b', re.IGNORECASE),
]

class IntentClassifierService:
    """
    Fast, deterministic intent classifier.

    Produces structured intent metadata that feeds into mode routing
    and drives the fast-path/slow-path decision.
    """

    def __init__(self):
        pass

    def classify(
        self,
        text: str,
        topic: str,
        context_warmth: float = 0.0,
        fact_count: int = 0,
        gist_count: int = 0,
        procedural_stats: Optional[Dict] = None,
        memory_confidence: float = 1.0,
        working_memory_turns: int = 0,
    ) -> Dict[str, Any]:
        """
        Classify user intent from prompt text + memory signals.

        Args:
            text: Raw user prompt
            topic: Current topic
            context_warmth: Pre-computed warmth (0-1)
            fact_count: Number of facts available
            gist_count: Number of gists available
            procedural_stats: Optional ranked skills from procedural memory
            memory_confidence: Pre-computed memory confidence score
            working_memory_turns: Number of turns in working memory

        Returns:
            Structured intent dict
        """
        start = time.time()

        tokens = text.split()
        token_count = len(tokens)

        # Intent type classification
        intent_type = self._classify_type(text, tokens)

        # Complexity estimation
        complexity = self._estimate_complexity(text, tokens)

        # Confidence
        confidence = self._calculate_confidence(
            intent_type, context_warmth, token_count
        )

        # Register detection
        register = self._detect_register(text)

        # Special intents
        is_cancel = self._is_cancel_intent(text)
        is_self_resolved = self._is_self_resolved(text)

        classification_time = time.time() - start

        result = {
            'intent_type': intent_type,
            'complexity': complexity,
            'memory_sufficient': True,
            'confidence': confidence,
            'register': register,
            'is_cancel': is_cancel,
            'is_self_resolved': is_self_resolved,
            'classification_time_ms': classification_time * 1000,
        }

        logger.info(
            f"[INTENT] {intent_type} | "
            f"complexity={complexity} | confidence={confidence:.2f} | "
            f"{classification_time*1000:.1f}ms"
        )

        return result

    def _classify_type(self, text: str, tokens: list) -> str:
        """Classify the intent type."""
        stripped = text.strip()

        if not stripped:
            return 'empty'
        if GREETING_PATTERNS.match(stripped):
            return 'greeting'
        if FEEDBACK_PATTERNS.search(text):
            return 'feedback'
        if any(p.search(text) for p in CANCEL_PATTERNS):
            return 'command'
        if COMMAND_PATTERNS.search(text) and not ('?' in text):
            return 'command'
        if '?' in text or INTERROGATIVE_PATTERNS.search(text):
            return 'question'
        if len(tokens) < 5 and not '?' in text:
            # Short statements are often continuations
            return 'continuation'
        return 'statement'

    def _estimate_complexity(self, text: str, tokens: list) -> str:
        """Estimate effort needed to answer."""
        token_count = len(tokens)

        if token_count > COMPLEXITY_LONG_THRESHOLD:
            return 'complex'
        if COMPLEXITY_MULTI_CLAUSE.search(text):
            return 'moderate'
        if token_count > 15:
            return 'moderate'
        return 'simple'

    def _calculate_confidence(
        self,
        intent_type: str,
        context_warmth: float,
        token_count: int,
    ) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.5

        # Clear type signals boost confidence
        if intent_type in ('greeting', 'feedback', 'empty'):
            confidence += 0.3
        if intent_type == 'question':
            confidence += 0.1

        # Context helps
        if context_warmth > 0.5:
            confidence += 0.1

        # Very short or very long prompts are harder to classify
        if token_count < 3:
            confidence -= 0.1
        if token_count > 50:
            confidence -= 0.1

        return max(0.1, min(1.0, confidence))

    def _detect_register(self, text: str) -> str:
        """Detect user's communication register."""
        words = text.lower().split()
        casual_count = sum(1 for w in words if w in CASUAL_MARKERS)
        formal_count = sum(1 for w in words if w in FORMAL_MARKERS)
        formal_count += sum(1 for p in FORMAL_PHRASES if p.search(text))

        if casual_count > formal_count:
            return 'casual'
        elif formal_count > casual_count:
            return 'formal'
        return 'neutral'

    def _is_cancel_intent(self, text: str) -> bool:
        """Check if user wants to cancel active work."""
        # Self-resolved takes priority over cancel
        if self._is_self_resolved(text):
            return False
        return any(p.search(text) for p in CANCEL_PATTERNS)

    def _is_self_resolved(self, text: str) -> bool:
        """Check if user solved it themselves."""
        return any(p.search(text) for p in SELF_RESOLVED_PATTERNS)
