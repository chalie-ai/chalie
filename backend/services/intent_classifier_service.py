"""
Intent Classifier Service — Fast, deterministic intent classification.

Runs BEFORE mode routing to produce structured intent metadata. No LLM call.
Uses NLP patterns + memory signals to classify the user's intent.

This layer feeds INTO the mode router as additional signals and drives
the fast-path/slow-path decision for ACT mode.
"""

import re
import random
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

# ── Template acknowledgments ───────────────────────────────────

TEMPLATE_POOL = {
    ('question', 'complex', 'casual'): [
        "Sure, on it...",
        "Let me dig into that...",
        "Checking on that now...",
    ],
    ('question', 'complex', 'formal'): [
        "Let me look into that for you...",
        "I'll research that — one moment...",
        "Checking on that, one moment...",
    ],
    ('question', 'complex', 'neutral'): [
        "Checking on that...",
        "Looking into that...",
        "Let me dig into that for you...",
        "Good question — looking into it...",
    ],
    ('question', 'moderate', 'casual'): [
        "Let me check on that...",
        "On it...",
    ],
    ('question', 'moderate', 'formal'): [
        "Let me check on that for you...",
        "Looking into that...",
    ],
    ('question', 'moderate', 'neutral'): [
        "Let me check on that...",
        "Looking into that...",
    ],
    ('command', 'complex', 'casual'): [
        "On it — this might take a moment...",
        "Got it, working on that...",
    ],
    ('command', 'complex', 'formal'): [
        "Working on that for you...",
        "On it — this might take a moment...",
    ],
    ('command', 'complex', 'neutral'): [
        "On it — this might take a moment...",
        "Working on that...",
        "Got it, on it...",
    ],
    ('command', 'moderate', 'casual'): [
        "On it...",
        "Sure, looking into it...",
    ],
    ('command', 'moderate', 'formal'): [
        "Working on that for you...",
        "Let me handle that...",
    ],
    ('command', 'moderate', 'neutral'): [
        "Working on that...",
        "On it...",
    ],
}

DEFAULT_TEMPLATES = {
    'casual': ["On it...", "Sure, checking...", "Let me look into that..."],
    'formal': ["Let me look into that for you...", "Working on that...", "One moment..."],
    'neutral': ["Let me look into that...", "Checking on that...", "One moment..."],
}

# Used when the ACT loop will mainly use innate skills (introspect/recall/associate)
# rather than external tools — tone should be reflective, not search-oriented.
REFLECTIVE_TEMPLATES = {
    'casual': [
        "Hmm, let me think about that...",
        "Let me reflect on that...",
        "Good question — thinking...",
    ],
    'formal': [
        "Let me give that some thought...",
        "I'll reflect on that for a moment...",
        "Let me consider that...",
    ],
    'neutral': [
        "Let me think about that...",
        "Hmm, reflecting on that...",
        "Let me consider this...",
    ],
}


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
        tool_relevance: Optional[Dict] = None,
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
            tool_relevance: Optional dict from ToolRelevanceService.score_relevance()

        Returns:
            Structured intent dict
        """
        start = time.time()

        tokens = text.split()
        token_count = len(tokens)

        # Intent type classification
        intent_type = self._classify_type(text, tokens)

        # Tool hints from embedding-based relevance (replaces regex TOOL_HINT_PATTERNS)
        tool_hints = []
        tool_relevance_score = 0.0
        if tool_relevance:
            tool_relevance_score = tool_relevance.get('max_relevance_score', 0.0)
            tool_hints = [
                item['name'] for item in tool_relevance.get('relevant_tools', [])
                if item.get('score', 0) >= 0.25
            ]

        # Complexity estimation
        complexity = self._estimate_complexity(text, tokens, tool_hints)

        # Memory sufficiency check
        memory_sufficient = self._check_memory_sufficient(
            intent_type, context_warmth, fact_count, gist_count,
            tool_hints, procedural_stats, tool_relevance_score
        )

        # Needs tools?
        needs_tools = self._needs_tools(
            intent_type, tool_hints, memory_sufficient, complexity, context_warmth,
            tool_relevance_score
        )

        # Confidence
        confidence = self._calculate_confidence(
            intent_type, tool_hints, context_warmth, token_count
        )

        # Register detection
        register = self._detect_register(text)

        # Special intents
        is_cancel = self._is_cancel_intent(text)
        is_self_resolved = self._is_self_resolved(text)

        classification_time = time.time() - start

        result = {
            'intent_type': intent_type,
            'needs_tools': needs_tools,
            'tool_hints': tool_hints,
            'complexity': complexity,
            'memory_sufficient': memory_sufficient,
            'confidence': confidence,
            'register': register,
            'is_cancel': is_cancel,
            'is_self_resolved': is_self_resolved,
            'classification_time_ms': classification_time * 1000,
        }

        logger.info(
            f"[INTENT] {intent_type} | needs_tools={needs_tools} | "
            f"complexity={complexity} | confidence={confidence:.2f} | "
            f"tools={tool_hints} | {classification_time*1000:.1f}ms"
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

    def _estimate_complexity(self, text: str, tokens: list, tool_hints: list) -> str:
        """Estimate effort needed to answer."""
        token_count = len(tokens)

        if token_count > COMPLEXITY_LONG_THRESHOLD:
            return 'complex'
        if len(tool_hints) > 1:
            return 'complex'
        if COMPLEXITY_MULTI_CLAUSE.search(text):
            return 'moderate'
        if token_count > 15 or tool_hints:
            return 'moderate'
        return 'simple'

    def _check_memory_sufficient(
        self,
        intent_type: str,
        context_warmth: float,
        fact_count: int,
        gist_count: int,
        tool_hints: list,
        procedural_stats: Optional[Dict],
        tool_relevance_score: float = 0.0,
    ) -> bool:
        """Check if the question can be answered from memory alone."""
        # Greetings, feedback, continuations — always memory-sufficient
        if intent_type in ('greeting', 'feedback', 'continuation', 'empty'):
            return True

        # Strong tool relevance overrides memory
        if tool_relevance_score > 0.55:
            return False

        # Explicit delegation request overrides memory
        if 'delegate' in tool_hints:
            return False

        # High warmth + available facts → memory sufficient
        if context_warmth > 0.6 and fact_count >= 2:
            return True

        # Procedural memory: if recall has high success rate for this topic
        if procedural_stats:
            recall_stats = procedural_stats.get('recall', {})
            if recall_stats.get('success_rate', 0) > 0.7 and recall_stats.get('attempts', 0) > 5:
                return True

        # Moderate warmth with gists
        if context_warmth > 0.4 and gist_count >= 3:
            return True

        return False

    def _needs_tools(
        self,
        intent_type: str,
        tool_hints: list,
        memory_sufficient: bool,
        complexity: str,
        context_warmth: float = 0.0,
        tool_relevance_score: float = 0.0,
    ) -> bool:
        """Determine if this prompt likely needs tool use.

        Embedding relevance is the sole authority — regex pattern matching
        no longer gates tool dispatch.
        """
        # Embedding relevance is the sole authority
        if tool_relevance_score > 0.35:
            return True

        # Social/empty intents never need tools
        if intent_type in ('greeting', 'feedback', 'empty'):
            return False

        return False

    def _calculate_confidence(
        self,
        intent_type: str,
        tool_hints: list,
        context_warmth: float,
        token_count: int,
    ) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.5

        # Clear type signals boost confidence
        if intent_type in ('greeting', 'feedback', 'empty'):
            confidence += 0.3
        if intent_type == 'command' and tool_hints:
            confidence += 0.25
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

        # Explicit tool hints increase confidence
        if tool_hints:
            confidence += 0.1

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

    def extract_topic_phrase(self, text: str) -> str:
        """
        Extract a short key phrase from the user's prompt for template personalization.

        Uses simple NLP extraction — no LLM call.
        """
        # For questions: noun phrase after interrogative word
        interrog_match = re.search(
            r'\b(?:what|where|when|who|why|how|which)\s+(?:is|are|was|were|do|does|did|can|could|would|should|about)?\s*(.+?)(?:\?|$)',
            text, re.IGNORECASE
        )
        if interrog_match:
            phrase = interrog_match.group(1).strip()
            # Trim to reasonable length
            words = phrase.split()
            if len(words) > 6:
                phrase = ' '.join(words[:6])
            if phrase:
                return phrase

        # For commands: complement after the verb
        cmd_match = re.search(
            r'\b(?:search|find|look\s*up|check|research|investigate|get|show|tell\s+me\s+about)\s+(?:for|about|on|into)?\s*(.+?)(?:\.|$)',
            text, re.IGNORECASE
        )
        if cmd_match:
            phrase = cmd_match.group(1).strip()
            words = phrase.split()
            if len(words) > 6:
                phrase = ' '.join(words[:6])
            if phrase:
                return phrase

        # For action requests: object after action verb
        action_match = re.search(
            r'\b(?:remind|schedule|set|cancel|create|delete|remove)\s+'
            r'(?:me\s+)?(?:to\s+|a\s+|an\s+|the\s+)?(.+?)(?:\.|!|$)',
            text, re.IGNORECASE
        )
        if action_match:
            phrase = action_match.group(1).strip()
            words = phrase.split()
            return ' '.join(words[:6]) if len(words) > 6 else phrase

        # Fallback: trim leading function words, take contiguous chunk
        words = text.split()
        if len(words) <= 5:
            return text.strip().rstrip('?.!')
        skip_prefixes = {'can', 'could', 'would', 'should', 'please', 'hey',
                         'hi', 'i', "i'd", "i'll", "i'm"}
        start = 0
        for i, w in enumerate(words):
            if w.lower() in skip_prefixes:
                start = i + 1
            else:
                break
        return ' '.join(words[start:start + 5]).rstrip('?.!')

    def select_template(
        self,
        intent_type: str,
        complexity: str,
        register: str,
        topic_phrase: str,
        redis_conn=None,
        user_id: str = 'default',
        is_reflective: bool = False,
    ) -> str:
        """
        Select a personalized acknowledgment template.

        Avoids repeating recently used templates (tracked in Redis).

        Args:
            intent_type: 'question', 'command', etc.
            complexity: 'simple', 'moderate', 'complex'
            register: 'casual', 'formal', 'neutral'
            topic_phrase: Extracted topic phrase for personalization
            redis_conn: Optional Redis connection for recency tracking
            user_id: User identifier for recency tracking
            is_reflective: When True, use thinking/reflective language instead of
                search/tool language (appropriate when ACT loop will use innate skills)

        Returns:
            Formatted acknowledgment string
        """
        # Reflective mode: innate skills (introspect/recall/associate) dominate
        if is_reflective:
            pool = REFLECTIVE_TEMPLATES.get(register, REFLECTIVE_TEMPLATES['neutral'])
            # Skip the TEMPLATE_POOL lookup entirely
            available = pool
            template = random.choice(available)
            return template.format(topic_phrase=topic_phrase or 'that', action_phrase=topic_phrase or 'that')

        # Look up template pool
        pool = TEMPLATE_POOL.get((intent_type, complexity, register))
        if not pool:
            pool = TEMPLATE_POOL.get((intent_type, complexity, 'neutral'))
        if not pool:
            pool = DEFAULT_TEMPLATES.get(register, DEFAULT_TEMPLATES['neutral'])

        # Filter recently used (if Redis available)
        recent_key = f"ack_history:{user_id}"
        recent = []
        if redis_conn:
            try:
                recent = redis_conn.lrange(recent_key, 0, 2)
                if recent:
                    recent = [r.decode('utf-8') if isinstance(r, bytes) else r for r in recent]
            except Exception:
                pass

        available = [t for t in pool if t not in recent]
        if not available:
            available = pool

        template = random.choice(available)

        # Track usage in Redis
        if redis_conn:
            try:
                redis_conn.lpush(recent_key, template)
                redis_conn.ltrim(recent_key, 0, 2)
                redis_conn.expire(recent_key, 3600)
            except Exception:
                pass

        # Personalize with topic phrase
        action_phrase = topic_phrase or 'that'
        topic_phrase = topic_phrase or 'that'

        return template.format(
            topic_phrase=topic_phrase,
            action_phrase=action_phrase,
        )
