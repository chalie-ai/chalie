"""Deterministic communication style measurement from message text.

No LLM required — pure regex/heuristic analysis.
Measures 5 dimensions on a 1-10 scale.

Emits: (none)
Consumes: (none)
Trigger: request-driven
Fail mode: fail-open (returns midpoint defaults)
"""

import re
import logging

logger = logging.getLogger(__name__)

# Hedging patterns for directness/certainty
HEDGE_PATTERNS = re.compile(
    r'\b(maybe|perhaps|possibly|might|could be|i think|i guess|i suppose|'
    r'not sure|kind of|sort of|probably|arguably|it seems|i feel like|'
    r'i wonder|would say|tend to)\b',
    re.IGNORECASE
)

# Formal patterns
FORMAL_PATTERNS = re.compile(
    r'\b(furthermore|nevertheless|consequently|accordingly|'
    r'regarding|pertaining|pursuant|hereby|therefore|thus|hence|'
    r'subsequently|moreover|notwithstanding)\b',
    re.IGNORECASE
)

# Casual/informal patterns
CASUAL_PATTERNS = re.compile(
    r"(lol|lmao|haha|gonna|wanna|gotta|kinda|y'all|nah|yep|nope|"
    r"btw|imo|imho|tbh|idk|omg|wtf|bruh|dude|bro|"
    r"ain't|shouldn't've|couldn't've)",
    re.IGNORECASE
)

# Contraction pattern
CONTRACTION_PATTERN = re.compile(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", re.IGNORECASE)

# Emoji/emoticon detection
EMOJI_PATTERN = re.compile(
    r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]|'
    r'[:;][-(]?[)D(P/\\|]'  # text emoticons
)


class StyleMetricsService:
    """Measures communication style deterministically from message text.

    Returns 5 dimensions on a 1-10 scale:
    - verbosity: message length/word count
    - directness: absence of hedging language
    - formality: formal markers vs casual markers and contractions
    - certainty: confidence of declarations vs hedging/questions
    - pacing: message length as proxy for deliberateness
    """

    def measure(self, message: str) -> dict:
        """Measure communication style from a single message.

        Args:
            message: Raw user message text.

        Returns:
            Dict with 5 dimensions (1-10 scale), or midpoint defaults on error.
        """
        try:
            if not message or not message.strip():
                return self._defaults()

            return {
                'verbosity': self._measure_verbosity(message),
                'directness': self._measure_directness(message),
                'formality': self._measure_formality(message),
                'certainty': self._measure_certainty(message),
                'pacing': self._measure_pacing(message),
            }
        except Exception as e:
            logger.debug(f"[STYLE_METRICS] Measurement failed: {e}")
            return self._defaults()

    def _defaults(self) -> dict:
        return {'verbosity': 5, 'directness': 5, 'formality': 5, 'certainty': 5, 'pacing': 5}

    def _measure_verbosity(self, message: str) -> int:
        """Word count → 1-10 scale.

        1-3 words = 1, 4-8 = 2, 9-15 = 3, 16-25 = 4, 26-40 = 5,
        41-60 = 6, 61-90 = 7, 91-130 = 8, 131-180 = 9, 181+ = 10
        """
        words = len(message.split())
        thresholds = [3, 8, 15, 25, 40, 60, 90, 130, 180]
        for i, threshold in enumerate(thresholds):
            if words <= threshold:
                return i + 1
        return 10

    def _measure_directness(self, message: str) -> int:
        """Absence of hedging → higher directness.

        Counts hedge phrases relative to word count.
        """
        words = max(len(message.split()), 1)
        hedges = len(HEDGE_PATTERNS.findall(message))
        hedge_ratio = hedges / words

        if hedge_ratio == 0:
            return 9
        elif hedge_ratio < 0.02:
            return 8
        elif hedge_ratio < 0.04:
            return 7
        elif hedge_ratio < 0.06:
            return 6
        elif hedge_ratio < 0.08:
            return 5
        elif hedge_ratio < 0.10:
            return 4
        elif hedge_ratio < 0.13:
            return 3
        elif hedge_ratio < 0.16:
            return 2
        else:
            return 1

    def _measure_formality(self, message: str) -> int:
        """Formal markers vs casual markers + contractions + emoji.

        Composite score from formal words, casual words, contractions, emoji.
        """
        words = max(len(message.split()), 1)

        formal_count = len(FORMAL_PATTERNS.findall(message))
        casual_count = len(CASUAL_PATTERNS.findall(message))
        contraction_count = len(CONTRACTION_PATTERN.findall(message))
        emoji_count = len(EMOJI_PATTERN.findall(message))

        formal_signal = formal_count / words
        casual_signal = (casual_count + contraction_count * 0.3 + emoji_count * 0.5) / words

        # Net formality: -1 (very casual) to +1 (very formal)
        net = min(1.0, max(-1.0, (formal_signal - casual_signal) * 10))

        return max(1, min(10, round(5 + net * 5)))

    def _measure_certainty(self, message: str) -> int:
        """Confidence of declarations vs hedging/questions.

        Combines hedge frequency with question ratio.
        """
        words = max(len(message.split()), 1)
        sentences = max(len(re.split(r'[.!?]+', message.strip())), 1)

        hedges = len(HEDGE_PATTERNS.findall(message))
        questions = message.count('?')

        hedge_ratio = hedges / words
        question_ratio = questions / sentences

        uncertainty = hedge_ratio * 5 + question_ratio * 3

        if uncertainty < 0.1:
            return 9
        elif uncertainty < 0.3:
            return 8
        elif uncertainty < 0.6:
            return 7
        elif uncertainty < 1.0:
            return 6
        elif uncertainty < 1.5:
            return 5
        elif uncertainty < 2.0:
            return 4
        elif uncertainty < 3.0:
            return 3
        elif uncertainty < 4.0:
            return 2
        else:
            return 1

    def _measure_pacing(self, message: str) -> int:
        """Message length as proxy for deliberateness.

        Short rapid messages = low pacing, long deliberate messages = high pacing.
        Similar to verbosity but with sentence structure consideration.
        """
        chars = len(message.strip())
        sentences = max(len(re.split(r'[.!?]+', message.strip())), 1)
        avg_sentence_len = chars / sentences

        if chars < 20:
            return 1
        elif chars < 50:
            return 2
        elif chars < 100:
            return 3
        elif avg_sentence_len < 40:
            return 4
        elif avg_sentence_len < 60:
            return 5
        elif avg_sentence_len < 80:
            return 6
        elif avg_sentence_len < 100:
            return 7
        elif avg_sentence_len < 130:
            return 8
        elif avg_sentence_len < 170:
            return 9
        else:
            return 10
