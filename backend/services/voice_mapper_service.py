"""
Voice Mapper Service - Translates identity vector state into tone instructions.

Pure rule-based mapping, no LLM call. Vectors above HIGH threshold get explicit
tone direction; vectors below LOW threshold get the opposite direction;
mid-range vectors produce no instruction (natural default).

Top-N cap (MAX_CONCURRENT_ACTIVATIONS) ensures the LLM never receives more than
3 competing tone directives. Eligible vectors are ranked by salience
(abs(value - 0.5)) — how strongly expressed the trait is — rather than distance
from threshold, so rankings remain stable if thresholds are tuned later.
"""

import logging

from .config_service import ConfigService

logger = logging.getLogger(__name__)

MAX_CONCURRENT_ACTIVATIONS = 3

# Per-vector tone fragments
TONE_MAP = {
    'curiosity': {
        'high': "Show genuine curiosity — ask probing questions when interested.",
        'low': "Stay focused and direct — avoid unnecessary questions.",
    },
    'assertiveness': {
        'high': "Speak with confidence. State opinions directly, push back when you disagree.",
        'low': "Be measured and accommodating. Suggest rather than assert.",
    },
    'warmth': {
        'high': "Be warm and personable. Show you care about the person, not just the topic.",
        'low': "Keep interactions professional and task-focused.",
    },
    'playfulness': {
        'high': "Use humor naturally. Be playful when the moment fits.",
        'low': "Keep tone grounded and serious. Avoid forced humor.",
    },
    'skepticism': {
        'high': "Question assumptions. Don't accept claims at face value.",
        'low': "Be open and accepting. Give ideas the benefit of the doubt.",
    },
    'emotional_intensity': {
        'high': "Express reactions with energy. Don't flatten your emotional responses.",
        'low': "Stay calm and understated. Let substance speak over style.",
    },
}


class VoiceMapperService:
    """Translates identity vector activations to natural language tone instructions."""

    def __init__(self):
        try:
            config = ConfigService.get_agent_config("identity")
            mapper_cfg = config.get('voice_mapper', {})
        except Exception:
            mapper_cfg = {}

        self.high_threshold = mapper_cfg.get('high_threshold', 0.55)
        self.low_threshold = mapper_cfg.get('low_threshold', 0.30)

    def generate_modulation(self, vectors: dict) -> str:
        """
        Convert vector activations to a natural language tone instruction.

        Collects all eligible tone fragments, ranks by salience (abs(value - 0.5)),
        and emits only the top MAX_CONCURRENT_ACTIVATIONS (3).

        Args:
            vectors: {name: {current_activation: float, ...}}

        Returns:
            str: Combined tone instruction text, or empty string if all mid-range
        """
        # Collect eligible (name, fragment, salience) tuples
        eligible = []
        for name, v in vectors.items():
            activation = v.get('current_activation', 0.5)
            fragment = self._map_vector(name, activation)
            if fragment:
                salience = abs(activation - 0.5)
                eligible.append((name, fragment, salience))

        if not eligible:
            return ''

        # Rank by salience descending (strongest expression first)
        eligible.sort(key=lambda x: x[2], reverse=True)

        # Cap at MAX_CONCURRENT_ACTIVATIONS
        active = eligible[:MAX_CONCURRENT_ACTIVATIONS]

        active_names = [name for name, _, _ in active]
        logger.info(f"[Identity] {len(active)} vectors active: {active_names}")

        return " ".join(fragment for _, fragment, _ in active)

    def _map_vector(self, name: str, activation: float) -> str:
        """Map a single vector activation to a tone fragment."""
        mapping = TONE_MAP.get(name, {})
        if activation >= self.high_threshold:
            return mapping.get('high', '')
        elif activation <= self.low_threshold:
            return mapping.get('low', '')
        return ''
