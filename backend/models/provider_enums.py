from __future__ import annotations

from enum import Enum


class ProviderType(Enum):
    CHAT = "chat"
    VISION = "vision"
    DELEGATE = "delegate"
    VISUAL_OUTPUT = "visual_output"  # reserved, not wired


class ThinkingLevel(Enum):
    """Formalises the existing low/medium/high strings; adds MAX (additive).

    Each thin client maps the level to its native flag internally.
    LOW is the floor — maps to "no flag" on Anthropic/OpenAI/Gemini.
    The Ollama quirk (think gated on model capability, level ignored) is
    preserved in OllamaClient and not represented here.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
