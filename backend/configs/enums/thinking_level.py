# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ThinkingLevel — formalises the existing low/medium/high strings; adds MAX (additive) and NONE (floor).

NONE is the floor: explicit off where the provider supports it, else that
provider's lowest setting.  LOW = no flag sent (provider default).
The Ollama quirk (think flag gated on model capability) is preserved in
OllamaClient and not represented here.
"""

from enum import Enum


class ThinkingLevel(Enum):
    """Each thin client maps the level to its native flag internally.

    LOW is the floor — maps to "no flag" on Anthropic/OpenAI/Gemini.
    The Ollama quirk (think gated on model capability, level ignored) is
    preserved in OllamaClient and not represented here.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
