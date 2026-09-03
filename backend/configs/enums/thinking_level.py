# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ThinkingLevel — formalises the existing low/medium/high strings; adds MAX (additive) and NONE (floor).

NONE is the floor: explicit off where the provider supports it, else that
provider's lowest setting.  LOW usually means no flag sent, taking the
provider default — but only where that default is itself cheap; on a vendor
that defaults to the top of its scale, silence would buy the *most* thinking
for the level named least, so those spell LOW out.  See thinking_map.
The Ollama quirk (think flag gated on model capability) is preserved in
OllamaClient and not represented here.
"""

from enum import Enum


class ThinkingLevel(Enum):
    """Each client maps the level to its vendor's own flag; see thinking_map.

    These names are Chalie's, not any vendor's, and the mapping is never
    identity: several vendors publish no ``medium``, one cannot be switched
    off at all, and five default to the top of their scale.
    The Ollama quirk (think gated on model capability, level ignored) is
    preserved in OllamaClient and not represented here.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
