# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ProviderType — routes a provider call to the client tier that should serve it."""

from __future__ import annotations

from enum import Enum


class ProviderType(Enum):
    CHAT = "chat"
    VISION = "vision"
    DELEGATE = "delegate"
    VISUAL_OUTPUT = "visual_output"  # reserved, not wired
