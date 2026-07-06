# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PolicyChannel — which policy channel a processor's tool calls are gated under."""

from __future__ import annotations

from enum import Enum


class PolicyChannel(str, Enum):
    CHAT           = "chat"
    SUBCONSCIOUS   = "subconscious"
    EXTERNAL_AGENT = "external_agent"
