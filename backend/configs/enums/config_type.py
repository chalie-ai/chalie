# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ConfigTypeEnum — top-level API routing identifier for ProcessorConfig.

Distinct from ``channel`` (the transcript channel string) and
``PolicyChannel`` (policy gating). The three configs exposed via the thread
API carry a type; every other config has none.
"""

from enum import Enum


class ConfigTypeEnum(str, Enum):
    USER      = "user"
    SCHEDULED = "scheduled"
    DISCOVERY = "discovery"
