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

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig


class ConfigTypeEnum(str, Enum):
    USER      = "user"
    SCHEDULED = "scheduled"
    DISCOVERY = "discovery"

    @classmethod
    def get_by_type(cls, config_type: ConfigTypeEnum | str) -> "ProcessorConfig":
        from configs.channels.discovery import DiscoveryConfig  # noqa: PLC0415
        from configs.channels.scheduled import ScheduledConfig  # noqa: PLC0415
        from configs.channels.user import UserConfig  # noqa: PLC0415

        if config_type == cls.USER:
            return UserConfig()
        if config_type == cls.SCHEDULED:
            return ScheduledConfig()
        if config_type == cls.DISCOVERY:
            return DiscoveryConfig()
        raise ValueError("Invalid type provided")
