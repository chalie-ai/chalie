# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-channel ProcessorConfig subclasses and re-exports."""

from __future__ import annotations

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE
from configs.channels.discovery import DiscoveryConfig
from configs.channels.dmn import DmnConfig
from configs.channels.episode_encoder import EpisodeEncoderConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.fact_extraction import FactExtractionConfig
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig
from configs.channels.scheduled import ScheduledConfig
from configs.channels.skill_association import SkillAssociationConfig
from configs.channels.skill_suggestion import SkillSuggestionConfig
from configs.channels.super_episode import SuperEpisodeConfig
from configs.channels.thread_gist import ThreadGistConfig
from configs.channels.user import UserConfig
from configs.channels.user_summary import UserSummaryConfig
from configs.channels.vision import VisionConfig
from configs.channels.web_browse import WebBrowseConfig
from configs.channels.web_search import WebSearchConfig
from configs.enums.config_type import ConfigTypeEnum
from services.processor_config import ProcessorConfig


class ChannelConfig:
    """Factory for routing ``ConfigTypeEnum`` / wire strings to ``ProcessorConfig`` instances."""

    @staticmethod
    def config_for(config_type: "ConfigTypeEnum | str") -> ProcessorConfig:
        """Map a routing type to its ProcessorConfig.

        The single factory used by the thread API and turn-execution services to
        resolve a ``ConfigTypeEnum`` (or its wire string) into the concrete config
        subclass that drives that channel.
        """
        if config_type == ConfigTypeEnum.USER:
            return UserConfig()
        if config_type == ConfigTypeEnum.SCHEDULED:
            return ScheduledConfig()
        if config_type == ConfigTypeEnum.DISCOVERY:
            return DiscoveryConfig()
        raise ValueError("Invalid type provided")


__all__ = [
    "SkillAssociationConfig",
    "DEFAULT_ALWAYS_AVAILABLE",
    "DiscoveryConfig",
    "DmnConfig",
    "EAMPConfig",
    "EpisodeEncoderConfig",
    "FactExtractionConfig",
    "GeoConfig",
    "PatternConfig",
    "ProcessorConfig",
    "ScheduledConfig",
    "SkillSuggestionConfig",
    "SuperEpisodeConfig",
    "ThreadGistConfig",
    "UserConfig",
    "UserSummaryConfig",
    "VisionConfig",
    "WebBrowseConfig",
    "WebSearchConfig",
    "ChannelConfig",
]
