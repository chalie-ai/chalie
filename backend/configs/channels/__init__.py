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
from configs.channels.skill_association import SkillAssociationConfig
from configs.channels.dmn import DmnConfig
from configs.channels.episode_encoder import EpisodeEncoderConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.fact_extraction import FactExtractionConfig, parse_fact_ops
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig, _pattern_existing_patterns_block
from configs.channels.scheduled import ScheduledConfig
from configs.channels.skill_suggestion import SkillSuggestionConfig
from configs.channels.super_episode import (
    SuperEpisodeConfig,
    _collect_transcript_ids,
    _fetch_transcript_spans,
    _safe_json_load_object,
)
from configs.channels.user import UserConfig
from configs.channels.user_summary import UserSummaryConfig, _should_synthesise
from services.processor_config import ProcessorConfig

__all__ = [
    "SkillAssociationConfig",
    "DEFAULT_ALWAYS_AVAILABLE",
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
    "UserConfig",
    "UserSummaryConfig",
    "_collect_transcript_ids",
    "_fetch_transcript_spans",
    "_pattern_existing_patterns_block",
    "_safe_json_load_object",
    "_should_synthesise",
    "parse_fact_ops",
]
