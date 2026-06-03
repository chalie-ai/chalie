# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Per-channel ProcessorConfig subclasses.

Spec: ACT Loop Orchestrator Refactor §3.

Each channel is a named ProcessorConfig subclass with a typed __init__.
The caller instantiates the subclass directly — no factory layer, no
extras dict.

Static channels (§3a) — zero-arg constructors:
  DmnConfig(), EpisodeEncoderConfig(), SkillSuggestionConfig(),
  CompactionConfig()

Per-instance channels (§3b) — typed constructors:
  UserConfig(metadata: dict | None = None)
  EAMPConfig(agent_name, project, loop_in_human, wrapper_id)
  PatternConfig(window_start, window_end)
  GeoConfig(window_start, window_end)
  UserSummaryConfig()
  SuperEpisodeConfig(channel, sources, spans)
"""

from __future__ import annotations

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE, DEFAULT_DISCOVERABLE
from configs.channels.compaction import CompactionConfig
from configs.channels.dmn import DmnConfig
from configs.channels.episode_encoder import EpisodeEncoderConfig
from configs.channels.external_agent import EAMPConfig
from configs.channels.geo_pattern import GeoConfig
from configs.channels.pattern import PatternConfig, _pattern_existing_patterns_block
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
    "CompactionConfig",
    "DEFAULT_ALWAYS_AVAILABLE",
    "DEFAULT_DISCOVERABLE",
    "DmnConfig",
    "EAMPConfig",
    "EpisodeEncoderConfig",
    "GeoConfig",
    "PatternConfig",
    "ProcessorConfig",
    "SkillSuggestionConfig",
    "SuperEpisodeConfig",
    "UserConfig",
    "UserSummaryConfig",
    "_collect_transcript_ids",
    "_fetch_transcript_spans",
    "_pattern_existing_patterns_block",
    "_safe_json_load_object",
    "_should_synthesise",
]
