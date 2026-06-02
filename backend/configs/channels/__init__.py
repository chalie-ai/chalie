# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Per-channel ProcessorConfig constants and factory functions.

Spec: ACT Loop Orchestrator Refactor §3.

Static channels (§3a) — constant ProcessorConfig instances:
  DMN_CONFIG, EPISODE_ENCODER_CONFIG, SKILL_SUGGESTION_CONFIG,
  COMPACTION_CONFIG

Per-instance channels (§3b) — factory functions:
  make_user_config(metadata) -> ProcessorConfig
  make_eamp_config(agent_name, project, loop_in_human, wrapper_id) -> ProcessorConfig
  make_pattern_config(window_start, window_end) -> ProcessorConfig
  make_geo_config(window_start, window_end) -> ProcessorConfig
  make_user_summary_config() -> ProcessorConfig
  make_super_episode_config(channel, sources, spans) -> ProcessorConfig

T7 wired UMP and EAMP with real implementations.
T8 wired all background/housekeeping channels with real implementations.
"""

from __future__ import annotations

from configs.channels._common import DEFAULT_ALWAYS_AVAILABLE, DEFAULT_DISCOVERABLE
from configs.channels.compaction import COMPACTION_CONFIG
from configs.channels.dmn import DMN_CONFIG
from configs.channels.episode_encoder import EPISODE_ENCODER_CONFIG
from configs.channels.external_agent import make_eamp_config
from configs.channels.geo_pattern import make_geo_config
from configs.channels.pattern import _pattern_existing_patterns_block, make_pattern_config
from configs.channels.skill_suggestion import SKILL_SUGGESTION_CONFIG
from configs.channels.super_episode import (
    _collect_transcript_ids,
    _fetch_transcript_spans,
    _safe_json_load_object,
    make_super_episode_config,
)
from configs.channels.user import make_user_config
from configs.channels.user_summary import _should_synthesise, make_user_summary_config
from services.processor_config import ProcessorConfig

__all__ = [
    "COMPACTION_CONFIG",
    "DEFAULT_ALWAYS_AVAILABLE",
    "DEFAULT_DISCOVERABLE",
    "DMN_CONFIG",
    "EPISODE_ENCODER_CONFIG",
    "ProcessorConfig",
    "SKILL_SUGGESTION_CONFIG",
    "_collect_transcript_ids",
    "_fetch_transcript_spans",
    "_pattern_existing_patterns_block",
    "_safe_json_load_object",
    "_should_synthesise",
    "make_eamp_config",
    "make_geo_config",
    "make_pattern_config",
    "make_super_episode_config",
    "make_user_config",
    "make_user_summary_config",
]
