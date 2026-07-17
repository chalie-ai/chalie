# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Channel — the transcript/telemetry namespace string scoping a conversation.

Every ``ProcessorConfig`` carries a ``channel``: the value written to (or
telemetry-tagged on) the ``transcript`` table's ``channel`` column. This enum
catalogues every channel the system currently defines, so callers can reference
them by name instead of repeating raw strings.

The ``external-agent`` channel is dynamic — one per registered external agent,
spelled ``external-agent:{name}``. ``EXTERNAL_AGENT`` holds the prefix; use
:meth:`for_external_agent` to build a concrete value.
"""

from __future__ import annotations

from enum import Enum


class Channel(str, Enum):
    # ── User-facing spines ───────────────────────────────────────────────────
    USER          = "user"
    DISCOVERY     = "discovery"
    SCHEDULE      = "schedule"

    # ── Reflection / subconscious ────────────────────────────────────────────
    DMN               = "dmn"
    USER_SUMMARY      = "user_summary"
    PATTERN_MATCH     = "pattern_match"
    GEO_PATTERN       = "geo_pattern"
    SKILLS_BUILDING   = "skills_building"
    SKILL_ASSOCIATION = "skill_association"

    # ── Episodic memory ──────────────────────────────────────────────────────
    EPISODE_ENCODER       = "episode_encoder"
    SUPER_EPISODE_ENCODER = "super_episode_encoder"

    # ── Fact extraction / compaction ─────────────────────────────────────────
    FACT_EXTRACTION = "fact_extraction"
    COMPACTION      = "compaction"

    # ── Delegates (tool-driven sub-loops) ────────────────────────────────────
    DELEGATE_PIM        = "delegate:pim"
    DELEGATE_THREAD_GIST = "delegate:thread_gist"
    DELEGATE_WEB_BROWSE  = "delegate:web_browse"
    DELEGATE_WEB_SEARCH  = "delegate:web_search"
    DELEGATE_VISION      = "delegate:vision"

    # ── External agents (dynamic: external-agent:{name}) ─────────────────────
    EXTERNAL_AGENT = "external-agent"

    @classmethod
    def for_external_agent(cls, name: str) -> str:
        """Build the dynamic ``external-agent:{name}`` channel string."""
        return f"{cls.EXTERNAL_AGENT.value}:{name}"
