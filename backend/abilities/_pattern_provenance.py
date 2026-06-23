# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared provenance derivation for the pattern-write abilities.

``save_pattern`` and ``save_graph`` are both reachable from two distinct
background passes that build separate MessageProcessors: the pattern pass
(``PatternConfig`` → channel ``pattern_match``) and the geo pass (``GeoConfig``
→ channel ``geo_pattern``). The row each ability writes should record WHICH pass
produced it, so the provenance is derived from the running processor's config
channel rather than hard-coded.
"""

from __future__ import annotations

from services.source_profiles import PROVENANCE_PATTERN_MATCH


def pattern_provenance(proc: object) -> str:
    """Reads ``proc.config.channel`` — ``pattern_match`` for the pattern pass,
    ``geo_pattern`` for the geo pass — and falls back to the shared pattern-match
    default (the historic literal) when no config/channel is present, so an
    unscoped call is indistinguishable from the pattern pass.
    """
    return getattr(getattr(proc, "config", None), "channel", None) or PROVENANCE_PATTERN_MATCH
