# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-source memory profiles — the single source of truth for which channels
contribute to which memory subsystem.

A channel's *profile* declares, for one transcript source, whether its rows
back-fill the user's live location (``location_backfill``).

The table is **allowlist-shaped**: a channel that is absent resolves to the
fully-muted default. Adding a write-capable channel therefore requires an
explicit profile row — silence is the safe default, never a leak.

Consumers (bidirectional dependency note):
  - services/transcript_service.py          — location back-fill
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.enums.channels import Channel

# ── Channel keys (exact) ──────────────────────────────────────────────────────
# Named constants so no consumer hard-codes a channel literal.
CHANNEL_USER = Channel.USER.value
CHANNEL_SKILLS_BUILDING = Channel.SKILLS_BUILDING.value
CHANNEL_SCHEDULE = Channel.SCHEDULE.value
CHANNEL_DISCOVERY = Channel.DISCOVERY.value
CHANNEL_MEMORY_HYGIENE = Channel.MEMORY_HYGIENE.value

# ── Channel patterns (SQL LIKE prefixes) ──────────────────────────────────────
# External-agent channels are tagged ``external-agent:<id>`` (HYPHEN + colon);
# delegate channels are ``delegate:<tool>``.
LIKE_EXTERNAL_AGENT = f"{Channel.EXTERNAL_AGENT.value}:%"
LIKE_DELEGATE = "delegate:%"


@dataclass(frozen=True)
class Profile:
    """One source's memory behaviour. Immutable — the table is a constant."""

    location_backfill: bool
    """True → rows written on this channel with no explicit location back-fill
    the user's live coordinates. Muted channels store NULL location instead so
    background activity never masquerades as the user being somewhere."""


# Fully-muted default for any channel absent from the table.
_MUTED = Profile(location_backfill=False)

# ── The allowlist table ───────────────────────────────────────────────────────
# Exact-match channels resolved first; LIKE-prefix patterns resolved second.
# Only channels that actually write transcript rows appear here. ``subagent`` and
# ``discord`` are intentionally omitted — no live config writes them.

_EXACT_PROFILES: dict[str, Profile] = {
    CHANNEL_USER: Profile(location_backfill=True),
    # A fired schedule encodes into episodic memory like a user turn (§13.4);
    # field-for-field identical to the fully-muted default but stated explicitly
    # because a write-capable channel requires a row.
    CHANNEL_SCHEDULE: _MUTED,
    CHANNEL_SKILLS_BUILDING: _MUTED,
    # The proactive-research loop writes transcript rows, so it needs an explicit
    # row (allowlist default is muted, but a write-capable channel states it). Its
    # findings are saved as discovery memories, never re-derived as episodes/facts.
    CHANNEL_DISCOVERY: _MUTED,
    # The hygiene pass reorganizes the memory stores in place; its consolidation
    # chatter must never be re-derived into new memories.
    CHANNEL_MEMORY_HYGIENE: _MUTED,
}

# Profile applied to every channel matching the prefix. The first matching
# pattern in iteration order wins; patterns are mutually exclusive by design.
_PREFIX_PROFILES: tuple[tuple[str, Profile], ...] = (
    # External-agent exchanges are field-for-field identical to the fully-muted
    # default but stated explicitly because a write-capable channel prefix
    # requires a row.
    (LIKE_EXTERNAL_AGENT, _MUTED),
    # Delegate research loops are fully muted.
    (LIKE_DELEGATE, _MUTED),
)

def profile_for(channel: str) -> Profile:
    """Return the :class:`Profile` for a channel — fully muted when absent (allowlist default)."""
    if not channel:
        return _MUTED
    exact = _EXACT_PROFILES.get(channel)
    if exact is not None:
        return exact
    for prefix, profile in _PREFIX_PROFILES:
        if channel.startswith(prefix.rstrip("%")):
            return profile
    return _MUTED

