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
count as user-activity in the geo and pattern windows, and whether their rows
back-fill the user's live location — three orthogonal switches (``geo_is_user``,
``pattern_is_user``, ``location_backfill``).

The table is **allowlist-shaped**: a channel that is absent resolves to the
fully-muted default. Adding a write-capable channel therefore requires an
explicit profile row — silence is the safe default, never a leak.

Consumers (bidirectional dependency note):
  - services/transcript_service.py          — location back-fill
  - abilities/_pattern_provenance.py        — PROVENANCE_PATTERN_MATCH import
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from configs.enums.channels import Channel

# ── Channel keys (exact) ──────────────────────────────────────────────────────
# Named constants so no consumer hard-codes a channel literal.
CHANNEL_USER = Channel.USER.value
CHANNEL_SKILLS_BUILDING = Channel.SKILLS_BUILDING.value
CHANNEL_SCHEDULE = Channel.SCHEDULE.value
CHANNEL_DISCOVERY = Channel.DISCOVERY.value

# ── Channel patterns (SQL LIKE prefixes) ──────────────────────────────────────
# External-agent channels are tagged ``external-agent:<id>`` (HYPHEN + colon);
# delegate channels are ``delegate:<tool>``.
LIKE_EXTERNAL_AGENT = f"{Channel.EXTERNAL_AGENT.value}:%"
LIKE_DELEGATE = "delegate:%"

# ── Provenance tags ───────────────────────────────────────────────────────────
# Provenance is derived at write time from the running processor's config
# channel (see abilities/_pattern_provenance.py) and from the episode channel in
# fact extraction; this constant is the shared default for the pattern pass.
PROVENANCE_PATTERN_MATCH = Channel.PATTERN_MATCH.value


@dataclass(frozen=True)
class Profile:
    """One source's memory behaviour. Immutable — the table is a constant."""

    geo_is_user: bool
    """True → this channel's location-tagged rows count as the user's own
    movement in the geo-pattern window and advance the geo cursor."""

    pattern_is_user: bool
    """True → this channel's rows count as user behaviour in the pattern
    window."""

    location_backfill: bool
    """True → rows written on this channel with no explicit location back-fill
    the user's live coordinates. Muted channels store NULL location instead so
    background activity never masquerades as the user being somewhere."""


# Fully-muted default for any channel absent from the table.
_MUTED = Profile(
    geo_is_user=False,
    pattern_is_user=False,
    location_backfill=False,
)

# ── The allowlist table ───────────────────────────────────────────────────────
# Exact-match channels resolved first; LIKE-prefix patterns resolved second.
# Only channels that actually write transcript rows appear here. ``subagent`` and
# ``discord`` are intentionally omitted — no live config writes them.

_EXACT_PROFILES: dict[str, Profile] = {
    CHANNEL_USER: Profile(
        geo_is_user=True,
        pattern_is_user=True,
        location_backfill=True,
    ),
    # A fired schedule encodes into episodic memory like a user turn (§13.4);
    # field-for-field identical to the fully-muted default but stated explicitly
    # because a write-capable channel requires a row.
    CHANNEL_SCHEDULE: _MUTED,
    CHANNEL_SKILLS_BUILDING: _MUTED,
    # The proactive-research loop writes transcript rows, so it needs an explicit
    # row (allowlist default is muted, but a write-capable channel states it). Its
    # findings are saved as discovery memories, never re-derived as episodes/facts.
    CHANNEL_DISCOVERY: _MUTED,
}

# Profile applied to every channel matching the prefix. The first matching
# pattern in iteration order wins; patterns are mutually exclusive by design.
_PREFIX_PROFILES: tuple[tuple[str, Profile], ...] = (
    # External-agent exchanges are never user activity in the geo/pattern windows;
    # field-for-field identical to the fully-muted default but stated explicitly
    # because a write-capable channel prefix requires a row.
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


def _exact_channels(predicate: Callable[[Profile], bool]) -> list[str]:
    return [ch for ch, profile in _EXACT_PROFILES.items() if predicate(profile)]


def _prefix_patterns(predicate: Callable[[Profile], bool]) -> list[str]:
    return [pat for pat, profile in _PREFIX_PROFILES if predicate(profile)]


def _allowlist_sql(column: str, predicate: Callable[[Profile], bool]) -> str:
    """Build an allowlist SQL fragment for ``column`` over the profiles passing
    ``predicate``.

    Emits ``(column IN ('a','b') OR column LIKE 'p:%')`` — the positive,
    allowlist form so a new muted channel is excluded by default rather than
    needing a matching ``NOT LIKE``. Returns ``0`` (always-false) when no
    profile qualifies, so the caller's predicate is never vacuously true.
    """
    exact = _exact_channels(predicate)
    patterns = _prefix_patterns(predicate)
    clauses: list[str] = []
    if exact:
        quoted = ", ".join(f"'{ch}'" for ch in exact)
        clauses.append(f"{column} IN ({quoted})")
    for pat in patterns:
        clauses.append(f"{column} LIKE '{pat}'")
    if not clauses:
        return "0"
    return "(" + " OR ".join(clauses) + ")"


def geo_user_channels_sql(column: str = "channel") -> str:
    return _allowlist_sql(column, lambda p: p.geo_is_user)


def pattern_user_channels_sql(column: str = "channel") -> str:
    return _allowlist_sql(column, lambda p: p.pattern_is_user)

