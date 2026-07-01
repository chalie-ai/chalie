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
become episodes, whether they feed fact extraction, whether they count as
user-activity in the geo and pattern windows, and whether their rows back-fill
the user's live location — five orthogonal switches (``extract_episodes``,
``extract_facts``, ``geo_is_user``, ``pattern_is_user``, ``location_backfill``).
Provenance is not a profile field: each derived row is tagged at write time from
the running processor's config channel (pattern/geo passes) or the source
episode's channel (fact extraction).

The table is **allowlist-shaped**: a channel that is absent resolves to the
fully-muted default. Adding a write-capable channel therefore requires an
explicit profile row — silence is the safe default, never a leak.

Consumers (bidirectional dependency note):
  - services/transcript_service.py          — episode gate + location back-fill
  - services/subconscious_worker.py         — consolidation set, geo cursor
  - services/decay_engine_service.py        — janitor HEAVY-channel protection
  - configs/channels/pattern.py             — pattern-window channel filter
  - configs/channels/geo_pattern.py         — geo-window channel filter

This module lives in ``services/`` (not ``configs/channels/``) so the SQL
readers under ``configs/channels`` can import it without a configs→services
import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ── Channel keys (exact) ──────────────────────────────────────────────────────
# Named constants so no consumer hard-codes a channel literal.
CHANNEL_USER = "user"
CHANNEL_DMN = "dmn"
CHANNEL_SKILLS_BUILDING = "skills_building"
CHANNEL_SCHEDULE = "schedule"
CHANNEL_DISCOVERY = "discovery"

# ── Channel patterns (SQL LIKE prefixes) ──────────────────────────────────────
# External-agent channels are tagged ``external-agent:<id>`` (HYPHEN + colon);
# delegate channels are ``delegate:<tool>``.
LIKE_EXTERNAL_AGENT = "external-agent:%"
LIKE_DELEGATE = "delegate:%"

# ── Provenance tags ───────────────────────────────────────────────────────────
# Provenance is derived at write time from the running processor's config
# channel (see abilities/_pattern_provenance.py) and from the episode channel in
# fact extraction; this constant is the shared default for the pattern pass.
PROVENANCE_PATTERN_MATCH = "pattern_match"


@dataclass(frozen=True)
class Profile:
    """One source's memory behaviour. Immutable — the table is a constant."""

    extract_episodes: bool
    """True → this channel's transcript rows trigger episode extraction."""

    extract_facts: bool
    """True → this channel's episodes are eligible for fact extraction.
    (Fact extraction drains a channel-agnostic backlog; this flag documents
    intent and is the basis for any future gating — the backlog itself never
    filters out an episode-producing channel.)"""

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
    extract_episodes=False,
    extract_facts=False,
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
        extract_episodes=True,
        extract_facts=True,
        geo_is_user=True,
        pattern_is_user=True,
        location_backfill=True,
    ),
    # DMN is HEAVY: full episode + fact participation, but its reflection loop is
    # not the user moving through the world, so it is excluded from geo/pattern
    # user-activity and never back-fills the user's live location.
    CHANNEL_DMN: Profile(
        extract_episodes=True,
        extract_facts=True,
        geo_is_user=False,
        pattern_is_user=False,
        location_backfill=False,
    ),
    # A schedule thread encodes into episodic memory like a user turn (§13.4):
    # full episode + fact participation. Excluded from geo/pattern user-activity
    # and location back-fill — a fired task is not the user moving through the
    # world (the same exclusions as DMN).
    CHANNEL_SCHEDULE: Profile(
        extract_episodes=True,
        extract_facts=True,
        geo_is_user=False,
        pattern_is_user=False,
        location_backfill=False,
    ),
    CHANNEL_SKILLS_BUILDING: _MUTED,
    # The proactive-research loop writes transcript rows, so it needs an explicit
    # row (allowlist default is muted, but a write-capable channel states it). Its
    # findings are saved as discovery memories, never re-derived as episodes/facts.
    CHANNEL_DISCOVERY: _MUTED,
}

# Profile applied to every channel matching the prefix. The first matching
# pattern in iteration order wins; patterns are mutually exclusive by design.
_PREFIX_PROFILES: tuple[tuple[str, Profile], ...] = (
    # External-agent exchanges produce episodes + facts (channel-tagged), but are
    # never user activity in the geo/pattern windows.
    (
        LIKE_EXTERNAL_AGENT,
        Profile(
            extract_episodes=True,
            extract_facts=True,
            geo_is_user=False,
            pattern_is_user=False,
            location_backfill=False,
        ),
    ),
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


def janitor_protected_sql(column: str = "channel") -> str:
    """Allowlist fragment selecting the HEAVY channels the fossil janitor must
    NOT tombstone (their leaves are legitimate apex memory).

    HEAVY = channels that produce episodes. dmn never consolidates, so its
    leaves are permanently apex and would otherwise be reaped after the fossil
    cutoff; protecting every episode-producing channel keeps that memory.
    """
    return _allowlist_sql(column, lambda p: p.extract_episodes)


def consolidating_exact_channels() -> list[str]:
    """Return the EXACT channels whose episodes are consolidated into
    super-episodes — every episode-producing exact channel.

    External-agent channels are discovered at runtime from the episodes table
    (one channel per agent id), so they are not enumerable here; the worker
    unions these exact channels with the live external-agent set.
    """
    return _exact_channels(lambda p: p.extract_episodes)
