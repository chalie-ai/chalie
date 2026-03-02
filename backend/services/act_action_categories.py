"""
Authoritative action behavior categories — single source of truth.

Categorizes actions by their safety, determinism, and side-effect profile.
Used by CriticService, ActDispatcherService, and ActLoopService.

All action category sets used across the codebase MUST be defined here.
Do NOT define local action sets elsewhere. Import from this module.
"""

from types import MappingProxyType

# ── Read-only actions (no side effects) ────────────────────────────────────
READ_ACTIONS: frozenset = frozenset({
    'recall', 'introspect', 'associate', 'autobiography',
})

# ── Deterministic actions (always succeed, high confidence) ────────────────
DETERMINISTIC_ACTIONS: frozenset = frozenset({
    'memorize', 'introspect',
})

# ── Safe actions (can be silently corrected by critic without user
#    confirmation — no irreversible side effects) ──────────────────────────
SAFE_ACTIONS: frozenset = frozenset({
    'recall', 'memorize', 'introspect', 'associate', 'autobiography', 'moment',
})

# ── Critic-skippable reads: simple reads where the critic is skipped
#    entirely when dispatcher confidence is above threshold ────────────────
CRITIC_SKIP_READS: frozenset = frozenset({
    'recall', 'introspect',
})

# ── Actions with explicit fatigue costs (others default to 1.0) ──────────
ACTION_FATIGUE_COSTS: MappingProxyType = MappingProxyType({
    'introspect': 0.5,
    'memorize': 0.8,
    'recall': 1.0,
    'associate': 1.0,
})
