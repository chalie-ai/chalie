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
    'memory', 'recall', 'find_tools',
})

# ── Deterministic actions (always succeed, high confidence) ────────────────
DETERMINISTIC_ACTIONS: frozenset = frozenset({
    'memory', 'memorize',
})

# ── Safe actions (can be silently corrected by critic without user
#    confirmation — no irreversible side effects) ──────────────────────────
SAFE_ACTIONS: frozenset = frozenset({
    'memory', 'recall', 'memorize', 'find_tools',
})

# ── Critic-skippable reads: simple reads where the critic is skipped
#    entirely when dispatcher confidence is above threshold ────────────────
CRITIC_SKIP_READS: frozenset = frozenset({
    'memory', 'recall', 'find_tools',
})

# ── Actions with explicit fatigue costs (others default to 1.0) ──────────
ACTION_FATIGUE_COSTS: MappingProxyType = MappingProxyType({
    'memory': 0.8,
    'memorize': 0.8,
    'recall': 1.0,
    'find_tools': 0.5,
})
