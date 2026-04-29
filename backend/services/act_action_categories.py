"""
Authoritative action behavior categories — single source of truth.

Categorizes actions by their determinism and side-effect profile.
Consumed by ActDispatcherService for confidence scoring.
"""

# ── Read-only actions (no side effects) ────────────────────────────────────
READ_ACTIONS: frozenset = frozenset({
    'memory', 'find_tools',
})

# ── Deterministic actions (always succeed, high confidence) ────────────────
DETERMINISTIC_ACTIONS: frozenset = frozenset({
    'memory',
})
