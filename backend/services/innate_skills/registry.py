"""
Authoritative skill membership sets — single source of truth.

All skill/action-type sets used across the codebase MUST be defined here.
Do NOT define local skill sets elsewhere. Import from this module.

The ground-truth skill list is the set of handler keys registered by
register_innate_skills() in __init__.py (currently 12 skills).
"""

# ── Authoritative: all skills registered in register_innate_skills() ────────
ALL_SKILL_NAMES: frozenset = frozenset({
    'recall', 'memorize', 'introspect', 'associate',
    'schedule', 'autobiography', 'focus', 'list',
    'moment', 'persistent_task', 'emit_card', 'document',
})

# ── LLM-visible for planning: excludes emit_card (internal trigger) and
#    moment (context read, not a user-facing skill) ──────────────────────────
PLANNING_SKILLS: frozenset = ALL_SKILL_NAMES - frozenset({'emit_card', 'moment'})

# ── Reflection filter: innate skills whose output should NOT go to
#    experience assimilation (all innate skills are filtered out) ────────────
REFLECTION_FILTER_SKILLS: frozenset = ALL_SKILL_NAMES

# ── Cognitive primitives: always injected into ACT mode regardless of
#    triage selection. These are the foundational memory operations. ─────────
COGNITIVE_PRIMITIVES: frozenset = frozenset({
    'recall', 'memorize', 'introspect', 'associate',
})

# ── Contextual skills: planning-visible skills minus primitives.
#    Triage selects from these based on prompt analysis. ────────────────────
CONTEXTUAL_SKILLS: frozenset = PLANNING_SKILLS - COGNITIVE_PRIMITIVES

# ── Triage-valid skills: skills the triage LLM is allowed to select ────────
TRIAGE_VALID_SKILLS: frozenset = PLANNING_SKILLS

# ── Ordered primitives list: used where insertion order matters
#    (e.g., prepending to skill lists in triage) ───────────────────────────
COGNITIVE_PRIMITIVES_ORDERED: list = ['recall', 'memorize', 'introspect', 'associate']

# ── Skills tracked by procedural memory (skill_outcome_recorder).
#    Only the 4 core primitives — dynamic tools are tracked separately
#    by ToolRegistryService._log_outcome. ──────────────────────────────────
PROCEDURAL_MEMORY_SKILLS: frozenset = frozenset({
    'recall', 'memorize', 'introspect', 'associate',
})

# ── Skill descriptions for tool profile bootstrapping ─────────────────────
SKILL_DESCRIPTIONS: dict = {
    'recall': 'Search memory, retrieve stored information, look up what Chalie knows about a topic or person',
    'memorize': 'Store information, save a note, remember a fact, keep something for later',
    'introspect': 'Self-examine internal state, check how much is known about a topic, inspect confidence',
    'associate': 'Find related concepts, explore connections, brainstorm associations between ideas',
    'list': 'Manage named lists: add, remove, check off, or view items in shopping, to-do, and other lists',
    'schedule': 'Set reminders, schedule tasks, create appointments and recurring events',
    'focus': 'Start and manage deep focus or work sessions, Pomodoro-style timers',
    'autobiography': 'Generate a personal autobiography or life summary based on stored memories',
    'persistent_task': 'Create, manage, and track multi-session background tasks with state machine lifecycle',
    'moment': 'Capture and read ambient context snapshots (time, place, energy, device)',
    'emit_card': 'Render deferred tool cards into the conversation stream (internal trigger)',
    'document': 'Search, view, and manage uploaded documents with hybrid retrieval',
}
