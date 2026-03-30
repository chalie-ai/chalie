"""
Authoritative skill membership sets — single source of truth.

All skill/action-type sets used across the codebase MUST be defined here.
Do NOT define local skill sets elsewhere. Import from this module.

The ground-truth skill list is the set of handler keys registered by
register_innate_skills() in __init__.py (currently 13 skills + backward-compat aliases).
"""

# ── Authoritative: all skills registered in register_innate_skills() ────────
ALL_SKILL_NAMES: frozenset = frozenset({
    'memory', 'introspect', 'associate',
    'schedule', 'autobiography', 'focus', 'list',
    'persistent_task', 'document',
    'read', 'reflect', 'find_tools', 'goals',
    'rich_render',
})

# Backward-compat aliases — old names that map to the unified 'memory' skill.
# These are NOT in ALL_SKILL_NAMES (they are aliases, not canonical skills).
SKILL_ALIASES: dict = {
    'recall': 'memory',
    'memorize': 'memory',
}

# ── LLM-visible for planning ──────────────────────────────────────────────
PLANNING_SKILLS: frozenset = ALL_SKILL_NAMES

# ── Reflection filter: innate skills whose output should NOT go to
#    experience assimilation (all innate skills are filtered out) ────────────
REFLECTION_FILTER_SKILLS: frozenset = ALL_SKILL_NAMES

# ── Cognitive primitives: always injected into ACT mode regardless of
#    triage selection. These are the foundational memory operations. ─────────
COGNITIVE_PRIMITIVES: frozenset = frozenset({
    'memory', 'associate', 'find_tools',
})

# ── Contextual skills: planning-visible skills minus primitives.
#    Triage selects from these based on prompt analysis. ────────────────────
CONTEXTUAL_SKILLS: frozenset = PLANNING_SKILLS - COGNITIVE_PRIMITIVES

# ── Triage-valid skills: skills the triage LLM is allowed to select ────────
TRIAGE_VALID_SKILLS: frozenset = PLANNING_SKILLS

# ── Ordered primitives list: used where insertion order matters
#    (e.g., prepending to skill lists in triage) ───────────────────────────
COGNITIVE_PRIMITIVES_ORDERED: list = ['memory', 'associate', 'find_tools']

# ── Skill descriptions for tool profile bootstrapping ─────────────────────
SKILL_DESCRIPTIONS: dict = {
    'memory': 'Store, recall, update, or forget knowledge — facts, preferences, traits, concepts, procedures, metrics, or arbitrary key-value data',
    'introspect': 'Self-examine internal state, check how much is known about a topic, inspect confidence',
    'associate': 'Find related concepts, explore connections, brainstorm associations between ideas',
    'list': 'Manage named lists: add, remove, check off, or view items in shopping, to-do, and other lists',
    'schedule': 'Set reminders, schedule tasks, create appointments and recurring events',
    'focus': 'Start and manage deep focus or work sessions, Pomodoro-style timers',
    'autobiography': 'Generate a personal autobiography or life summary based on stored memories',
    'persistent_task': 'Create, manage, track, complete, and update progress on multi-session background tasks with state machine lifecycle',
    'document': 'Search, view, and manage uploaded documents with hybrid retrieval',
    'read': 'Fetch and read web page content for information gathering and research',
    'reflect': 'Synthesize recent experience into insights — what worked, what didn\'t, patterns noticed, connections formed',
    'find_tools': 'Search for registered tools and capabilities by describing what you need — discover web search, email, calendar, messaging, and other integrations on demand',
    'goals': 'Create, view, confirm, complete, dismiss tracked goals, see accumulated evidence, or get a narrative synthesis of goal evolution',
    'rich_render': 'Returns block rendering reference for producing rich visual output (metrics, cards, charts, progress, timelines)',
    # Backward-compat aliases (for lookup only)
    'recall': 'Search memory, retrieve stored information, look up what Chalie knows about a topic or person',
    'memorize': 'Store information, save a note, remember a fact, keep something for later',
}

# ── Skill effort tiers (innate skills are controlled by us — no injection risk) ─
SKILL_EFFORT: dict = {
    'memory': 'trivial',
    'introspect': 'trivial',
    'associate': 'light',
    'list': 'trivial',
    'schedule': 'light',
    'focus': 'trivial',
    'autobiography': 'light',
    'persistent_task': 'deep',
    'document': 'light',
    'read': 'light',
    'reflect': 'light',
    'find_tools': 'trivial',
    'goals': 'trivial',
    'rich_render': 'trivial',
    # Backward-compat aliases
    'recall': 'trivial',
    'memorize': 'trivial',
}

# ── Skill categories ───────────────────────────────────────────────────────────
SKILL_CATEGORIES: dict = {
    'memory': 'memory',
    'introspect': 'cognition',
    'associate': 'cognition',
    'list': 'productivity',
    'schedule': 'productivity',
    'focus': 'productivity',
    'autobiography': 'identity',
    'persistent_task': 'task_management',
    'document': 'knowledge',
    'read': 'research',
    'reflect': 'cognition',
    'find_tools': 'cognition',
    'goals': 'cognition',
    'rich_render': 'presentation',
    # Backward-compat aliases
    'recall': 'memory',
    'memorize': 'memory',
}
