"""
Authoritative skill membership sets — single source of truth.

All skill/action-type sets used across the codebase MUST be defined here.
Do NOT define local skill sets elsewhere. Import from this module.

The ground-truth skill list is the set of handler keys registered by
register_innate_skills() in __init__.py (currently 12 skills + backward-compat aliases).
"""

# ── Authoritative: all skills registered in register_innate_skills() ────────
ALL_SKILL_NAMES: frozenset = frozenset({
    'memory', 'introspect',
    'schedule', 'list',
    'goal_pursuit', 'document',
    'read', 'find_tools', 'goals',
    'rich_render', 'review_tool_calls',
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
    'memory', 'find_tools', 'review_tool_calls',
})

# ── Contextual skills: planning-visible skills minus primitives.
#    Triage selects from these based on prompt analysis. ────────────────────
CONTEXTUAL_SKILLS: frozenset = PLANNING_SKILLS - COGNITIVE_PRIMITIVES

# ── Triage-valid skills: skills the triage LLM is allowed to select ────────
TRIAGE_VALID_SKILLS: frozenset = PLANNING_SKILLS

# ── Ordered primitives list: used where insertion order matters
#    (e.g., prepending to skill lists in triage) ───────────────────────────
COGNITIVE_PRIMITIVES_ORDERED: list = ['memory', 'find_tools', 'review_tool_calls']

# ── Skill descriptions for tool profile bootstrapping ─────────────────────
SKILL_DESCRIPTIONS: dict = {
    'memory': 'Store, recall, or update knowledge — facts, preferences, traits, concepts, procedures, metrics, or arbitrary key-value data',
    'introspect': 'Self-examine internal state, check how much is known about a topic, inspect confidence',
    'list': 'Manage named lists: add, remove, check off, or view items in shopping, to-do, and other lists',
    'schedule': 'Set reminders, schedule tasks, create appointments and recurring events',
    'goal_pursuit': 'Use this tool to launch a separate instance of yourself with unconstrained limits to research or perform long running tasks. Only use this if you expect a request to take longer than 2 minutes to complete. If not, use other tools to resolve the request.',
    'document': 'Search, view, and manage uploaded documents with hybrid retrieval',
    'read': 'Fetch and read web page content for information gathering and research',
    'find_tools': 'Search for registered tools and capabilities by describing what you need — discover web search, email, calendar, messaging, and other integrations on demand',
    'goals': 'Create, view, confirm, complete, dismiss tracked goals, see accumulated evidence, or get a narrative synthesis of goal evolution',
    'rich_render': 'Returns block rendering reference for producing rich visual output (metrics, cards, charts, progress, timelines)',
    'review_tool_calls': 'Review raw tool call records from a previous turn by timestamp — drill into details not captured in the synthesis',
    # Backward-compat aliases (for lookup only)
    'recall': 'Search memory, retrieve stored information, look up what Chalie knows about a topic or person',
    'memorize': 'Store information, save a note, remember a fact, keep something for later',
}

# ── Skill effort tiers (innate skills are controlled by us — no injection risk) ─
SKILL_EFFORT: dict = {
    'memory': 'trivial',
    'introspect': 'trivial',
    'list': 'trivial',
    'schedule': 'light',
    'goal_pursuit': 'deep',
    'document': 'light',
    'read': 'light',
    'find_tools': 'trivial',
    'goals': 'trivial',
    'rich_render': 'trivial',
    'review_tool_calls': 'trivial',
    # Backward-compat aliases
    'recall': 'trivial',
    'memorize': 'trivial',
}

# ── Mode gate index: which cognitive modes each non-primitive skill serves ─────
# Used by ModeGateService to resolve which tools to promote for active modes.
# Primitives (memory, find_tools, review_tool_calls) are ALWAYS injected —
# they must NOT appear here. Every non-primitive innate skill MUST be listed
# (boot validation enforces this). Empty list [] is legal: "never promote,
# find_tools only". See backend/services/mode_gate_service.py and spec §5.3.
SKILL_MODES: dict[str, list[str]] = {
    'introspect':   ['converse', 'analyze'],
    'list':         ['plan', 'write', 'research', 'brainstorm'],
    'schedule':     ['plan', 'brainstorm'],
    'goal_pursuit': ['plan', 'research'],
    'document':     ['research', 'analyze', 'write', 'brainstorm'],
    'read':         ['research', 'analyze', 'write'],
    'goals':        ['plan'],
    'rich_render':  ['analyze', 'plan', 'write', 'brainstorm'],
}

# ── Skill categories ───────────────────────────────────────────────────────────
SKILL_CATEGORIES: dict = {
    'memory': 'memory',
    'introspect': 'cognition',
    'list': 'productivity',
    'schedule': 'productivity',
    'goal_pursuit': 'task_management',
    'document': 'knowledge',
    'read': 'research',
    'find_tools': 'cognition',
    'goals': 'cognition',
    'rich_render': 'presentation',
    'review_tool_calls': 'cognition',
    # Backward-compat aliases
    'recall': 'memory',
    'memorize': 'memory',
}
