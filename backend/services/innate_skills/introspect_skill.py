"""
Introspect Skill — Comprehensive internal state report.

Returns four natural-language scopes covering memory health, skill and tool
usage, reasoning state, and identity snapshot. All deterministic — no LLM
calls. Every scope is wrapped in try/except so one failure cannot take down
the others.
"""

import logging

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "introspect",
    "description": (
        "Perception directed inward. Returns a comprehensive internal state report "
        "covering four scopes: memory health (episode/concept counts, working memory depth, "
        "consolidation recency), skill and tool usage (table with counts and last results), "
        "reasoning state (active focus, upcoming reminders, recent autonomous actions), "
        "and identity (relationship depth, communication style, personality, "
        "autobiography recency). All deterministic — no LLM calls. "
        "Use when the user asks about system state, capabilities, or what you have been doing, "
        "or when you need to gauge how much context you have before deciding what to do."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# Identity dimension labels and their natural-language descriptions
_DIMENSION_LABELS = {
    'curiosity': 'curious',
    'warmth': 'warm',
    'assertiveness': 'assertive',
    'skepticism': 'skeptical',
    'playfulness': 'playful',
    'uncertainty_tolerance': 'uncertainty-tolerant',
}

# Communication style dimension humanisation
_VERBOSITY_LABELS = [(8, 'very verbose'), (6, 'verbose'), (5, 'balanced'),
                     (3, 'concise'), (1, 'terse')]
_DIRECTNESS_LABELS = [(8, 'very direct'), (6, 'direct'), (5, 'balanced'),
                      (3, 'indirect'), (1, 'very indirect')]
_FORMALITY_LABELS = [(8, 'formal'), (6, 'semi-formal'), (5, 'neutral'),
                     (3, 'casual'), (1, 'very casual')]


def handle_introspect(channel: str, params: dict) -> str:
    """
    Return a comprehensive natural-language internal state report.

    Args:
        channel: Current conversation channel (unused — all scopes are global)
        params: {} (no parameters required)

    Returns:
        Multi-section natural language report covering memory, skills,
        reasoning state, and identity.
    """
    sections = [
        ('[INTROSPECT]', ''),
        ('## Memory Health', _scope_memory_health()),
        ('## Skill & Tool Usage', _scope_skill_tool_usage()),
        ('## Reasoning State', _scope_reasoning_state()),
        ('## Identity', _scope_identity_snapshot()),
    ]

    parts = []
    for header, body in sections:
        if header == '[INTROSPECT]':
            parts.append(header)
        else:
            parts.append(f'\n{header}\n{body}')

    return '\n'.join(parts)


# ── Scope 1: Memory Health ───────────────────────────────────────


def _scope_memory_health() -> str:
    try:
        from services.self_model_service import SelfModelService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        snapshot = SelfModelService(db).get_snapshot()
        pressure = snapshot.get('operational', {}).get('memory_pressure', {})
        epistemic = snapshot.get('epistemic', {})

        episode_count = pressure.get('episode_count', 0)
        concept_count = pressure.get('concept_count', 0)
        wm_depth = epistemic.get('working_memory_depth', 0)

        # Health label based on episode count
        if episode_count >= 200:
            health_label = 'healthy'
        elif episode_count >= 50:
            health_label = 'developing'
        else:
            health_label = 'sparse'

        # Last consolidation timestamp
        last_consolidation = _query_last_consolidation(db)

        # Semantic layer structure
        structure = _query_semantic_structure(db)

        return (
            f'Memory: {health_label} — {episode_count:,} episodes, '
            f'{concept_count:,} concepts, {wm_depth}/12 working memory slots'
            f'{last_consolidation}.{structure}'
        )
    except Exception as e:
        logger.debug(f'[INTROSPECT] memory_health scope failed: {e}')
        return 'Memory: unavailable.'


def _query_last_consolidation(db) -> str:
    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT MAX(updated_at) FROM episodes')
            row = cursor.fetchone()
            cursor.close()
        if row and row[0]:
            rel = _relative_time(row[0])
            return f', last consolidation {rel}'
        return ''
    except Exception:
        return ''


def _query_semantic_structure(db) -> str:
    """Summarise the data graph: top kinds, relationship count, density."""
    try:
        with db.connection() as conn:
            cursor = conn.cursor()

            # Top kinds in data_graph table
            cursor.execute(
                """
                SELECT kind, COUNT(*) AS c
                FROM data_graph
                WHERE deleted_at IS NULL AND active=1
                GROUP BY kind
                ORDER BY c DESC
                LIMIT 5
                """
            )
            kind_rows = cursor.fetchall()

            # Contact count (contact: prefix in user_specific)
            cursor.execute(
                "SELECT COUNT(*) FROM data_graph WHERE kind = 'user_specific' "
                "AND key LIKE 'contact:%' AND deleted_at IS NULL AND active=1"
            )
            rel_count = cursor.fetchone()[0] or 0

            # user_specific count for density
            cursor.execute(
                "SELECT COUNT(*) FROM data_graph WHERE kind = 'user_specific' "
                "AND deleted_at IS NULL AND active=1"
            )
            concept_count = cursor.fetchone()[0] or 0

            cursor.close()

        if not kind_rows and rel_count == 0:
            return ''

        parts = []
        if kind_rows:
            kind_strs = [f'{row[0]} ({row[1]})' for row in kind_rows]
            parts.append(f'Knowledge kinds: {", ".join(kind_strs)}')

        if concept_count > 0:
            density = rel_count / concept_count if concept_count else 0
            parts.append(
                f'{rel_count:,} relationships ({density:.1f} per concept)'
            )

        return '\n' + '; '.join(parts) + '.'

    except Exception as e:
        logger.debug(f'[INTROSPECT] semantic_structure failed: {e}')
        return ''


# ── Scope 2: Skill & Tool Usage ──────────────────────────────────


def _scope_skill_tool_usage() -> str:
    rows = []

    # External tools via user_tool_preferences
    rows.extend(_external_tool_rows())

    if not rows:
        return 'No usage data recorded yet.'

    header = (
        '| Name            | Summary                                           '
        '| Uses | Last Used     | Last Result |\n'
        '|-----------------|---------------------------------------------------|'
        '------|---------------|-------------|'
    )
    table_rows = '\n'.join(rows)
    return f'{header}\n{table_rows}'


def _external_tool_rows() -> list:
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT tool_name, usage_count, success_count, last_used_at '
                'FROM user_tool_preferences '
                'WHERE usage_count > 0 '
                'ORDER BY last_used_at DESC'
            )
            tool_rows = cursor.fetchall()
            cursor.close()

        rows = []
        for row in tool_rows:
            tool_name = row[0]
            usage_count = row[1] or 0
            success_count = row[2] or 0
            last_used_at = row[3] or ''
            last_used = _relative_time(last_used_at) if last_used_at else 'never'
            last_result = 'success' if success_count >= usage_count * 0.5 else 'failed'
            summary = _tool_summary(tool_name)[:50]

            rows.append(
                f'| {tool_name:<15} | {summary:<50}| {usage_count:<4} | {last_used:<13} | {last_result:<11} |'
            )
        return rows
    except Exception as e:
        logger.debug(f'[INTROSPECT] external_tool_rows failed: {e}')
        return []


def _tool_summary(tool_name: str) -> str:
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT short_summary FROM tool_capability_profiles WHERE tool_name = ? LIMIT 1',
                (tool_name,)
            )
            row = cursor.fetchone()
            cursor.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return ''


# ── Scope 3: Reasoning State ─────────────────────────────────────


def _scope_reasoning_state() -> str:
    parts = []

    # Upcoming reminders
    reminder_line = _reasoning_upcoming_reminders()
    if reminder_line:
        parts.append(reminder_line)

    # Recent autonomous actions
    action_line = _reasoning_recent_actions()
    if action_line:
        parts.append(action_line)

    if not parts:
        return 'No active reasoning state.'
    return '\n'.join(parts)



def _reasoning_upcoming_reminders() -> str:
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message, due_at FROM scheduled_items "
                "WHERE status = 'pending' AND due_at > datetime('now') AND hidden = 0 "
                "ORDER BY due_at ASC LIMIT 5"
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return ''

        count = len(rows)
        next_msg = (rows[0][0] or '').strip()[:40]
        next_due = _relative_time(rows[0][1]) if rows[0][1] else 'soon'
        label = 'upcoming reminder' if count == 1 else 'upcoming reminders'
        return f"{count} {label} (next: '{next_msg}' in {next_due})."
    except Exception as e:
        logger.debug(f'[INTROSPECT] reasoning_upcoming_reminders failed: {e}')
        return ''


def _reasoning_recent_actions() -> str:
    RELEVANT_TYPES = ('proactive_sent', 'plan_proposed')
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        placeholders = ','.join(['?'] * len(RELEVANT_TYPES))
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT event_type, payload, created_at '
                f'FROM interaction_log '
                f'WHERE event_type IN ({placeholders}) '
                f'ORDER BY created_at DESC LIMIT 3',
                RELEVANT_TYPES
            )
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return ''

        summaries = []
        for row in rows:
            event_type = row[0] or ''
            created_at = row[2] or ''
            when = _relative_time(created_at) if created_at else 'recently'
            label = _action_label(event_type)
            summaries.append(f'{label} {when}')

        return 'Recent: ' + ', '.join(summaries) + '.'
    except Exception as e:
        logger.debug(f'[INTROSPECT] reasoning_recent_actions failed: {e}')
        return ''


def _action_label(event_type: str) -> str:
    labels = {
        'proactive_sent': 'sent proactive message',
        'plan_proposed': 'proposed plan',
    }
    return labels.get(event_type, event_type.replace('_', ' '))


# ── Scope 4: Identity Snapshot ───────────────────────────────────


def _scope_identity_snapshot() -> str:
    parts = []

    # Relationship depth from interaction count
    interaction_label = _identity_relationship_depth()
    if interaction_label:
        parts.append(interaction_label)

    # Communication style
    style_line = _identity_communication_style()
    if style_line:
        parts.append(style_line)

    # Personality from identity vectors
    personality_line = _identity_personality()
    if personality_line:
        parts.append(personality_line)

    # Autobiography recency
    auto_line = _identity_autobiography()
    if auto_line:
        parts.append(auto_line)

    if not parts:
        return 'Identity data unavailable.'
    return ' '.join(parts)


def _identity_relationship_depth() -> str:
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM interaction_log')
            row = cursor.fetchone()
            cursor.close()

        count = row[0] if row else 0

        if count < 50:
            depth = 'new'
        elif count < 500:
            depth = 'developing'
        elif count < 2000:
            depth = 'established'
        else:
            depth = 'deep'

        return f'Relationship: {depth} (~{count:,} interactions).'
    except Exception as e:
        logger.debug(f'[INTROSPECT] identity_relationship_depth failed: {e}')
        return ''


def _identity_communication_style() -> str:
    try:
        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()
        rows = dgs.fetch(kinds=['user_specific'])

        # Look up communication style traits from data_graph
        style = {}
        for dim in ('verbosity', 'directness', 'formality'):
            key = f'communication_style_{dim}'
            entry = next((r for r in rows if r.get('key') == key), None)
            if entry and entry.get('value'):
                try:
                    style[dim] = float(entry['value'])
                except (ValueError, TypeError):
                    pass

        if not style:
            return ''

        descriptors = []

        verbosity = style.get('verbosity')
        if verbosity is not None:
            for threshold, label in _VERBOSITY_LABELS:
                if verbosity >= threshold:
                    descriptors.append(label)
                    break

        directness = style.get('directness')
        if directness is not None:
            for threshold, label in _DIRECTNESS_LABELS:
                if directness >= threshold:
                    descriptors.append(label)
                    break

        formality = style.get('formality')
        if formality is not None:
            for threshold, label in _FORMALITY_LABELS:
                if formality >= threshold:
                    descriptors.append(label)
                    break

        if not descriptors:
            return ''
        return f'Communication style: {", ".join(descriptors)}.'
    except Exception as e:
        logger.debug(f'[INTROSPECT] identity_communication_style failed: {e}')
        return ''


def _identity_personality() -> str:
    try:
        from services.identity_service import IdentityService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        vectors = IdentityService(db).get_vectors()
        if not vectors:
            return ''

        # Pick top dimensions by current_activation, threshold > 0.6
        scored = []
        for dim_name, data in vectors.items():
            activation = data.get('current_activation', 0.0)
            if activation > 0.6 and dim_name in _DIMENSION_LABELS:
                scored.append((activation, _DIMENSION_LABELS[dim_name]))

        scored.sort(reverse=True)
        top = [label for _, label in scored[:3]]
        if not top:
            return ''
        return f'Personality leans toward {" and ".join(top)}.'
    except Exception as e:
        logger.debug(f'[INTROSPECT] identity_personality failed: {e}')
        return ''


def _identity_autobiography() -> str:
    try:
        from services.autobiography_service import AutobiographyService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        narrative = AutobiographyService(db).get_current_narrative()
        if not narrative:
            return 'No self-narrative synthesized yet.'

        created_at = narrative.get('created_at', '')
        if created_at:
            when = _relative_time(created_at)
            return f'Self-narrative last updated {when}.'
        return 'Self-narrative exists.'
    except Exception as e:
        logger.debug(f'[INTROSPECT] identity_autobiography failed: {e}')
        return ''


# ── Time helpers ─────────────────────────────────────────────────


def _relative_time(dt_value) -> str:
    """Convert a datetime string or object to a human-readable relative string."""
    try:
        dt = parse_utc(dt_value)
        now = utc_now()
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < 0:
            return 'in the future'
        if seconds < 60:
            return 'just now'
        if seconds < 3600:
            mins = seconds // 60
            unit = 'min' if mins == 1 else 'min'
            return f'{mins} {unit} ago'
        if seconds < 86400:
            hours = seconds // 3600
            unit = 'hour' if hours == 1 else 'hours'
            return f'{hours} {unit} ago'
        days = seconds // 86400
        unit = 'day' if days == 1 else 'days'
        return f'{days} {unit} ago'
    except Exception:
        return 'unknown'
