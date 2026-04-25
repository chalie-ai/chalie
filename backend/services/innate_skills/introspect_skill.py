"""
Introspect Skill — Comprehensive internal state report.

Returns four natural-language scopes covering memory health, skill and tool
usage, reasoning state, and identity snapshot. All deterministic — no LLM
calls. Every scope is wrapped in try/except so one failure cannot take down
the others.
"""

import logging

from services.time_formatter_service import TimeFormatterService

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "introspect",
    "description": (
        "Perception directed inward. Returns a comprehensive internal state report "
        "covering three scopes: memory health (episode/concept counts, working memory depth, "
        "consolidation recency), reasoning state (active focus, upcoming reminders, "
        "recent autonomous actions), and identity (relationship depth, communication style, "
        "personality). All deterministic — no LLM calls. "
        "Use when the user asks about system state, capabilities, or what you have been doing, "
        "or when you need to gauge how much context you have before deciding what to do."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
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
            rel = TimeFormatterService.ago(row[0])
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


# ── Scope 2: Reasoning State ─────────────────────────────────────


def _scope_reasoning_state() -> str:
    parts = []

    reminder_line = _reasoning_upcoming_reminders()
    if reminder_line:
        parts.append(reminder_line)

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
        next_due = TimeFormatterService.ago(rows[0][1]) if rows[0][1] else 'soon'
        label = 'upcoming reminder' if count == 1 else 'upcoming reminders'
        return f"{count} {label} (next: '{next_msg}' in {next_due})."
    except Exception as e:
        logger.debug(f'[INTROSPECT] reasoning_upcoming_reminders failed: {e}')
        return ''


# ── Scope 4: Identity Snapshot ───────────────────────────────────


def _scope_identity_snapshot() -> str:
    parts = []

    style_line = _identity_communication_style()
    if style_line:
        parts.append(style_line)

    if not parts:
        return 'Identity data unavailable.'
    return ' '.join(parts)


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


