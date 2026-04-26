"""
Review Tool Calls Skill — Drill back into raw tool call data from a previous turn.

Lets the LLM re-read raw tool call records from earlier in the conversation when
the compact synthesis in Previous Turns doesn't include a specific detail it needs.
Returns all tool calls (including ephemeral records) within ±5 minutes of a given
timestamp.
"""

import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    'name': 'review_tool_calls',
    'description': (
        'Review raw tool call data from a previous turn. Returns all tool calls '
        '(including results, steers, and synthesis) within ±5 minutes of the given '
        'timestamp. Use when the conversation history references tool use but you need '
        'specific details not captured in the synthesis.'
    ),
    'input_schema': {
        'type': 'object',
        'properties': {
            'date_time': {
                'type': 'string',
                'description': (
                    'ISO timestamp to anchor the search '
                    '(e.g. 2026-04-07T14:30:00+00:00). Tool calls within ±5 minutes '
                    'of this time will be returned.'
                ),
            },
        },
        'required': ['date_time'],
    },
}


def handle_review_tool_calls(channel: str, params: dict) -> dict:
    """Return raw tool call records within ±5 minutes of the given timestamp.

    Args:
        channel: Current conversation channel (unused but required by skill contract).
        params: {
            date_time (str, required): ISO timestamp to anchor the search.
        }

    Returns:
        dict with 'text' key containing a formatted summary of matching records.
    """
    date_time = (params.get('date_time') or '').strip()
    if not date_time:
        return {'text': "[REVIEW TOOL CALLS] Error: 'date_time' parameter is required."}

    try:
        from services.tool_call_service import ToolCallService
        records = ToolCallService().get_by_timerange(date_time, buffer_minutes=5)
    except Exception as e:
        logger.error(f"[REVIEW TOOL CALLS] Query failed for date_time={date_time!r}: {e}", exc_info=True)
        return {'text': f"[REVIEW TOOL CALLS] Error retrieving records: {str(e)[:200]}"}

    if not records:
        return {'text': f"No tool calls found within ±5 minutes of {date_time}"}

    lines = [f"[REVIEW TOOL CALLS] {len(records)} record(s) within ±5 min of {date_time}:\n"]
    for rec in records:
        tool_name = rec.get('tool_name', 'unknown')
        params_str = rec.get('params', '{}')
        result = str(rec.get('result') or '')
        status_hint = 'error' if result.lower().startswith('error') else 'ok'
        from services.time_formatter_service import TimeFormatterService
        created = TimeFormatterService.local(rec.get('created_at'), fmt='%Y-%m-%d %H:%M:%S') \
            or str(rec.get('created_at', ''))[:19]
        # Truncate long results for readability
        if len(result) > 300:
            result = result[:300] + '...'
        lines.append(f"  [{created}] {tool_name} params={params_str} → {result} ({status_hint})")

    return {'text': '\n'.join(lines)}
