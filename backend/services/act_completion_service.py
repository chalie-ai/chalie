"""
ACT Completion Service — Detects when expected tools were not invoked.

Extracted from tool_worker.py. Injects a [NO_ACTION_TAKEN] signal into
the act_history_context so the followup prompt knows the action failed.
"""

import logging

from services.innate_skills.registry import REFLECTION_FILTER_SKILLS

logger = logging.getLogger(__name__)


def _is_ephemeral_tool(tool_name: str) -> bool:
    """Return True if the tool declares output.ephemeral=true in its manifest."""
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        tool = registry.tools.get(tool_name)
        if tool:
            return tool.get('manifest', {}).get('output', {}).get('ephemeral', False)
    except Exception:
        pass
    return False


def inject_no_action_signal(
    act_history: list,
    act_history_context: str,
    relevant_tools_list: list,
) -> str:
    """If action-oriented tools were expected but none succeeded, prepend a signal.

    Returns the (possibly modified) act_history_context string.
    """
    expected_action_tools = [
        t['name'] for t in relevant_tools_list
        if t.get('type') == 'tool' and not _is_ephemeral_tool(t['name'])
    ]
    if not expected_action_tools:
        return act_history_context

    action_tool_used = any(
        not _is_ephemeral_tool(r.get('action_type', ''))
        and r.get('action_type', '') not in REFLECTION_FILTER_SKILLS
        and r.get('status') == 'success'
        for r in act_history
    )
    if action_tool_used:
        return act_history_context

    failed_tools = ', '.join(expected_action_tools)
    return (
        f"[NO_ACTION_TAKEN] The requested action could not be completed. "
        f"Expected tool(s) [{failed_tools}] were not successfully invoked. "
        f"Do NOT claim the action was performed.\n\n"
        + act_history_context
    )
