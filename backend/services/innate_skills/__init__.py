"""
Innate Skills Package — Built-in cognitive skills for the ACT loop.

Skills provide the ACT loop LLM with data, capabilities, and access to systems.
They are tools, not thoughts — the ACT prompt is the sole reasoner.

All innate skills are non-LLM operations (fast, sub-cortical services).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: All skill/action-type sets MUST be defined in:
  - services/innate_skills/registry.py (skill membership)
  - services/act_action_categories.py (action behavior categories)
Do NOT define local skill sets elsewhere. Import from the registry.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from services.innate_skills.memory_skill import handle_memory
from services.innate_skills.introspect_skill import handle_introspect
from services.innate_skills.scheduler_skill import handle_scheduler
from services.innate_skills.autobiography_skill import handle_autobiography
from services.innate_skills.list_skill import handle_list
from services.innate_skills.goal_pursuit_skill import handle_goal_pursuit
from services.innate_skills.document_skill import handle_document
from services.innate_skills.read_skill import handle_read
from services.innate_skills.find_tools_skill import handle_find_tools
from services.innate_skills.goals_skill import handle_goals
from services.innate_skills.rich_render_skill import handle_rich_render
from services.innate_skills.review_tool_calls_skill import handle_review_tool_calls

# Backward-compat aliases: old names point to unified memory handler
handle_recall = handle_memory
handle_memorize = handle_memory


_SKILL_HANDLERS = {
    'memory': handle_memory,
    'recall': handle_memory,       # backward-compat alias
    'memorize': handle_memory,     # backward-compat alias
    'introspect': handle_introspect,
    'schedule': handle_scheduler,
    'autobiography': handle_autobiography,
    'list': handle_list,
    'goal_pursuit': handle_goal_pursuit,
    'document': handle_document,
    'read': handle_read,
    'find_tools': handle_find_tools,
    'goals': handle_goals,
    'rich_render': handle_rich_render,
    'review_tool_calls': handle_review_tool_calls,
}


def get_skill_handler(skill_name: str):
    """Get an innate skill handler by name. Returns None if not found."""
    return _SKILL_HANDLERS.get(skill_name)


def register_innate_skills(dispatcher) -> None:
    """
    Register all innate skills with the ACT dispatcher.

    Args:
        dispatcher: ActDispatcherService instance
    """
    # Register innate skills
    dispatcher.handlers["memory"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["introspect"] = lambda channel, action: handle_introspect(channel, action)
    dispatcher.handlers["schedule"] = lambda channel, action: handle_scheduler(channel, action)
    dispatcher.handlers["autobiography"] = lambda channel, action: handle_autobiography(channel, action)
    dispatcher.handlers["list"] = lambda channel, action: handle_list(channel, action)
    dispatcher.handlers["goal_pursuit"] = lambda channel, action: handle_goal_pursuit(channel, action)
    dispatcher.handlers["document"] = lambda channel, action: handle_document(channel, action)
    dispatcher.handlers["read"] = lambda channel, action: handle_read(channel, action)
    dispatcher.handlers["find_tools"] = lambda channel, action: handle_find_tools(channel, action)
    dispatcher.handlers["goals"] = lambda channel, action: handle_goals(channel, action)
    dispatcher.handlers["rich_render"] = lambda channel, action: handle_rich_render(channel, action)
    dispatcher.handlers["review_tool_calls"] = lambda channel, action: handle_review_tool_calls(channel, action)

    # Backward-compatibility aliases (old name -> unified memory handler)
    dispatcher.handlers["recall"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["memorize"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["memory_query"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["memory_write"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["world_state_read"] = lambda channel, action: handle_introspect(channel, action)
    dispatcher.handlers["internal_reasoning"] = lambda channel, action: handle_memory(channel, action)
    dispatcher.handlers["semantic_query"] = lambda channel, action: handle_memory(channel, action)

    # Register on_demand tools from the tool registry (dynamic plugins)
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        for tool_name in registry.get_on_demand_tools():
            # Innate skills take precedence over same-named external tools
            if tool_name in dispatcher.handlers:
                logging.debug(f"[INNATE SKILLS] Skipping dynamic tool '{tool_name}' — innate skill registered")
                continue
            # Strip "type" key before passing to registry — action dict
            # contains type for dispatcher routing, tools don't need it
            dispatcher.handlers[tool_name] = (
                lambda topic, action, tn=tool_name: registry.invoke(
                    tn, topic,
                    {k: v for k, v in action.items() if k not in ('type', 'exchange_id')},
                    exchange_id=action.get('exchange_id', ''),
                )
            )
    except Exception as e:
        logging.warning(f"[INNATE SKILLS] Tool registry failed to load: {e}")

