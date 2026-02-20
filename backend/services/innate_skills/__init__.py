"""
Innate Skills Package — Built-in cognitive skills for the ACT loop.

Skills provide the ACT loop LLM with data, capabilities, and access to systems.
They are tools, not thoughts — the ACT prompt is the sole reasoner.

All innate skills are non-LLM operations (fast, sub-cortical services).
"""

import logging
from services.innate_skills.recall_skill import handle_recall
from services.innate_skills.memorize_skill import handle_memorize
from services.innate_skills.introspect_skill import handle_introspect
from services.innate_skills.associate_skill import handle_associate


def register_innate_skills(dispatcher) -> None:
    """
    Register all innate skills with the ACT dispatcher.

    Args:
        dispatcher: ActDispatcherService instance
    """
    # Register innate skills
    dispatcher.handlers["recall"] = lambda topic, action: handle_recall(topic, action)
    dispatcher.handlers["memorize"] = lambda topic, action: handle_memorize(topic, action)
    dispatcher.handlers["introspect"] = lambda topic, action: handle_introspect(topic, action)
    dispatcher.handlers["associate"] = lambda topic, action: handle_associate(topic, action)

    # Backward-compatibility aliases (old name -> new handler)
    dispatcher.handlers["memory_query"] = lambda topic, action: handle_recall(topic, action)
    dispatcher.handlers["memory_write"] = lambda topic, action: handle_memorize(topic, action)
    dispatcher.handlers["world_state_read"] = lambda topic, action: handle_introspect(topic, action)
    dispatcher.handlers["internal_reasoning"] = lambda topic, action: handle_recall(topic, action)
    dispatcher.handlers["semantic_query"] = lambda topic, action: handle_recall(topic, action)

    # Register on_demand tools from the tool registry (dynamic plugins)
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        for tool_name in registry.get_on_demand_tools():
            # Strip "type" key before passing to registry — action dict
            # contains type for dispatcher routing, tools don't need it
            dispatcher.handlers[tool_name] = (
                lambda topic, action, tn=tool_name: registry.invoke(
                    tn, topic, {k: v for k, v in action.items() if k != 'type'}
                )
            )
    except Exception as e:
        logging.warning(f"[INNATE SKILLS] Tool registry failed to load: {e}")

    # Add "reminder" alias for "scheduler" tool (issue #3 — backward compatibility)
    if "scheduler" in dispatcher.handlers:
        dispatcher.handlers["reminder"] = dispatcher.handlers["scheduler"]
