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
from services.innate_skills.recall_skill import handle_recall
from services.innate_skills.memorize_skill import handle_memorize
from services.innate_skills.introspect_skill import handle_introspect
from services.innate_skills.associate_skill import handle_associate
from services.innate_skills.scheduler_skill import handle_scheduler
from services.innate_skills.autobiography_skill import handle_autobiography
from services.innate_skills.focus_skill import handle_focus
from services.innate_skills.list_skill import handle_list
from services.innate_skills.moment_skill import handle_moment
from services.innate_skills.persistent_task_skill import handle_persistent_task
from services.innate_skills.emit_card_skill import handle_emit_card
from services.innate_skills.document_skill import handle_document


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
    dispatcher.handlers["schedule"] = lambda topic, action: handle_scheduler(topic, action)
    dispatcher.handlers["autobiography"] = lambda topic, action: handle_autobiography(topic, action)
    dispatcher.handlers["focus"] = lambda topic, action: handle_focus(topic, action)
    dispatcher.handlers["list"] = lambda topic, action: handle_list(topic, action)
    dispatcher.handlers["moment"] = lambda topic, action: handle_moment(topic, action)
    dispatcher.handlers["persistent_task"] = lambda topic, action: handle_persistent_task(topic, action)
    dispatcher.handlers["emit_card"] = lambda topic, action: handle_emit_card(topic, action)
    dispatcher.handlers["document"] = lambda topic, action: handle_document(topic, action)

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
