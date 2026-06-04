from __future__ import annotations

from typing import Any

from services.processor_config import ProcessorConfig

from configs.channels._common import (
    DEFAULT_ALWAYS_AVAILABLE,
    DEFAULT_DISCOVERABLE,
    DELEGATE_INTERNAL_TOOLS,
    PATTERN_WRITE_TOOLS,
    substitute_provider_content_field,
)

# ── EAMP prompt builders ──────────────────────────────────────────────────────

def _eamp_build_user_definition(agent_name: str, project: str) -> Any:
    """Return a builder callable for the EAMP user definition.

    Returns a zero-arg-relative callable that, when called with mp, returns
    the static agent identity string.  §3b.
    """
    _text = (
        f"The user is {agent_name}, an external agent. "
        f"This conversation is about: {project}."
    )

    def _build(_mp: object) -> str:
        return _text

    return _build


def _eamp_build_system_prompt(agent_name: str, project: str, wrapper_id: str) -> Any:
    """Return a builder callable for the EAMP system prompt.

    Fills in {user_name}, {agent_name}, {project_or_task_name} template
    variables from data_graph and the EAMP constructor args.  §3b.
    """
    _agent_name = agent_name
    _project = project

    def _build(_mp: object) -> str:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            from services.system_message_prompt import ExternalAgentSystemMessagePrompt  # noqa: PLC0415
            body = ExternalAgentSystemMessagePrompt().get_prompt()
            body = substitute_provider_content_field(body, "external_agent")

            # Resolve the user's first name from data_graph.
            user_name = "the user"
            try:
                from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
                dgs = get_data_graph_service()
                rows = dgs.fetch(kinds=["system"])
                for row in rows:
                    key = row.get("key") if isinstance(row, dict) else getattr(row, "key", None)
                    val = row.get("value") if isinstance(row, dict) else getattr(row, "value", None)
                    if key == "user_summary" and val:
                        first_word = val.split()[0] if val else ""
                        if first_word:
                            user_name = first_word
                        break
            except Exception:
                pass

            user_def = (
                f"The user is {_agent_name}, an external agent. "
                f"This conversation is about: {_project}."
            )
            body = (
                body
                .replace("{user_name}", user_name)
                .replace("{agent_name}", _agent_name)
                .replace("{project_or_task_name}", _project)
            )
            return f"{user_def}\n\n{body}"
        except Exception as exc:
            _log.warning("[EAMP] system prompt build failed: %s", exc)
            return ""

    return _build


def _eamp_build_user_prompt(mp: object) -> str:
    """EAMP user-message body for one ACT iteration.

    Stripped compared to UMP: no world state, no user definition (it lives
    in the system prompt for EAMP).  Keeps: previous messages, ACT trail,
    input line.  §3b.
    """
    import logging  # noqa: PLC0415
    _log = logging.getLogger(__name__)
    parts: list[str] = []

    # Previous Messages
    try:
        prev = mp.get_previous_messages()  # type: ignore[attr-defined]
        if prev:
            parts.append(f"## Previous Messages\n{prev}")
    except Exception as exc:
        _log.debug("[EAMP] get_previous_messages failed: %s", exc)

    parts.append("")

    # Input line — BEFORE the trail (OLD get_user_prompt ordering).
    parts.append(f"user: {mp._raw_input}")  # type: ignore[attr-defined]

    # ACT loop trail (carries the turn-0 memory seed once it has fired).
    try:
        trail = mp._render_act_trail()  # type: ignore[attr-defined]
        if trail:
            parts.append(trail)
    except Exception as exc:
        _log.debug("[EAMP] _render_act_trail failed: %s", exc)

    return "\n".join(parts)


def _make_eamp_post_turn(
    agent_name: str,
    project: str,
    loop_in_human: bool,
) -> Any:
    """Return a post_turn callable for EAMP.

    When loop_in_human=True, dispatches a disclosure message to the user
    channel after the ACT loop completes.  §3b / §4d.
    """
    if not loop_in_human:
        return None

    _agent_name = agent_name
    _project = project

    def _post_turn(mp: object, response_text: str) -> None:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        raw_input = getattr(mp, "_raw_input", "")
        disclosure_input = (
            f"An external agent called '{_agent_name}' just contacted you "
            f"about '{_project}'. "
            f"Here's what they said:\n\n\"{raw_input}\"\n\n"
            f"You replied:\n\n\"{response_text}\"\n\n"
            "Let the user know about this exchange in your own words."
        )
        try:
            from api.chat import dispatch_message  # noqa: PLC0415
            dispatch_message(disclosure_input, source="external_agent", hidden_input=True)
        except Exception as exc:
            _log.warning("[EAMP] disclosure dispatch failed: %s", exc)

    return _post_turn


class EAMPConfig(ProcessorConfig):
    """External-Agent Message Processor config.

    channel='external-agent:{agent_name}', role='external_agent'.
    suppress_history=False (conversational), memory_seed=True.
    post_turn dispatches disclosure when loop_in_human (§3b).
    """

    def __init__(
        self,
        agent_name: str,
        project: str,
        loop_in_human: bool,
        wrapper_id: str,
    ) -> None:
        super().__init__(
            channel=f"external-agent:{agent_name}",
            role="external_agent",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.EXTERNAL_AGENT,
            build_user_prompt=_eamp_build_user_prompt,
            build_user_definition=_eamp_build_user_definition(agent_name, project),
            build_system_prompt=_eamp_build_system_prompt(agent_name, project, wrapper_id),
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            discoverable=DEFAULT_DISCOVERABLE,
            blocked=PATTERN_WRITE_TOOLS | DELEGATE_INTERNAL_TOOLS,
            max_iterations=200,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            broadcast_to=None,
            memory_seed=True,
            post_turn=_make_eamp_post_turn(agent_name, project, loop_in_human),
        )
