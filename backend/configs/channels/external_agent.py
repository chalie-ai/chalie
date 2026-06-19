from __future__ import annotations

from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig

from configs.channels._common import (
    DEFAULT_ALWAYS_AVAILABLE,
    substitute_provider_content_field,
)


class DiscloseToHumanHook(PostTurnHook):
    """EAMP after-turn: dispatch a disclosure turn to the user channel so the
    human learns about the external-agent exchange.  §3b / §4d / §4.8.

    The closure-over-constructor-args of the old factory becomes honest object
    fields.  Composed onto EAMPConfig only when ``loop_in_human`` is set.  The
    hidden-input UserConfig turn it spawns is the §4.0 cross-channel surface —
    a background, emit-only processor invisible to the foreground turn machinery.
    """

    def __init__(self, agent_name: str, project: str) -> None:
        self._agent_name = agent_name
        self._project = project

    def run(self, mp, response_text: str) -> None:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        raw_input = getattr(mp, "_raw_input", "")
        disclosure_input = (
            f"An external agent called '{self._agent_name}' just contacted you "
            f"about '{self._project}'. "
            f"Here's what they said:\n\n\"{raw_input}\"\n\n"
            f"You replied:\n\n\"{response_text}\"\n\n"
            "Let the user know about this exchange in your own words."
        )
        try:
            from api.chat import dispatch_message  # noqa: PLC0415
            dispatch_message(disclosure_input, source="external_agent", hidden_input=True)
        except Exception as exc:
            _log.warning("[EAMP] disclosure dispatch failed: %s", exc)


class EAMPConfig(ProcessorConfig):
    """External-Agent Message Processor config.

    channel='external-agent:{agent_name}', role='external_agent'.
    suppress_history=False (conversational), memory_seed=True.
    post_turn dispatches disclosure when loop_in_human (§3b).

    agent_name / project are captured on the instance (the prompt-builder
    methods read them via self).  wrapper_id is accepted for call-site
    compatibility but is not used by any prompt builder.
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
            policy_channel=ProcessorConfig.PolicyChannel.EXTERNAL_AGENT,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            broadcast_to=None,
            memory_seed=True,
            post_turn_hooks=(
                (DiscloseToHumanHook(agent_name, project),) if loop_in_human else ()
            ),
        )
        object.__setattr__(self, "_agent_name", agent_name)
        object.__setattr__(self, "_project", project)

    def get_user_definition(self, mp) -> str:
        """Static agent identity string.  §3b."""
        return (
            f"The user is {self._agent_name}, an external agent. "
            f"This conversation is about: {self._project}."
        )

    def get_system_prompt(self, mp) -> str:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        _agent_name = self._agent_name
        _project = self._project
        try:
            from services.system_message_prompt import ExternalAgentSystemMessagePrompt  # noqa: PLC0415
            body = ExternalAgentSystemMessagePrompt().get_prompt()
            body = substitute_provider_content_field(body, mp)

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

            user_def = self.get_user_definition(mp)
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

    def get_user_prompt(self, mp) -> str:
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
