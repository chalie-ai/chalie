from __future__ import annotations

from typing import Any

from services.post_turn_hook import PostTurnHook
from services.processor_config import ProcessorConfig

from configs.channels._common import (
    DEFAULT_ALWAYS_AVAILABLE,
    substitute_provider_content_field,
)


class ProactiveSuggestionHook(PostTurnHook):
    """Fires on a turn that ran 4+ tool calls. Non-blocking (daemon thread inside"""

    def run(self, mp, response_text: str) -> None:
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        turn_id = getattr(mp, "turn_id", None)
        if turn_id is None:
            return
        try:
            from services.act_trail import ActTrail  # noqa: PLC0415
            rows = ActTrail().fetch_by_turn(mp.config.channel, turn_id)
            tool_call_count = sum(
                1 for r in rows if r.get("tool_name") != "chat_history_compactor"
            )
            if tool_call_count < 4:
                return
            from services.skill_suggestion_message_processor import maybe_suggest_skill  # noqa: PLC0415
            rendered = mp._render_act_trail()  # type: ignore[attr-defined]
            act_trail = rendered.split("\n") if rendered else []
            raw_input = getattr(mp, "_raw_input", "")
            maybe_suggest_skill(act_trail, raw_input, mp.config.channel, turn_id)
        except Exception as exc:
            _log.warning("[POSTTURN] skill suggestion failed: %s", exc)


class UserConfig(ProcessorConfig):
    """Attachments auto-fire document.upload on turn 0 (presence of"""

    SUPPORTS_ASYNC = True

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        _metadata = metadata or {}
        super().__init__(
            channel="user",
            role="user",
            policy_channel=ProcessorConfig.PolicyChannel.CHAT,
            always_available=DEFAULT_ALWAYS_AVAILABLE,
            skip_transcript=False,
            skip_input_row=bool(_metadata.get("hidden_input")),
            suppress_history=False,
            broadcast_to="user",
            memory_seed=True,
            post_turn_hooks=(ProactiveSuggestionHook(),),
        )

    def get_user_definition(self, mp) -> str:
        """Per-turn cached on mp._user_definition_cached so each ACT iteration"""
        _FALLBACK = "The user is a real human. Treat this conversation as peer-to-peer dialogue."
        cached = getattr(mp, "_user_definition_cached", None)
        if cached is not None:
            return cached

        try:
            from services.mode_gate_service import STEER_THRESHOLD  # noqa: PLC0415
            mode_state = getattr(mp, "_mode_state_cached", None)
            if mode_state is None:
                mode_state = {}
            prefer_long = mode_state.get("converse", 0.0) >= STEER_THRESHOLD
        except Exception:
            prefer_long = False

        try:
            from services.data_graph_service import get_data_graph_service  # noqa: PLC0415
            dgs = get_data_graph_service()
            rows = dgs.fetch(kinds=["system"], order_by="retrieval_weight DESC")
            by_key = {r.get("key"): r for r in rows if r.get("key")}

            preferred_key = "user_summary_long" if prefer_long else "user_summary"
            entry = by_key.get(preferred_key)
            if (not entry or not entry.get("value")) and prefer_long:
                entry = by_key.get("user_summary")
            if entry and entry.get("value"):
                result = entry["value"]
                mp._user_definition_cached = result  # type: ignore[attr-defined]
                return result
        except Exception:
            pass

        mp._user_definition_cached = _FALLBACK  # type: ignore[attr-defined]
        return _FALLBACK

    def get_system_prompt(self, mp) -> str:
        """Voice line sits at the very top for cache warmth. The"""
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        try:
            from services.personality.personality_service import personality_service  # noqa: PLC0415
            from services.system_message_prompt import UnifiedSystemMessagePrompt  # noqa: PLC0415
            template = UnifiedSystemMessagePrompt().get_prompt()
            voice_line = f"When responding; {personality_service.get_voice()}"
            prompt = f"{voice_line}\n\n{template}"
            prompt = substitute_provider_content_field(prompt, mp)
        except Exception as exc:
            _log.warning("[UMP] system prompt build failed: %s", exc)
            return ""

        # Mode-gate steering directives.
        try:
            mode_gate = getattr(mp, "_mode_gate_cached", None)
            if mode_gate is not None:
                additions = mode_gate.get_system_prompt_additions()
                if additions:
                    prompt = f"{prompt}\n\n{additions}"
        except Exception as exc:
            _log.debug("[UMP] mode-gate additions failed: %s", exc)

        return prompt

    def get_user_prompt(self, mp) -> str:
        """Section order: user_def, World State, Previous Messages, blank,"""
        import logging  # noqa: PLC0415
        _log = logging.getLogger(__name__)
        parts: list[str] = []

        # 1. User definition
        user_def = self.get_user_definition(mp)
        if user_def:
            parts.append(user_def)

        # 2. World State
        try:
            from services.world_state import world_state  # noqa: PLC0415
            rendered_ws = world_state.render()
            if rendered_ws:
                _log.info(
                    "[WorldState] injected rendered block into user prompt (%d chars)",
                    len(rendered_ws),
                )
                parts.append(rendered_ws)
        except Exception as exc:
            _log.debug("[UMP] world_state.render failed: %s", exc)

        # 3. Previous Messages
        try:
            prev = mp.get_previous_messages()  # type: ignore[attr-defined]
            if prev:
                parts.append(f"## Previous Messages\n{prev}")
        except Exception as exc:
            _log.debug("[UMP] get_previous_messages failed: %s", exc)

        # Blank separator
        parts.append("")

        # 3b. Post-compaction continuity banner — only on the continuation MP
        #     spawned right after a mid-turn compaction. The collapse cost the
        #     model its working context, so the current-state block opens by
        #     restating the user's request and pointing at the Checkpoint section
        #     (the compacted summary, prepended by the framework envelope) and the
        #     transcript-reading tool, mirroring how a fresh post-compaction agent
        #     recovers continuity. Sits ABOVE the input line.
        if getattr(mp, "post_compaction_continuation", False):
            query = getattr(mp, "continuation_user_query", None) or mp._raw_input  # type: ignore[attr-defined]
            parts.append(
                "You are continuing after a mid-turn compaction. "
                f"The user query was: {query}. "
                "Read the Checkpoint section above to recover what you were "
                "working on, and use the review_transcript tool to read the "
                "previous turns of this conversation."
            )

        # 4. Input line with optional nudge — BEFORE the trail (OLD ordering).
        nudge_tag = (getattr(mp, "_metadata", None) or {}).get("nudge_tag") or ""
        turn_line = f"user: {mp._raw_input}"  # type: ignore[attr-defined]
        if nudge_tag:
            turn_line += " " + nudge_tag
        parts.append(turn_line)

        # 5. ACT loop trail (empty before any tools have run; carries the turn-0
        #    memory seed once it has fired).
        try:
            trail = mp._render_act_trail()  # type: ignore[attr-defined]
            if trail:
                parts.append(trail)
        except Exception as exc:
            _log.debug("[UMP] _render_act_trail failed: %s", exc)

        return "\n".join(parts)
