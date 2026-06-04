# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Foreground-channel configs — observable runtime behaviour only.

Each foreground channel is now a ``ProcessorConfig`` rather than a
``MessageProcessor`` subclass.  These tests cover the behaviours those config
fields actually produce at runtime:

  * EAMP ``post_turn`` dispatches a disclosure turn iff ``loop_in_human``.
  * ``_emit`` broadcasts iff ``broadcast_to`` is set (the WS gate).
  * UMP ``post_turn`` fires skill suggestion only on a clean, multi-iteration exit.
  * The flat user path injects the thinking-exploration block into the prompt.

They deliberately do NOT re-enumerate ``config.field == constant`` tables, assert
payload shapes, or restate "function X does not call Y" guards.  Narration trail
rows / WS payloads and tool-event payloads are owned by
``test_act_trail_query.py`` and ``test_ability_dispatch.py`` respectively.
"""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ── helpers ───────────────────────────────────────────────────────────────────

def _post_turn_mp(**attrs):
    """Single shared mp stand-in for ``post_turn`` callables.

    Replaces the per-class ``_make_mp`` / ``_make_mp_flat`` copies.  ``post_turn``
    only reads flat attributes (``_raw_input``, ``loop_exited_cleanly``,
    ``current_iteration``) and calls ``_render_act_trail()`` (§4c) — no real
    loop object is needed.
    """
    mp = MagicMock(spec=object)
    mp._render_act_trail = lambda: "trail-line-1"
    for key, value in attrs.items():
        setattr(mp, key, value)
    return mp


def _make_config(**overrides):
    """Build a ProcessorConfig with inert defaults; override only what matters.

    Single shared config factory — replaces the per-test inline ProcessorConfig
    builds.
    """
    from services.processor_config import ProcessorConfig
    from tests.helpers import StubProcessorConfig

    defaults = dict(
        channel="user",
        role="user",
        policy_channel=ProcessorConfig.POLICY_CHANNEL.CHAT,
        build_user_prompt=lambda _mp: "",
        build_user_definition=lambda _mp: "",
        build_system_prompt=lambda _mp: "",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=None,
        skip_transcript=True,
        skip_input_row=True,
        suppress_history=True,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )
    defaults.update(overrides)
    return StubProcessorConfig(**defaults)


# ── B3: EAMP disclosure fires only when loop_in_human (§3b) ────────────────────

class TestEampDisclosure:
    """loop_in_human gates whether post_turn dispatches a user-facing disclosure."""

    def test_disclosure_dispatched_when_true(self):
        """loop_in_human=True → post_turn dispatches a hidden external-agent turn."""
        from configs.channels import EAMPConfig
        cfg = EAMPConfig(
            agent_name="bot", project="proj", loop_in_human=True, wrapper_id="",
        )
        mp = _post_turn_mp(_raw_input="hello from agent")
        with patch("api.chat.dispatch_message") as mock_dispatch:
            cfg.post_turn(mp, "agent-response-text")

        mock_dispatch.assert_called_once()
        call = mock_dispatch.call_args
        assert call.kwargs.get("source") == "external_agent"
        assert call.kwargs.get("hidden_input") is True

    def test_disclosure_text_includes_agent_name_and_project(self):
        """The dispatched disclosure body names the calling agent and project (§3b)."""
        from configs.channels import EAMPConfig
        cfg = EAMPConfig(
            agent_name="mybot", project="my-project", loop_in_human=True, wrapper_id="",
        )
        mp = _post_turn_mp(_raw_input="hello from agent")
        captured = []
        with patch("api.chat.dispatch_message",
                   side_effect=lambda text, **kw: captured.append(text)):
            cfg.post_turn(mp, "agent replied this")

        assert len(captured) == 1
        assert "mybot" in captured[0]
        assert "my-project" in captured[0]


# ── N1/N2: the WS broadcast gate (§5 / AC-28) ──────────────────────────────────

class TestWsEmitGate:
    """_emit broadcasts the event iff config.broadcast_to is set; no-op otherwise."""

    def test_noop_when_broadcast_to_none(self):
        """broadcast_to=None → no WebSocketBroker is even constructed (N1)."""
        from abilities._base import _emit

        config = MagicMock()
        config.broadcast_to = None
        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            _emit(config, {"type": "act_narration", "text": "hello", "step": 0})

        mock_broker_cls.assert_not_called()

    def test_broadcasts_event_when_broadcast_to_set(self):
        """broadcast_to set → broker.broadcast is called with the exact event (N2)."""
        from abilities._base import _emit

        config = MagicMock()
        config.broadcast_to = "user"
        event = {"type": "act_narration", "text": "thinking...", "step": 1}
        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            broker = MagicMock()
            mock_broker_cls.return_value = broker
            _emit(config, event)

        broker.broadcast.assert_called_once_with(event)


# ── C6: UMP post_turn fires skill suggestion only on a clean, deep exit ─────────

class TestUmpPostTurnSkillSuggestion:
    """UMP post_turn fires skill suggestion only when the loop exited cleanly
    after 4+ iterations — and does nothing else (§4e / §6)."""

    def test_fired_on_clean_exit_with_enough_iterations(self):
        """clean exit + iteration>=4 → skill suggestion fires."""
        from configs.channels import UserConfig
        cfg = UserConfig()
        mp = _post_turn_mp(loop_exited_cleanly=True, current_iteration=4,
                           _raw_input="test input")
        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "assistant response text")

        mock_suggest.assert_called_once()

    def test_not_fired_when_loop_did_not_exit_cleanly(self):
        """loop_exited_cleanly=False → no skill suggestion, even with many iterations."""
        from configs.channels import UserConfig
        cfg = UserConfig()
        mp = _post_turn_mp(loop_exited_cleanly=False, current_iteration=10,
                           _raw_input="test input")
        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "response")

        mock_suggest.assert_not_called()

    def test_not_fired_when_too_few_iterations(self):
        """iteration < 4 → no skill suggestion, even on a clean exit."""
        from configs.channels import UserConfig
        cfg = UserConfig()
        mp = _post_turn_mp(loop_exited_cleanly=True, current_iteration=2,
                           _raw_input="test input")
        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "response")

        mock_suggest.assert_not_called()


# ── Per-provider content-field placeholder is resolved, not emitted literally ──

class TestContentFieldPlaceholderSubstitution:
    """The user system prompt must reference the active provider's real response
    field, never the literal ``{{provider_content_field_name}}`` template token
    (regression: the ACT-loop refactor dropped the substitution from the user
    channel, so the model was told to write into a non-existent field)."""

    def test_user_system_prompt_resolves_provider_content_field(self):
        """UserConfig.get_system_prompt swaps the placeholder for the provider's
        CONTENT_FIELD_LABEL."""
        from configs.channels import UserConfig

        provider = MagicMock()
        provider.CONTENT_FIELD_LABEL = "content[].text"
        providers_singleton = MagicMock()
        providers_singleton._resolve.return_value = provider

        with patch("services.providers.Providers") as mock_providers_cls:
            mock_providers_cls.instance.return_value = providers_singleton
            cfg = UserConfig()
            object.__setattr__(cfg, "mp", MagicMock(spec=object))
            body = cfg.get_system_prompt()

        assert "{{provider_content_field_name}}" not in body
        assert "content[].text" in body

    def test_passthrough_when_provider_lookup_fails(self):
        """A provider-resolution failure leaves the body intact (best-effort)."""
        from configs.channels._common import substitute_provider_content_field

        with patch("services.providers.Providers") as mock_providers_cls:
            mock_providers_cls.instance.side_effect = RuntimeError("no provider")
            out = substitute_provider_content_field(
                "In the {{provider_content_field_name}} field ...", "user"
            )

        assert out == "In the {{provider_content_field_name}} field ..."


# ── M7: thinking-exploration reaches the user prompt on the flat path ──────────

class TestThinkingGateExplorationReachesPrompt:
    """flat user path — a high-deliberation exploration block actually lands in
    the user message sent to the provider (§4 / §6)."""

    def test_exploration_injected_into_user_prompt_on_high_deliberation(self):
        """High deliberation scalar → exploration text appears in the built prompt."""
        from services.message_processor import MessageProcessor
        from services.deliberation_score_service import DeliberationScoreService
        from services.deliberation_ema_service import DeliberationEmaService

        EXPLORATION_TEXT = "My chain of thought: I should check the calendar first."

        def _build_user_prompt(mp):
            exploration = getattr(mp, "thinking_exploration", None)
            if exploration:
                return (
                    "## Chain of Thought\n"
                    "Below is your initial reaction to this prompt, played back. "
                    "Use it as grounding but pivot as needed based on the conversation."
                    "\n\n---\n\n"
                    f"{exploration}\n\n---"
                    f"\nuser: {mp._raw_input}"
                )
            return f"user: {mp._raw_input}"

        config = _make_config(build_user_prompt=_build_user_prompt)

        mock_score_svc = MagicMock(spec_set=DeliberationScoreService)
        mock_score_svc.classify.return_value = 0.92  # high scalar
        mock_ema_svc = MagicMock(spec_set=DeliberationEmaService)
        mock_ema_svc.update_and_bucket.return_value = (0.90, "high")

        with patch(
            "services.deliberation_score_service.DeliberationScoreService",
            return_value=mock_score_svc,
        ), patch(
            "services.deliberation_ema_service.DeliberationEmaService",
            return_value=mock_ema_svc,
        ), patch.object(
            MessageProcessor,
            "_run_thinking_exploration",
            return_value=EXPLORATION_TEXT,
        ), patch(
            "abilities._registry.AbilityRegistry.build_tools",
            return_value=[],
        ), patch(
            "services.providers.Providers",
        ) as mock_providers_cls:
            fake_response = MagicMock()
            fake_response.text = "assistant reply"
            fake_response.tool_calls = []

            mock_provider = MagicMock()
            mock_provider.send_messages.return_value = fake_response
            mock_provider.calculate.return_value = 0.1
            mock_providers_cls.instance.return_value = mock_provider

            result = MessageProcessor.process("schedule a meeting", config)

        assert mock_provider.send_messages.called
        call_args = mock_provider.send_messages.call_args
        messages = call_args.args[1] if call_args.args else call_args[1]
        user_content = messages[0]["content"]
        assert EXPLORATION_TEXT in user_content, (
            f"Exploration text must appear in the user prompt on the flat path. "
            f"Got: {user_content!r}"
        )
        assert result == "assistant reply"
