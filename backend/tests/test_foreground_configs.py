# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
§9a B1-B3, B13, N1-N5, C6 — Foreground configs (UMP/EAMP) and WS gate.

Tests authored BLIND from spec §1-8 only.  Per the GOVERNING TEST PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the code is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.

T7 covers:
  B1  — UMP config shape: channel/role/max_iterations/suppress_history/
          broadcast_to/memory_seed/post_turn.
  B2  — EAMP config shape: channel/role/max_iterations/suppress_history/
          memory_seed/post_turn.
  B3  — EAMP disclosure fires only when loop_in_human=True.
  B13 — Only UMP and EAMP have suppress_history=False; all others True.
  N1  — WS.emit (via _emit) is a no-op when broadcast_to=None.
  N2  — WS.emit broadcasts when broadcast_to is set.
  N3  — Narration payload shape: {type:'act_narration', text:<sanitized>, step}.
  N4  — Tool start/end payload shapes (act_tool_start with summary; act_tool_end with ok).
  N5  — Background loops (broadcast_to=None) emit nothing at all.
  C6  — UMP post_turn does ONLY skill suggestion (no phase update, no metrics).
"""

from unittest.mock import MagicMock, patch, call
import threading

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# B1. UMP config shape                                                   (§3b)
# ---------------------------------------------------------------------------


class TestUmpConfig:
    """B1: make_user_config returns the correct ProcessorConfig."""

    def test_channel_and_role(self):
        """channel='user', role='user'."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.channel == "user"
        assert cfg.role == "user"

    def test_max_iterations_is_none(self):
        """max_iterations=None — UMP runs an unbounded loop (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.max_iterations is None

    def test_suppress_history_false(self):
        """suppress_history=False — UMP is a conversational channel (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.suppress_history is False

    def test_broadcast_to_user(self):
        """broadcast_to='user' — live output enabled for UMP (§3b / AC-28)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.broadcast_to == "user"

    def test_memory_seed_true(self):
        """memory_seed=True — UMP seeds memory recall on turn 0 (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.memory_seed is True

    def test_post_turn_is_callable(self):
        """post_turn is a callable (skill suggestion hook) (§3b / §4e)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert callable(cfg.post_turn)

    def test_skip_input_row_false_by_default(self):
        """skip_input_row=False when metadata has no hidden_input (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config(metadata={})
        assert cfg.skip_input_row is False

    def test_skip_input_row_true_for_hidden_input(self):
        """skip_input_row=True when metadata['hidden_input']=True (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config(metadata={"hidden_input": True})
        assert cfg.skip_input_row is True

    def test_usage_class_is_chat(self):
        """usage_class='chat' (§3b)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        assert cfg.usage_class == "chat"


# ---------------------------------------------------------------------------
# B2. EAMP config shape                                                  (§3b)
# ---------------------------------------------------------------------------


class TestEampConfig:
    """B2: make_eamp_config returns the correct ProcessorConfig."""

    def _make(self, loop_in_human=False):
        from configs.channels import make_eamp_config
        return make_eamp_config(
            agent_name="testbot",
            project="test-project",
            loop_in_human=loop_in_human,
            wrapper_id="w1",
        )

    def test_channel_contains_agent_name(self):
        """channel starts with 'external-agent:' and includes agent_name (§3b)."""
        cfg = self._make()
        assert "external-agent:" in cfg.channel
        assert "testbot" in cfg.channel

    def test_role_is_external_agent(self):
        """role='external_agent' (§3b)."""
        cfg = self._make()
        assert cfg.role == "external_agent"

    def test_max_iterations_is_200(self):
        """max_iterations=200 (§3b)."""
        cfg = self._make()
        assert cfg.max_iterations == 200

    def test_suppress_history_false(self):
        """suppress_history=False — EAMP is a conversational channel (§3b)."""
        cfg = self._make()
        assert cfg.suppress_history is False

    def test_memory_seed_true(self):
        """memory_seed=True — EAMP seeds memory recall on turn 0 (§3b)."""
        cfg = self._make()
        assert cfg.memory_seed is True

    def test_broadcast_to_none(self):
        """broadcast_to=None — EAMP runs silently, no WS streaming (§3b)."""
        cfg = self._make()
        assert cfg.broadcast_to is None

    def test_post_turn_none_when_no_disclosure(self):
        """post_turn=None when loop_in_human=False (§3b)."""
        cfg = self._make(loop_in_human=False)
        assert cfg.post_turn is None

    def test_post_turn_callable_when_disclosure(self):
        """post_turn is a callable when loop_in_human=True (§3b)."""
        cfg = self._make(loop_in_human=True)
        assert callable(cfg.post_turn)

    def test_usage_class_is_external_agent(self):
        """usage_class='external_agent' (§3b)."""
        cfg = self._make()
        assert cfg.usage_class == "external_agent"

    def test_wrapper_id_in_channel(self):
        """wrapper_id is included in the channel when non-empty (§3b)."""
        from configs.channels import make_eamp_config
        cfg = make_eamp_config(
            agent_name="bot",
            project="proj",
            loop_in_human=False,
            wrapper_id="wrap1",
        )
        assert "wrap1" in cfg.channel


# ---------------------------------------------------------------------------
# B3. EAMP disclosure fires only when loop_in_human                     (§3b)
# ---------------------------------------------------------------------------


class TestEampDisclosure:
    """B3: post_turn dispatches disclosure iff loop_in_human=True."""

    def _make_mp(self):
        """Return a minimal mp-like object usable by post_turn."""
        mp = MagicMock()
        mp._raw_input = "hello from agent"
        return mp

    def test_disclosure_not_dispatched_when_false(self):
        """loop_in_human=False → post_turn=None → no dispatch call (§3b)."""
        from configs.channels import make_eamp_config
        cfg = make_eamp_config(
            agent_name="bot",
            project="proj",
            loop_in_human=False,
            wrapper_id="",
        )
        assert cfg.post_turn is None

    def test_disclosure_dispatched_when_true(self):
        """loop_in_human=True → post_turn callable dispatches disclosure (§3b)."""
        from configs.channels import make_eamp_config
        cfg = make_eamp_config(
            agent_name="bot",
            project="proj",
            loop_in_human=True,
            wrapper_id="",
        )
        assert callable(cfg.post_turn)
        mp = self._make_mp()
        with patch("api.chat.dispatch_message") as mock_dispatch:
            cfg.post_turn(mp, "agent-response-text")
        mock_dispatch.assert_called_once()
        call_kwargs = mock_dispatch.call_args
        # Verify the source and hidden_input kwargs
        assert call_kwargs.kwargs.get("source") == "external_agent" or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "external_agent"
        ) or "external_agent" in str(call_kwargs)
        assert call_kwargs.kwargs.get("hidden_input") is True

    def test_disclosure_text_includes_agent_name(self):
        """Disclosure message includes the agent name and project (§3b)."""
        from configs.channels import make_eamp_config
        cfg = make_eamp_config(
            agent_name="mybot",
            project="my-project",
            loop_in_human=True,
            wrapper_id="",
        )
        mp = self._make_mp()
        captured_text = []
        with patch("api.chat.dispatch_message", side_effect=lambda t, **kw: captured_text.append(t)):
            cfg.post_turn(mp, "agent replied this")
        assert len(captured_text) == 1
        assert "mybot" in captured_text[0]
        assert "my-project" in captured_text[0]


# ---------------------------------------------------------------------------
# B13. Only UMP + EAMP have suppress_history=False                       (§2)
# ---------------------------------------------------------------------------


class TestSuppressHistoryBoundary:
    """B13: exactly UMP + EAMP keep history; all others suppress it."""

    def test_ump_suppress_history_false(self):
        from configs.channels import make_user_config
        assert make_user_config().suppress_history is False

    def test_eamp_suppress_history_false(self):
        from configs.channels import make_eamp_config
        cfg = make_eamp_config(
            agent_name="bot", project="p", loop_in_human=False, wrapper_id=""
        )
        assert cfg.suppress_history is False

    def test_dmn_suppress_history_true(self):
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.suppress_history is True

    def test_episode_encoder_suppress_history_true(self):
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.suppress_history is True

    def test_skill_suggestion_suppress_history_true(self):
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.suppress_history is True

    def test_compaction_suppress_history_true(self):
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.suppress_history is True

    def test_subagent_compaction_suppress_history_true(self):
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        assert SUBAGENT_COMPACTION_CONFIG.suppress_history is True

    def test_pattern_config_suppress_history_true(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(window_start=0, window_end=10)
        assert cfg.suppress_history is True

    def test_geo_config_suppress_history_true(self):
        from configs.channels import make_geo_config
        cfg = make_geo_config(window_start=0, window_end=10)
        assert cfg.suppress_history is True

    def test_user_summary_suppress_history_true(self):
        from configs.channels import make_user_summary_config
        assert make_user_summary_config().suppress_history is True

    def test_super_episode_suppress_history_true(self):
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config(channel="user", sources=[], spans=[])
        assert cfg.suppress_history is True


# ---------------------------------------------------------------------------
# N1. WS.emit is a no-op when broadcast_to=None                          (§2)
# ---------------------------------------------------------------------------


class TestWsEmitNoOp:
    """N1: _emit is a no-op when config.broadcast_to is None."""

    def test_emit_noop_broadcast_to_none(self):
        """No broadcast call when broadcast_to=None (N1)."""
        from abilities._base import _emit

        mock_config = MagicMock()
        mock_config.broadcast_to = None

        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            _emit(mock_config, {"type": "act_narration", "text": "hello", "step": 0})

        mock_broker_cls.assert_not_called()

    def test_emit_noop_config_none(self):
        """No broadcast call when config itself is None (N1)."""
        from abilities._base import _emit

        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            _emit(None, {"type": "act_tool_start", "name": "memory", "id": "x"})

        mock_broker_cls.assert_not_called()


# ---------------------------------------------------------------------------
# N2. WS.emit broadcasts when broadcast_to is set                        (§2)
# ---------------------------------------------------------------------------


class TestWsEmitBroadcasts:
    """N2: _emit calls WebSocketBroker().broadcast() when broadcast_to is set."""

    def test_emit_broadcasts_when_broadcast_to_set(self):
        """broadcast_to='user' → broker.broadcast called with the event (N2)."""
        from abilities._base import _emit

        mock_config = MagicMock()
        mock_config.broadcast_to = "user"
        event = {"type": "act_narration", "text": "thinking...", "step": 1}

        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            mock_broker = MagicMock()
            mock_broker_cls.return_value = mock_broker
            _emit(mock_config, event)

        mock_broker.broadcast.assert_called_once_with(event)

    def test_emit_broadcasts_any_truthy_broadcast_to(self):
        """Any truthy broadcast_to value triggers the broadcast (N2)."""
        from abilities._base import _emit

        mock_config = MagicMock()
        mock_config.broadcast_to = "some_channel"
        event = {"type": "act_tool_end", "name": "memory", "id": "abc", "ok": True}

        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            mock_broker = MagicMock()
            mock_broker_cls.return_value = mock_broker
            _emit(mock_config, event)

        mock_broker.broadcast.assert_called_once_with(event)


# ---------------------------------------------------------------------------
# N3. Narration payload shape                                             (§4)
# ---------------------------------------------------------------------------


class TestNarrationPayloadShape:
    """N3: act_narration event has {type, text, step}."""

    def test_narration_payload_keys(self):
        """_record_narration emits {type:'act_narration', text:<sanitized>, step:int} (N3)."""
        from services.message_processor import MessageProcessor

        # Build a minimal flat-path mp with broadcast_to set.
        from services.processor_config import ProcessorConfig
        config = ProcessorConfig(
            channel="user",
            role="user",
            usage_class="chat",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=None,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to="user",
            memory_seed=False,
            post_turn=None,
        )

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "test", {})
        mp.config = config
        mp.uid = None
        mp.current_iteration = 2
        mp.cancel_event = threading.Event()
        mp.discovered_tools = []
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.deadline = None

        captured = []

        class _FakeResponse:
            text = "I am thinking about this"

        with patch("abilities._base._emit") as mock_emit:
            with patch.object(type(mp), '_record_narration', MessageProcessor._record_narration):
                mp._record_narration(_FakeResponse())

        # Verify _emit was called with the correct payload structure.
        assert mock_emit.called
        call_args = mock_emit.call_args
        event = call_args.args[1] if call_args.args else call_args[0][1]
        assert event["type"] == "act_narration"
        assert "text" in event
        assert "step" in event
        assert isinstance(event["step"], int)

    def test_narration_step_matches_current_iteration(self):
        """step field equals mp.current_iteration at emission time (N3)."""
        from services.message_processor import MessageProcessor
        from services.processor_config import ProcessorConfig

        config = ProcessorConfig(
            channel="user",
            role="user",
            usage_class="chat",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=None,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to="user",
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "test", {})
        mp.config = config
        mp.uid = None
        mp.current_iteration = 5
        mp.cancel_event = threading.Event()
        mp.discovered_tools = []
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.deadline = None

        class _FakeResponse:
            text = "narration text"

        with patch("abilities._base._emit") as mock_emit:
            mp._record_narration(_FakeResponse())

        call_args = mock_emit.call_args
        event = call_args.args[1] if call_args.args else call_args[0][1]
        assert event["step"] == 5


# ---------------------------------------------------------------------------
# N4. Tool start/end payload shapes                                       (§5)
# ---------------------------------------------------------------------------


class TestToolEventPayloads:
    """N4: act_tool_start has {type, name, id, summary}; act_tool_end has {type, name, id, ok}."""

    def _make_mp(self, broadcast_to="user"):
        from services.message_processor import MessageProcessor
        from services.processor_config import ProcessorConfig

        config = ProcessorConfig(
            channel="user",
            role="user",
            usage_class="chat",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=None,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=broadcast_to,
            memory_seed=False,
            post_turn=None,
        )
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "test", {})
        mp.config = config
        mp.uid = 1
        mp.current_iteration = 0
        mp.cancel_event = threading.Event()
        mp.discovered_tools = []
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.deadline = None
        return mp

    def test_tool_start_payload_shape(self):
        """act_tool_start event contains type, name, id, summary (N4)."""
        from abilities._base import Ability

        mp = self._make_mp()
        emitted = []

        with patch("abilities._base._emit", side_effect=lambda cfg, ev: emitted.append(ev)):
            with patch("abilities._base.AbilityRegistry.get") as mock_get:
                mock_ability = MagicMock()
                mock_ability.ASYNC_CAPABLE = False
                mock_ability.TIMEOUT = 5
                mock_ability.run.return_value = {"status": "success", "result": "ok"}
                mock_get.return_value = mock_ability
                with patch("abilities._base.PolicyService.enforce", return_value=None):
                    with patch("abilities._base.Ability.record"):
                        Ability.dispatch(mp, "some_tool", {"act_summary": "doing X"}, "call-id-1")

        start_events = [e for e in emitted if e.get("type") == "act_tool_start"]
        assert len(start_events) >= 1
        ev = start_events[0]
        assert ev["type"] == "act_tool_start"
        assert "name" in ev
        assert "id" in ev
        assert "summary" in ev

    def test_tool_end_payload_shape(self):
        """act_tool_end event contains type, name, id, ok (N4)."""
        from abilities._base import Ability

        mp = self._make_mp()
        emitted = []

        with patch("abilities._base._emit", side_effect=lambda cfg, ev: emitted.append(ev)):
            with patch("abilities._base.AbilityRegistry.get") as mock_get:
                mock_ability = MagicMock()
                mock_ability.ASYNC_CAPABLE = False
                mock_ability.TIMEOUT = 5
                mock_ability.run.return_value = {"status": "success", "result": "ok"}
                mock_get.return_value = mock_ability
                with patch("abilities._base.PolicyService.enforce", return_value=None):
                    with patch("abilities._base.Ability.record"):
                        Ability.dispatch(mp, "some_tool", {}, "call-id-2")

        end_events = [e for e in emitted if e.get("type") == "act_tool_end"]
        assert len(end_events) >= 1
        ev = end_events[0]
        assert ev["type"] == "act_tool_end"
        assert "name" in ev
        assert "id" in ev
        assert "ok" in ev
        assert isinstance(ev["ok"], bool)


# ---------------------------------------------------------------------------
# N5. Background loops emit nothing                                  (§2 / §6)
# ---------------------------------------------------------------------------


class TestBackgroundLoopsEmitNothing:
    """N5: configs with broadcast_to=None never emit WS events."""

    def test_dmn_config_has_no_broadcast(self):
        """DMN_CONFIG.broadcast_to is None — background loop (N5)."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.broadcast_to is None

    def test_compaction_config_has_no_broadcast(self):
        """COMPACTION_CONFIG.broadcast_to is None — background loop (N5)."""
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.broadcast_to is None

    def test_skill_suggestion_has_no_broadcast(self):
        """SKILL_SUGGESTION_CONFIG.broadcast_to is None — background loop (N5)."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.broadcast_to is None

    def test_emit_noop_for_background_config(self):
        """_emit is a no-op when the config has broadcast_to=None (N5 / N1)."""
        from abilities._base import _emit
        from configs.channels import DMN_CONFIG

        with patch("abilities._base.WebSocketBroker") as mock_broker_cls:
            _emit(DMN_CONFIG, {"type": "act_narration", "text": "hi", "step": 0})

        mock_broker_cls.assert_not_called()


# ---------------------------------------------------------------------------
# C6. UMP post_turn does ONLY skill suggestion                      (§4e / §6)
# ---------------------------------------------------------------------------


class TestUmpPostTurnSkillSuggestionOnly:
    """C6: UMP post_turn fires skill suggestion only — no phase update, no metrics."""

    def _make_mp_flat(self, loop_exited_cleanly=True, iteration=5):
        """Return a flat-path mp-like object for post_turn testing."""
        mp = MagicMock(spec=object)
        mp._loop_exited_cleanly = loop_exited_cleanly
        mp._act_trail = ["trail-line-1"]
        mp._raw_input = "test input"
        mp.current_iteration = iteration
        mp._current_iteration = iteration
        return mp

    def test_skill_suggestion_fired_on_clean_exit_with_many_iterations(self):
        """Skill suggestion fires when loop_exited_cleanly=True, iteration>=4 (C6)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        mp = self._make_mp_flat(loop_exited_cleanly=True, iteration=4)

        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "assistant response text")

        mock_suggest.assert_called_once()

    def test_skill_suggestion_not_fired_when_not_clean_exit(self):
        """Skill suggestion NOT fired when loop did not exit cleanly (C6)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        mp = self._make_mp_flat(loop_exited_cleanly=False, iteration=10)

        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "response")

        mock_suggest.assert_not_called()

    def test_skill_suggestion_not_fired_when_too_few_iterations(self):
        """Skill suggestion NOT fired when iteration < 4 (C6)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        mp = self._make_mp_flat(loop_exited_cleanly=True, iteration=2)

        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ) as mock_suggest:
            cfg.post_turn(mp, "response")

        mock_suggest.assert_not_called()

    def test_no_metrics_recording_in_post_turn(self):
        """UMP post_turn does NOT call MetricsService (§4e — metrics moved to gateway) (C6)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        mp = self._make_mp_flat(loop_exited_cleanly=True, iteration=5)

        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ):
            with patch("services.metrics_service.MetricsService") as mock_metrics_cls:
                cfg.post_turn(mp, "response")

        mock_metrics_cls.assert_not_called()

    def test_no_phase_update_in_post_turn(self):
        """UMP post_turn does NOT call ConversationPhaseService (feature killed) (C6)."""
        from configs.channels import make_user_config
        cfg = make_user_config()
        mp = self._make_mp_flat(loop_exited_cleanly=True, iteration=5)

        with patch(
            "services.skill_suggestion_message_processor.maybe_suggest_skill"
        ):
            # ConversationPhaseService was removed; verify no import attempt.
            with patch.dict("sys.modules", {"services.conversation_phase_service": None}):
                # Should not raise even without that module.
                cfg.post_turn(mp, "response")
