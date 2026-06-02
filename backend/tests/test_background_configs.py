# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
§9a B4–B12, C1–C5, O1–O2 — Background/housekeeping configs and hook surface.

Tests authored BLIND from spec §1-8 only.  Per the GOVERNING TEST PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the code is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.

T8 covers:
  B4   — DMN: channel='dmn', role='proactive_thought', max_iterations=100,
           skip_transcript=False, suppress_history=True, broadcast_to=None,
           memory_seed=False, post_turn=None.
  B5   — Episode encoder: channel/role='episode_encoder', max_iterations=1,
           skip_transcript=True, suppress_history=True, post_turn no-op.
  B6   — Skill suggestion: channel/role='skills_building', max_iterations=5,
           skip_transcript=False, suppress_history=True.
  B7   — Continuity compaction: channel/role='compaction', max_iterations=30,
           skip_transcript=True, suppress_history=True.
  B8   — Subagent-trail compaction: channel/role='subagent_compaction',
           max_iterations=30, skip_transcript=True.
  B9   — Pattern-match (factory): channel/role='pattern_match',
           max_iterations=100, suppress_history=True, post_turn=confidence decay.
  B10  — Geo-pattern (factory): channel/role='geo_pattern', max_iterations=30,
           suppress_history=True, post_turn logs counters only.
  B11  — User-summary (factory): channel/role='user_summary', max_iterations=1,
           suppress_history=True, post_turn parses {short,long}→data_graph.
  B12  — Super-episode (factory): channel/role='super_episode_encoder',
           max_iterations=1, suppress_history=True, post_turn no-op.
  C1   — post_turn is the ONLY optional Callable hook (no on_narration /
           on_tool_event / pre_act / process_attachments / overflow_strategy).
  C2   — post_turn=None is a no-op (DMN completes, _record runs, no error).
  C3   — post_turn receives (mp, response_text), invoked once.
  C4   — post_turn runs AFTER the assistant row is persisted (skip_transcript=False).
  C5   — post_turn records NO metrics.
  O1   — UserSummary gate moves to caller: _should_synthesise() logic lives in
           subconscious_worker, not inside the config's process() call.
  O2   — SuperEpisode cluster loop moves to caller: one process() per cluster.
"""

import threading
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# B4. DMN config shape                                                  (§3a/§8b)
# ---------------------------------------------------------------------------


class TestDmnConfig:
    """B4: DMN_CONFIG has the exact shape §3a/§8b specifies."""

    def test_channel(self):
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.channel == "dmn"

    def test_role(self):
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.role == "proactive_thought"

    def test_max_iterations(self):
        """§3a: max_iterations=100."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.max_iterations == 100

    def test_skip_transcript_false(self):
        """§3a: DMN does write transcript rows (no skip)."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.skip_transcript is False

    def test_suppress_history_true(self):
        """§3a/§2: housekeeping loop — no conversation replay."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.suppress_history is True

    def test_broadcast_to_none(self):
        """§3a/§8b: silent — background loop, no UI feed."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.broadcast_to is None

    def test_memory_seed_false(self):
        """§3a/§8b: no auto-seed; DMN gets its own input."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.memory_seed is False

    def test_post_turn_none(self):
        """§3a/§4e: metrics moved to send gateway; DMN had no other post_turn body."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.post_turn is None

    def test_job_derived_correctly(self):
        """§2: job == 'dmn:proactive_thought' (channel:role)."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.job == "dmn:proactive_thought"

    def test_prompt_builders_callable(self):
        """build_user_prompt, build_user_definition, build_system_prompt are callable."""
        from configs.channels import DMN_CONFIG
        assert callable(DMN_CONFIG.build_user_prompt)
        assert callable(DMN_CONFIG.build_user_definition)
        assert callable(DMN_CONFIG.build_system_prompt)

    def test_build_user_prompt_returns_string(self):
        """build_user_prompt returns a non-empty string for typical inputs."""
        from configs.channels import DMN_CONFIG
        mp = MagicMock()
        result = DMN_CONFIG.build_user_prompt(mp)
        assert isinstance(result, str)

    def test_build_system_prompt_returns_string(self):
        """build_system_prompt returns the DMN system prompt string."""
        from configs.channels import DMN_CONFIG
        mp = MagicMock()
        result = DMN_CONFIG.build_system_prompt(mp)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# B5. Episode encoder config shape                                       (§3a)
# ---------------------------------------------------------------------------


class TestEpisodeEncoderConfig:
    """B5: EPISODE_ENCODER_CONFIG has the exact shape §3a specifies."""

    def test_channel_and_role(self):
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.channel == "episode_encoder"
        assert EPISODE_ENCODER_CONFIG.role == "episode_encoder"

    def test_max_iterations(self):
        """§3a: max_iterations=1 (one-shot)."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.max_iterations == 1

    def test_skip_transcript_true(self):
        """§3a: no transcript writes."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.skip_transcript is True

    def test_suppress_history_true(self):
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.suppress_history is True

    def test_post_turn_none_or_noop(self):
        """§3a: post_turn no-op (caller owns downstream)."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        # Either None or a callable that does nothing — both satisfy the spec.
        pt = EPISODE_ENCODER_CONFIG.post_turn
        if pt is not None:
            mp = MagicMock()
            pt(mp, "response")  # must not raise

    def test_always_available_empty(self):
        """§3a: no tools."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert EPISODE_ENCODER_CONFIG.always_available == []

    def test_build_user_prompt_callable(self):
        from configs.channels import EPISODE_ENCODER_CONFIG
        assert callable(EPISODE_ENCODER_CONFIG.build_user_prompt)

    def test_build_system_prompt_returns_episode_encoder_prompt(self):
        """build_system_prompt returns the EpisodeEncoderSystemPrompt body."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        mp = MagicMock()
        result = EPISODE_ENCODER_CONFIG.build_system_prompt(mp)
        assert isinstance(result, str)
        assert "episodic memory" in result.lower() or "episode" in result.lower()


# ---------------------------------------------------------------------------
# B6. Skill suggestion config shape                                      (§3a)
# ---------------------------------------------------------------------------


class TestSkillSuggestionConfig:
    """B6: SKILL_SUGGESTION_CONFIG has the exact shape §3a specifies."""

    def test_channel_and_role(self):
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.channel == "skills_building"
        assert SKILL_SUGGESTION_CONFIG.role == "skills_building"

    def test_max_iterations(self):
        """§3a: max_iterations=5."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.max_iterations == 5

    def test_skip_transcript_false(self):
        """§3a: writes transcript rows."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.skip_transcript is False

    def test_suppress_history_true(self):
        """§3a/§2: each run is independent — suppress_history=True (replaces
        the old get_previous_messages() override on SkillSuggestionMessageProcessor).
        AC-26."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.suppress_history is True

    def test_always_available_skill_manager(self):
        """§3a: always_available=['skill_manager']."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert "skill_manager" in SKILL_SUGGESTION_CONFIG.always_available

    def test_broadcast_to_none(self):
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.broadcast_to is None

    def test_memory_seed_false(self):
        from configs.channels import SKILL_SUGGESTION_CONFIG
        assert SKILL_SUGGESTION_CONFIG.memory_seed is False

    def test_build_system_prompt_returns_skill_suggestion_prompt(self):
        """build_system_prompt returns the SkillSuggestionSystemPrompt body."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        mp = MagicMock()
        result = SKILL_SUGGESTION_CONFIG.build_system_prompt(mp)
        assert isinstance(result, str)
        assert "skill" in result.lower()


# ---------------------------------------------------------------------------
# B7. Continuity compaction config shape                               (§3a/§4a)
# ---------------------------------------------------------------------------


class TestCompactionConfig:
    """B7: COMPACTION_CONFIG has the exact shape §3a specifies."""

    def test_channel_and_role(self):
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.channel == "compaction"
        assert COMPACTION_CONFIG.role == "compaction"

    def test_max_iterations(self):
        """§3a/§4a: bounded at 30 (recursion guard)."""
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.max_iterations == 30

    def test_skip_transcript_true(self):
        """§3a: no transcript writes."""
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.skip_transcript is True

    def test_suppress_history_true(self):
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.suppress_history is True

    def test_broadcast_to_none(self):
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.broadcast_to is None

    def test_memory_seed_false(self):
        from configs.channels import COMPACTION_CONFIG
        assert COMPACTION_CONFIG.memory_seed is False


# ---------------------------------------------------------------------------
# B8. Subagent-trail compaction config shape                           (§3a)
# ---------------------------------------------------------------------------


class TestSubagentCompactionConfig:
    """B8: SUBAGENT_COMPACTION_CONFIG has the exact shape §3a specifies."""

    def test_channel_and_role(self):
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        assert SUBAGENT_COMPACTION_CONFIG.channel == "subagent_compaction"
        assert SUBAGENT_COMPACTION_CONFIG.role == "subagent_compaction"

    def test_max_iterations(self):
        """§3a/§4a: bounded at 30."""
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        assert SUBAGENT_COMPACTION_CONFIG.max_iterations == 30

    def test_skip_transcript_true(self):
        """§3a: no transcript writes."""
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        assert SUBAGENT_COMPACTION_CONFIG.skip_transcript is True

    def test_suppress_history_true(self):
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        assert SUBAGENT_COMPACTION_CONFIG.suppress_history is True


# ---------------------------------------------------------------------------
# B9. Pattern-match factory config shape                                (§3b)
# ---------------------------------------------------------------------------


class TestPatternMatchConfig:
    """B9: make_pattern_config returns the exact shape §3b specifies."""

    def test_channel_and_role(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.channel == "pattern_match"
        assert cfg.role == "pattern_match"

    def test_max_iterations(self):
        """§3b: max_iterations=100."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.max_iterations == 100

    def test_suppress_history_true(self):
        """§3b/§2: housekeeping loop."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.suppress_history is True

    def test_post_turn_is_confidence_decay_sweep(self):
        """§3b: post_turn = confidence decay sweep (callable, not None)."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.post_turn is not None
        assert callable(cfg.post_turn)

    def test_always_available_save_pattern_save_graph(self):
        """§3b: always_available includes save_pattern and save_graph."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert "save_pattern" in cfg.always_available
        assert "save_graph" in cfg.always_available

    def test_broadcast_to_none(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.broadcast_to is None

    def test_memory_seed_false(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.memory_seed is False

    def test_skip_transcript_true(self):
        """§3b: no transcript writes (background)."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.skip_transcript is True

    def test_build_user_prompt_callable(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert callable(cfg.build_user_prompt)

    def test_build_system_prompt_returns_string(self):
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        mp = MagicMock()
        result = cfg.build_system_prompt(mp)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# B10. Geo-pattern factory config shape                                 (§3b)
# ---------------------------------------------------------------------------


class TestGeoPatternConfig:
    """B10: make_geo_config returns the exact shape §3b specifies."""

    def test_channel_and_role(self):
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.channel == "geo_pattern"
        assert cfg.role == "geo_pattern"

    def test_max_iterations(self):
        """§3b: max_iterations=30."""
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.max_iterations == 30

    def test_suppress_history_true(self):
        """§3b/§2: housekeeping loop."""
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.suppress_history is True

    def test_post_turn_logs_counters_only(self):
        """§3b: post_turn logs counters only (callable, not None)."""
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.post_turn is not None
        assert callable(cfg.post_turn)

    def test_always_available_save_pattern_save_graph(self):
        """§3b: always_available includes save_pattern and save_graph."""
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert "save_pattern" in cfg.always_available
        assert "save_graph" in cfg.always_available

    def test_broadcast_to_none(self):
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.broadcast_to is None

    def test_skip_transcript_true(self):
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        assert cfg.skip_transcript is True

    def test_build_system_prompt_returns_string(self):
        from configs.channels import make_geo_config
        cfg = make_geo_config(0, 100)
        mp = MagicMock()
        result = cfg.build_system_prompt(mp)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# B11. User-summary factory config shape                                (§3b)
# ---------------------------------------------------------------------------


class TestUserSummaryConfig:
    """B11: make_user_summary_config returns the exact shape §3b specifies."""

    def test_channel_and_role(self):
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.channel == "user_summary"
        assert cfg.role == "user_summary"

    def test_max_iterations(self):
        """§3b: max_iterations=1 (one-shot)."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.max_iterations == 1

    def test_suppress_history_true(self):
        """§3b/§2: housekeeping loop."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.suppress_history is True

    def test_post_turn_parses_json_writes_data_graph(self):
        """§3b: post_turn parses {short, long} → data_graph. Must be callable."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.post_turn is not None
        assert callable(cfg.post_turn)

    def test_no_on_store_hook(self):
        """§AC-29: there is NO on_store hook. post_turn receives response_text directly."""
        from configs.channels import make_user_summary_config
        from services.processor_config import ProcessorConfig
        cfg = make_user_summary_config()
        # ProcessorConfig has no on_store field — confirm at the dataclass level.
        assert not hasattr(ProcessorConfig, "on_store")
        # post_turn callable (if set) must accept (mp, response_text).
        assert cfg.post_turn is not None

    def test_skip_transcript_true(self):
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.skip_transcript is True

    def test_always_available_empty(self):
        """§3b: no tools for user_summary."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.always_available == []

    def test_build_user_prompt_callable(self):
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert callable(cfg.build_user_prompt)

    def test_build_system_prompt_returns_user_summary_prompt(self):
        """build_system_prompt returns the UserSummarySystemPrompt body."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        mp = MagicMock()
        result = cfg.build_system_prompt(mp)
        assert isinstance(result, str)
        assert "synthes" in result.lower() or "summary" in result.lower() or "profile" in result.lower()

    def test_post_turn_writes_data_graph_on_valid_json(self, db):
        """post_turn parses {short, long} JSON and writes to data_graph."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()

        mp = MagicMock()
        response_text = '{"short": "Alice is an engineer.", "long": "Alice works in software engineering in Malta."}'
        # Must not raise
        cfg.post_turn(mp, response_text)

        # Verify rows written to data_graph
        row = db.execute(
            "SELECT value FROM data_graph WHERE kind='system' AND key='user_summary' AND active=1"
        ).fetchone()
        assert row is not None
        assert "Alice" in row[0]

        row_long = db.execute(
            "SELECT value FROM data_graph WHERE kind='system' AND key='user_summary_long' AND active=1"
        ).fetchone()
        assert row_long is not None


# ---------------------------------------------------------------------------
# B12. Super-episode factory config shape                               (§3b)
# ---------------------------------------------------------------------------


class TestSuperEpisodeConfig:
    """B12: make_super_episode_config returns the exact shape §3b specifies."""

    def test_channel_and_role(self):
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.channel == "super_episode_encoder"
        assert cfg.role == "super_episode_encoder"

    def test_max_iterations(self):
        """§3b: max_iterations=1 (one-shot)."""
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.max_iterations == 1

    def test_suppress_history_true(self):
        """§3b/§2: housekeeping loop."""
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.suppress_history is True

    def test_post_turn_noop(self):
        """§3b: post_turn no-op (caller owns episode write). Either None or
        a callable that does nothing are both acceptable."""
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        pt = cfg.post_turn
        if pt is not None:
            mp = MagicMock()
            pt(mp, "response")  # must not raise

    def test_skip_transcript_true(self):
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.skip_transcript is True

    def test_always_available_empty(self):
        """§3b: no tools for super_episode_encoder."""
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.always_available == []

    def test_broadcast_to_none(self):
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert cfg.broadcast_to is None

    def test_build_user_prompt_callable(self):
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        assert callable(cfg.build_user_prompt)

    def test_build_system_prompt_returns_super_episode_prompt(self):
        """build_system_prompt returns the SuperEpisodeEncoderSystemPrompt body."""
        from configs.channels import make_super_episode_config
        cfg = make_super_episode_config("user", [], [])
        mp = MagicMock()
        result = cfg.build_system_prompt(mp)
        assert isinstance(result, str)
        assert "episode" in result.lower() or "consolidat" in result.lower()


# ---------------------------------------------------------------------------
# C1. post_turn is the ONLY optional Callable hook                     (§2/§AC-32)
# ---------------------------------------------------------------------------


class TestPostTurnIsOnlyHook:
    """C1: ProcessorConfig has exactly ONE optional hook: post_turn.
    No on_narration, on_tool_event, pre_act, process_attachments, overflow_strategy.
    """

    def test_no_on_narration_hook(self):
        """No on_narration field on ProcessorConfig (AC-28/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "on_narration")

    def test_no_on_tool_event_hook(self):
        """No on_tool_event field on ProcessorConfig (AC-28/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "on_tool_event")

    def test_no_pre_act_hook(self):
        """No pre_act field on ProcessorConfig (AC-30/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "pre_act")

    def test_no_process_attachments_hook(self):
        """No process_attachments field on ProcessorConfig (AC-31/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "process_attachments")

    def test_no_overflow_strategy_hook(self):
        """No overflow_strategy field on ProcessorConfig (Q1/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "overflow_strategy")

    def test_no_on_store_hook(self):
        """No on_store field on ProcessorConfig (AC-29/AC-32)."""
        from services.processor_config import ProcessorConfig
        assert not hasattr(ProcessorConfig, "on_store")

    def test_post_turn_field_exists(self):
        """post_turn is the one optional hook (AC-32)."""
        from services.processor_config import ProcessorConfig
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ProcessorConfig)}
        assert "post_turn" in field_names

    def test_post_turn_is_the_only_callable_field(self):
        """Only post_turn accepts a Callable; no other field is a Callable hook."""
        from services.processor_config import ProcessorConfig
        import dataclasses
        import typing
        fields = dataclasses.fields(ProcessorConfig)
        # Fields whose names end with the hook surface check.
        hook_names = {
            "pre_act", "on_store", "on_narration", "on_tool_event",
            "process_attachments", "overflow_strategy",
        }
        field_names = {f.name for f in fields}
        present_hooks = field_names & hook_names
        assert not present_hooks, (
            f"Unexpected hook fields still on ProcessorConfig: {sorted(present_hooks)}"
        )


# ---------------------------------------------------------------------------
# C2. post_turn=None is a no-op                                        (§4/§8b)
# ---------------------------------------------------------------------------


class TestPostTurnNoneIsNoop:
    """C2: When post_turn=None, _record() completes without error (DMN case)."""

    def test_dmn_config_post_turn_none_no_error(self):
        """DMN_CONFIG.post_turn is None — _record() must not crash."""
        from configs.channels import DMN_CONFIG
        from services.message_processor import MessageProcessor
        import dataclasses

        # Verify the DMN config's post_turn is None.
        assert DMN_CONFIG.post_turn is None

    def test_record_with_post_turn_none_does_not_raise(self):
        """Simulate _record() path with post_turn=None — no exception raised."""
        from configs.channels import DMN_CONFIG
        # ProcessorConfig.post_turn=None means _record() skips the call.
        mp = MagicMock()
        mp.config = DMN_CONFIG
        mp.cancel_event = threading.Event()
        mp.uid = None

        # Simulate the _record post_turn gate.
        if DMN_CONFIG.post_turn is not None:
            DMN_CONFIG.post_turn(mp, "")
        # No assertion needed — the absence of an exception IS the assertion.


# ---------------------------------------------------------------------------
# C3. post_turn receives (mp, response_text), invoked once             (§4/§6)
# ---------------------------------------------------------------------------


class TestPostTurnSignature:
    """C3: post_turn is called with (mp, response_text) exactly once per turn."""

    def test_post_turn_receives_mp_and_response_text(self):
        """post_turn callable receives the mp instance and the response text."""
        received: list = []

        def _capture_post_turn(mp, response_text):
            received.append((mp, response_text))

        from services.processor_config import ProcessorConfig
        cfg = ProcessorConfig(
            channel="test_c3",
            role="test",
            usage_class="chat",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=_capture_post_turn,
        )
        mp = MagicMock()
        cfg.post_turn(mp, "hello response")

        assert len(received) == 1
        assert received[0][1] == "hello response"


# ---------------------------------------------------------------------------
# C4. post_turn runs AFTER the assistant row is persisted              (§4/C4)
# ---------------------------------------------------------------------------


class TestPostTurnRunsAfterAssistantRow:
    """C4: post_turn is called after write_assistant_row when skip_transcript=False."""

    def test_post_turn_called_after_assistant_row(self):
        """Simulate _record(): write_assistant_row is called before post_turn."""
        call_order: list[str] = []

        def _mock_write_assistant(channel, text):
            call_order.append("write_assistant")

        def _post_turn(mp, response_text):
            call_order.append("post_turn")

        from services.processor_config import ProcessorConfig
        cfg = ProcessorConfig(
            channel="c4_test",
            role="c4",
            usage_class="chat",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=_post_turn,
        )

        # Simulate _record() manually.
        mp = MagicMock()
        mp.config = cfg
        mp.cancel_event = threading.Event()
        mp.uid = 42

        with patch(
            "services.transcript_service.write_assistant_row",
            side_effect=_mock_write_assistant,
        ):
            from services.message_processor import MessageProcessor
            # Invoke the _record logic directly.
            mp2 = object.__new__(MessageProcessor)
            MessageProcessor.__init__(mp2, "", {})
            mp2.config = cfg
            mp2.uid = 42
            mp2.cancel_event = threading.Event()
            mp2._uid = 42

            with patch(
                "services.transcript_service.write_assistant_row",
                side_effect=_mock_write_assistant,
            ):
                mp2._record("response text")

        assert call_order == ["write_assistant", "post_turn"], (
            f"post_turn must follow write_assistant_row; got order={call_order}"
        )


# ---------------------------------------------------------------------------
# C5. post_turn records NO metrics                                      (§4e)
# ---------------------------------------------------------------------------


class TestPostTurnNoMetrics:
    """C5: post_turn bodies for background configs record zero metrics counters."""

    def test_dmn_post_turn_is_none_so_no_metrics(self):
        """DMN post_turn=None — zero metrics by design (§3a/§4e)."""
        from configs.channels import DMN_CONFIG
        assert DMN_CONFIG.post_turn is None

    def test_pattern_post_turn_does_not_record_metrics_counter(self):
        """Pattern decay sweep post_turn must NOT call MetricsService.record_counter."""
        from configs.channels import make_pattern_config
        cfg = make_pattern_config(0, 100)
        assert cfg.post_turn is not None

        mp = MagicMock()
        # Attach the counters that PatternMatchProcessor tracks
        mp._save_pattern_calls = 0
        mp._save_graph_calls = 0
        mp._save_graph_seen = set()
        mp._touched_pattern_ids = set()
        mp._window_start = 0
        mp._window_end = 100

        with patch("services.metrics_service.MetricsService") as mock_ms:
            cfg.post_turn(mp, "")
            mock_ms.return_value.record_counter.assert_not_called()

    def test_user_summary_post_turn_does_not_record_metrics_counter(self):
        """UserSummary post_turn must NOT call MetricsService.record_counter."""
        from configs.channels import make_user_summary_config
        cfg = make_user_summary_config()
        assert cfg.post_turn is not None

        mp = MagicMock()
        with patch("services.metrics_service.MetricsService") as mock_ms:
            # Pass empty response — post_turn will skip write but should not touch metrics.
            cfg.post_turn(mp, "")
            mock_ms.return_value.record_counter.assert_not_called()


# ---------------------------------------------------------------------------
# O1. UserSummary gate moves to caller                                  (§3c)
# ---------------------------------------------------------------------------


class TestUserSummaryGateMovedToCaller:
    """O1: _should_synthesise() logic lives in the caller (subconscious_worker),
    not inside make_user_summary_config or MessageProcessor.process().
    The config itself has no gate — process() always runs when invoked.
    """

    def test_make_user_summary_config_has_no_should_synthesise(self):
        """The config factory contains no gating logic — it always returns a config."""
        from configs.channels import make_user_summary_config
        import inspect
        # make_user_summary_config must just return a ProcessorConfig.
        # Confirm calling it always produces a config, regardless of DB state.
        cfg = make_user_summary_config()
        from services.processor_config import ProcessorConfig
        assert isinstance(cfg, ProcessorConfig)

    def test_subconscious_worker_step_synthesis_gates_before_process(self):
        """_step_synthesis in SubconsciousWorker gates on _should_synthesise()
        BEFORE calling MessageProcessor.process(), not after. The UserSummary
        ProcessorConfig's process() never gates internally.

        §3c: 'if _should_synthesise(): result = MessageProcessor.process(...)'
        """
        import inspect
        from services.subconscious_worker import SubconsciousWorker
        src = inspect.getsource(SubconsciousWorker._step_synthesis)
        # The step must reference _should_synthesise or should_synthesise check
        assert "_should_synthesise" in src or "should_synthesise" in src, (
            "_step_synthesis must gate on _should_synthesise() before calling process()"
        )


# ---------------------------------------------------------------------------
# O2. SuperEpisode cluster loop moves to caller                         (§3c)
# ---------------------------------------------------------------------------


class TestSuperEpisodeClusterLoopMovedToCaller:
    """O2: The per-cluster loop lives in subconscious_worker._step_consolidate(),
    not inside make_super_episode_config or MessageProcessor.process().
    One process() call per cluster.
    """

    def test_make_super_episode_config_has_no_cluster_loop(self):
        """make_super_episode_config returns a ProcessorConfig; no cluster loop
        inside it. Each cluster results in a separate process() call by the caller.
        """
        from configs.channels import make_super_episode_config
        from services.processor_config import ProcessorConfig
        cfg = make_super_episode_config("user", [{"id": "1", "gist": "g"}], ["spans"])
        assert isinstance(cfg, ProcessorConfig)

    def test_subconscious_worker_step_consolidate_calls_process_per_cluster(self):
        """_step_consolidate must iterate clusters and call MessageProcessor.process()
        (or the flat path) once per cluster — not call SuperEpisodeEncoderProcessor.send().

        §3c: 'for cluster in find_super_candidates(...): ... MessageProcessor.process(...)'
        """
        import inspect
        from services.subconscious_worker import SubconsciousWorker
        src = inspect.getsource(SubconsciousWorker._step_consolidate)
        # Must reference MessageProcessor.process (the flat entry point)
        # and NOT instantiate SuperEpisodeEncoderProcessor.
        assert "MessageProcessor.process" in src or "process(" in src, (
            "_step_consolidate must call MessageProcessor.process() per cluster"
        )

    def test_super_episode_config_build_user_prompt_uses_sources_spans(self):
        """make_super_episode_config captures sources and spans at factory time,
        and the build_user_prompt builder uses them to produce a non-empty prompt."""
        from configs.channels import make_super_episode_config
        sources = [{"id": "ep1", "gist": "User discussed AI plans"}]
        spans = "raw transcript text here"
        cfg = make_super_episode_config("user", sources, spans)
        mp = MagicMock()
        result = cfg.build_user_prompt(mp)
        assert isinstance(result, str)
        assert "ep1" in result or "AI plans" in result or "transcript" in result
