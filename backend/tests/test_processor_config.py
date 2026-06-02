# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
§9a A1–A4 — ProcessorConfig contract & identity.

Tests authored BLIND from spec §1–8 only.  Per the GOVERNING TEST PRINCIPLE:
  - Encode behaviour exactly as §9a states.
  - A failing test means the code is wrong, not the test.
  - Never weaken, delete, xfail, or "correct" a §9a test.
"""

import dataclasses

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# A1. `job` derives as `channel:role`                                   (§2)
# ---------------------------------------------------------------------------

class TestJobDerivedProperty:
    """A1: config.job == f'{channel}:{role}' — always derived, never stored."""

    def test_job_dmn(self):
        """DMN channel: job == 'dmn:proactive_thought'."""
        from services.processor_config import ProcessorConfig

        cfg = ProcessorConfig(
            channel="dmn",
            role="proactive_thought",
            usage_class="subconscious",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=100,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

        assert cfg.job == "dmn:proactive_thought"

    def test_job_user(self):
        """UMP channel: job == 'user:user'."""
        from services.processor_config import ProcessorConfig

        cfg = ProcessorConfig(
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
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            broadcast_to="user",
            memory_seed=True,
            post_turn=None,
        )

        assert cfg.job == "user:user"

    def test_job_arbitrary_channel_and_role(self):
        """job == channel:role for arbitrary values."""
        from services.processor_config import ProcessorConfig

        cfg = ProcessorConfig(
            channel="external-agent:mybot",
            role="external_agent",
            usage_class="external_agent",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=200,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=False,
            broadcast_to=None,
            memory_seed=True,
            post_turn=None,
        )

        assert cfg.job == "external-agent:mybot:external_agent"


# ---------------------------------------------------------------------------
# A2. Telemetry label is the derived job — no separate LOG_LABEL         (§2)
# ---------------------------------------------------------------------------

class TestNoLogLabel:
    """A2: ProcessorConfig has no LOG_LABEL attribute; telemetry uses job."""

    def test_no_log_label_attribute(self):
        """ProcessorConfig must not expose a LOG_LABEL field or property."""
        from services.processor_config import ProcessorConfig

        cfg = ProcessorConfig(
            channel="dmn",
            role="proactive_thought",
            usage_class="subconscious",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=100,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

        assert not hasattr(cfg, "LOG_LABEL"), (
            "ProcessorConfig must not have LOG_LABEL; telemetry label is config.job"
        )

    def test_job_is_the_label(self):
        """job == channel:role is the telemetry label passed to Providers."""
        from services.processor_config import ProcessorConfig

        cfg = ProcessorConfig(
            channel="compaction",
            role="compaction",
            usage_class="subconscious",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=30,
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

        # The telemetry label (passed to Providers.calculate / send_messages as
        # the `job` argument) is exactly config.job — no other field needed.
        assert cfg.job == f"{cfg.channel}:{cfg.role}"


# ---------------------------------------------------------------------------
# A3. ProcessorConfig is immutable per-turn — frozen dataclass           (§2)
# ---------------------------------------------------------------------------

class TestProcessorConfigImmutable:
    """A3: Mutating any field raises FrozenInstanceError (frozen=True)."""

    def _make_config(self):
        from services.processor_config import ProcessorConfig

        return ProcessorConfig(
            channel="skills_building",
            role="skills_building",
            usage_class="subconscious",
            build_user_prompt=lambda _: "",
            build_user_definition=lambda _: "",
            build_system_prompt=lambda _: "",
            always_available=[],
            discoverable=[],
            blocked=frozenset(),
            max_iterations=5,
            skip_transcript=False,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
            post_turn=None,
        )

    def test_channel_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.channel = "hacked"

    def test_role_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.role = "hacked"

    def test_max_iterations_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.max_iterations = 9999

    def test_suppress_history_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.suppress_history = False

    def test_broadcast_to_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.broadcast_to = "user"

    def test_memory_seed_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.memory_seed = True

    def test_post_turn_is_immutable(self):
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            cfg.post_turn = lambda _mp, _r: None

    def test_all_fields_present(self):
        """All §2 fields are present on the dataclass."""
        from services.processor_config import ProcessorConfig

        fields = {f.name for f in dataclasses.fields(ProcessorConfig)}
        required = {
            "channel",
            "role",
            "usage_class",
            "build_user_prompt",
            "build_user_definition",
            "build_system_prompt",
            "always_available",
            "discoverable",
            "blocked",
            "max_iterations",
            "skip_transcript",
            "skip_input_row",
            "suppress_history",
            "broadcast_to",
            "memory_seed",
            "post_turn",
        }
        missing = required - fields
        assert not missing, f"ProcessorConfig missing §2 fields: {missing}"


# ---------------------------------------------------------------------------
# A4. Static channels = constants; per-instance channels = factories     (§3)
# ---------------------------------------------------------------------------

class TestStaticVsFactoryConfigs:
    """A4: Static configs are ProcessorConfig instances; factory configs are callables."""

    # ── Static constants (§3a) ─────────────────────────────────────────────

    def test_dmn_config_is_constant(self):
        """DMN_CONFIG is a ProcessorConfig instance, not a callable."""
        from configs.channels import DMN_CONFIG
        from services.processor_config import ProcessorConfig

        assert isinstance(DMN_CONFIG, ProcessorConfig), (
            "DMN_CONFIG must be a ProcessorConfig constant"
        )
        assert DMN_CONFIG.channel == "dmn"
        assert DMN_CONFIG.role == "proactive_thought"

    def test_episode_encoder_config_is_constant(self):
        """EPISODE_ENCODER_CONFIG is a ProcessorConfig instance."""
        from configs.channels import EPISODE_ENCODER_CONFIG
        from services.processor_config import ProcessorConfig

        assert isinstance(EPISODE_ENCODER_CONFIG, ProcessorConfig)
        assert EPISODE_ENCODER_CONFIG.channel == "episode_encoder"
        assert EPISODE_ENCODER_CONFIG.role == "episode_encoder"

    def test_skill_suggestion_config_is_constant(self):
        """SKILL_SUGGESTION_CONFIG is a ProcessorConfig instance."""
        from configs.channels import SKILL_SUGGESTION_CONFIG
        from services.processor_config import ProcessorConfig

        assert isinstance(SKILL_SUGGESTION_CONFIG, ProcessorConfig)
        assert SKILL_SUGGESTION_CONFIG.channel == "skills_building"
        assert SKILL_SUGGESTION_CONFIG.role == "skills_building"

    def test_compaction_config_is_constant(self):
        """COMPACTION_CONFIG is a ProcessorConfig instance."""
        from configs.channels import COMPACTION_CONFIG
        from services.processor_config import ProcessorConfig

        assert isinstance(COMPACTION_CONFIG, ProcessorConfig)
        assert COMPACTION_CONFIG.channel == "compaction"
        assert COMPACTION_CONFIG.role == "compaction"

    def test_subagent_compaction_config_is_constant(self):
        """SUBAGENT_COMPACTION_CONFIG is a ProcessorConfig instance."""
        from configs.channels import SUBAGENT_COMPACTION_CONFIG
        from services.processor_config import ProcessorConfig

        assert isinstance(SUBAGENT_COMPACTION_CONFIG, ProcessorConfig)
        assert SUBAGENT_COMPACTION_CONFIG.channel == "subagent_compaction"
        assert SUBAGENT_COMPACTION_CONFIG.role == "subagent_compaction"

    # ── Factory functions (§3b) ────────────────────────────────────────────

    def test_make_user_config_is_factory(self):
        """make_user_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_user_config
        from services.processor_config import ProcessorConfig

        assert callable(make_user_config), "make_user_config must be a factory function"
        result = make_user_config(metadata={})
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "user"
        assert result.role == "user"

    def test_make_eamp_config_is_factory(self):
        """make_eamp_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_eamp_config
        from services.processor_config import ProcessorConfig

        assert callable(make_eamp_config)
        result = make_eamp_config(
            agent_name="testbot",
            project="proj1",
            loop_in_human=False,
            wrapper_id="w1",
        )
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "external-agent:w1:testbot"
        assert result.role == "external_agent"

    def test_make_pattern_config_is_factory(self):
        """make_pattern_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_pattern_config
        from services.processor_config import ProcessorConfig

        assert callable(make_pattern_config)
        result = make_pattern_config(window_start=0, window_end=100)
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "pattern_match"
        assert result.role == "pattern_match"

    def test_make_geo_config_is_factory(self):
        """make_geo_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_geo_config
        from services.processor_config import ProcessorConfig

        assert callable(make_geo_config)
        result = make_geo_config(window_start=0, window_end=100)
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "geo_pattern"
        assert result.role == "geo_pattern"

    def test_make_user_summary_config_is_factory(self):
        """make_user_summary_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_user_summary_config
        from services.processor_config import ProcessorConfig

        assert callable(make_user_summary_config)
        result = make_user_summary_config()
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "user_summary"
        assert result.role == "user_summary"

    def test_make_super_episode_config_is_factory(self):
        """make_super_episode_config is a callable (factory) that returns ProcessorConfig."""
        from configs.channels import make_super_episode_config
        from services.processor_config import ProcessorConfig

        assert callable(make_super_episode_config)
        result = make_super_episode_config(channel="user", sources=[], spans=[])
        assert isinstance(result, ProcessorConfig)
        assert result.channel == "super_episode_encoder"
        assert result.role == "super_episode_encoder"
