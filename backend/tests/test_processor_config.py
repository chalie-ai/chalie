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

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("channel", "hacked"),
            ("role", "hacked"),
            ("max_iterations", 9999),
            ("suppress_history", False),
            ("broadcast_to", "user"),
            ("memory_seed", True),
            ("post_turn", lambda _mp, _r: None),
        ],
    )
    def test_field_is_immutable(self, field, value):
        """Mutating any §2 field on a frozen ProcessorConfig fails (A3)."""
        cfg = self._make_config()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            setattr(cfg, field, value)

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

    @pytest.mark.parametrize(
        ("name", "channel", "role"),
        [
            ("DMN_CONFIG", "dmn", "proactive_thought"),
            ("EPISODE_ENCODER_CONFIG", "episode_encoder", "episode_encoder"),
            ("SKILL_SUGGESTION_CONFIG", "skills_building", "skills_building"),
            ("COMPACTION_CONFIG", "compaction", "compaction"),
            ("SUBAGENT_COMPACTION_CONFIG", "subagent_compaction", "subagent_compaction"),
        ],
    )
    def test_static_config_is_constant(self, name, channel, role):
        """Static channels are ProcessorConfig instances, not callables (A4 §3a)."""
        import configs.channels as channels
        from services.processor_config import ProcessorConfig

        cfg = getattr(channels, name)
        assert isinstance(cfg, ProcessorConfig), (
            f"{name} must be a ProcessorConfig constant"
        )
        assert cfg.channel == channel
        assert cfg.role == role

    # ── Factory functions (§3b) ────────────────────────────────────────────

    @pytest.mark.parametrize(
        ("name", "kwargs", "channel", "role"),
        [
            ("make_user_config", {"metadata": {}}, "user", "user"),
            (
                "make_eamp_config",
                {
                    "agent_name": "testbot",
                    "project": "proj1",
                    "loop_in_human": False,
                    "wrapper_id": "w1",
                },
                "external-agent:testbot",
                "external_agent",
            ),
            (
                "make_pattern_config",
                {"window_start": 0, "window_end": 100},
                "pattern_match",
                "pattern_match",
            ),
            (
                "make_geo_config",
                {"window_start": 0, "window_end": 100},
                "geo_pattern",
                "geo_pattern",
            ),
            ("make_user_summary_config", {}, "user_summary", "user_summary"),
            (
                "make_super_episode_config",
                {"channel": "user", "sources": [], "spans": []},
                "super_episode_encoder",
                "super_episode_encoder",
            ),
        ],
    )
    def test_factory_config_is_callable_returning_config(
        self, name, kwargs, channel, role
    ):
        """Per-instance channels are callable factories returning ProcessorConfig (A4 §3b)."""
        import configs.channels as channels
        from services.processor_config import ProcessorConfig

        maker = getattr(channels, name)
        assert callable(maker), f"{name} must be a factory function"
        result = maker(**kwargs)
        assert isinstance(result, ProcessorConfig)
        assert result.channel == channel
        assert result.role == role
