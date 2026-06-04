# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Background/housekeeping configs — observable runtime behaviour only.

Each former background subclass is now a ProcessorConfig. These tests cover the
behaviour those configs drive at runtime: the user-summary post_turn writes to
data_graph, _record() runs post_turn after the assistant row is persisted, and
the super-episode factory threads its captured sources/spans into the prompt.

They deliberately do NOT test config-field constants, frozen-ness/shape
contracts, "post_turn does not record metrics" guards, or source-text scans of
the subconscious worker — those assert structure, not behaviour.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# User-summary post_turn writes data_graph                              (§3b)
# ---------------------------------------------------------------------------


class TestUserSummaryConfig:
    """UserSummaryConfig's post_turn parses {short, long} → data_graph."""

    def test_post_turn_writes_data_graph_on_valid_json(self, db):
        """post_turn parses {short, long} JSON and writes both rows to data_graph."""
        from configs.channels import UserSummaryConfig
        cfg = UserSummaryConfig()

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
# post_turn runs AFTER the assistant row is persisted                  (§4/C4)
# ---------------------------------------------------------------------------


class TestPostTurnRunsAfterAssistantRow:
    """C4: _record() calls write_assistant_row before post_turn when skip_transcript=False."""

    def test_post_turn_called_after_assistant_row(self):
        """In _record(), the assistant row is written before post_turn fires."""
        call_order: list[str] = []

        def _mock_write_assistant(channel, text):
            call_order.append("write_assistant")

        def _post_turn(mp, response_text):
            call_order.append("post_turn")

        from services.processor_config import ProcessorConfig
        from tests.helpers import StubProcessorConfig
        cfg = StubProcessorConfig(
            channel="c4_test",
            role="c4",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.CHAT,
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

        from services.message_processor import MessageProcessor
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "", {})
        mp.config = cfg
        mp.uid = 42
        mp._uid = 42
        mp.cancel_event = threading.Event()

        with patch(
            "services.transcript_service.write_assistant_row",
            side_effect=_mock_write_assistant,
        ):
            mp._record("response text")

        assert call_order == ["write_assistant", "post_turn"], (
            f"post_turn must follow write_assistant_row; got order={call_order}"
        )


# ---------------------------------------------------------------------------
# Super-episode factory threads captured sources/spans into the prompt  (§3c)
# ---------------------------------------------------------------------------


class TestSuperEpisodeConfig:
    """SuperEpisodeConfig captures sources/spans at construction time and the
    build_user_prompt closure renders them — one config per cluster (O2)."""

    def test_super_episode_config_build_user_prompt_uses_sources_spans(self):
        """build_user_prompt produces a non-empty prompt containing the cluster's data."""
        from configs.channels import SuperEpisodeConfig
        sources = [{"id": "ep1", "gist": "User discussed AI plans"}]
        spans = "raw transcript text here"
        cfg = SuperEpisodeConfig("user", sources, spans)
        mp = MagicMock()
        object.__setattr__(cfg, "mp", mp)
        result = cfg.get_user_prompt()
        assert isinstance(result, str)
        assert "ep1" in result or "AI plans" in result or "transcript" in result
