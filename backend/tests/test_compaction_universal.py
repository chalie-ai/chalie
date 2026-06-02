# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""D1–D16 — Two universal compaction strategies (spec §4a).

Tests for the flat-path _compact_trail() and _compact_history() methods on
MessageProcessor, and the compaction dispatch logic inside _loop().

All tests use MessageProcessor.process() (the flat path), not old-path send().
The flat path uses ProcessorConfig + the new _compact_trail/_compact_history
implementations from T6.

Spec §9a areas D1–D16 / AC-23.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ── helpers ──────────────────────────────────────────────────────────────────


def _llm_response(text="done", tool_calls=None):
    from services.llm_service import LLMResponse
    return LLMResponse(
        text=text,
        model="test-model",
        provider="mock",
        tool_calls=tool_calls or [],
    )


def _make_flat_config(
    channel="test_d_channel",
    role="test",
    max_iterations=10,
    suppress_history=True,
    skip_transcript=True,
    system_prompt_text="sys",
    user_prompt_text="hi",
):
    """Return a ProcessorConfig for flat-path testing."""
    from configs.channels import ProcessorConfig

    return ProcessorConfig(
        channel=channel,
        role=role,
        usage_class="subconscious",
        build_user_prompt=lambda _mp: user_prompt_text,
        build_user_definition=lambda _mp: "test user",
        build_system_prompt=lambda _mp: system_prompt_text,
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=max_iterations,
        skip_transcript=skip_transcript,
        skip_input_row=False,
        suppress_history=suppress_history,
        broadcast_to=None,
        memory_seed=False,
        post_turn=None,
    )


def _patch_providers_calculate(calculate_seq, send_response=None):
    """Return a context manager that patches Providers.instance().

    calculate_seq: iterable of floats returned by calculate() in order.
        When the sequence is exhausted, returns 0.0 (safe default — no compaction).
    send_response: LLMResponse returned by send_messages (default: _llm_response()).
    """
    from services.providers import Providers

    resp = send_response if send_response is not None else _llm_response()
    fake = MagicMock()
    if isinstance(calculate_seq, (list, tuple)):
        values = list(calculate_seq)
        idx = [0]

        def _side_effect(*a, **kw):
            if idx[0] < len(values):
                v = values[idx[0]]
                idx[0] += 1
                return v
            return 0.0  # safe default once sequence exhausted

        fake.calculate.side_effect = _side_effect
    else:
        fake.calculate.return_value = calculate_seq
    fake.send_messages.return_value = resp
    return patch.object(Providers, "instance", return_value=fake), fake


# ── D1 — Universal: no per-channel overflow hook ─────────────────────────────


class TestD1UniversalNoOverflowHook:
    """D1: compaction is universal across all channels; ProcessorConfig has no
    overflow_strategy hook (spec §4a/§2)."""

    def test_processor_config_has_no_overflow_strategy(self):
        from services.processor_config import ProcessorConfig
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ProcessorConfig)}
        assert "overflow_strategy" not in field_names, (
            "ProcessorConfig must NOT have an overflow_strategy field — "
            "compaction is universal (D1 / §2 / Q1)"
        )

    def test_compaction_fires_on_non_user_channel(self, db):
        """Compaction is not gated on channel='user' — fires on any channel."""
        config = _make_flat_config(channel="dmn", role="proactive_thought")
        ctx, fake = _patch_providers_calculate(
            # first call >90% with no trail → falls to >80% → _compact_history
            # then after reset, <80% → send
            [0.95, 0.10],
        )
        with ctx:
            # _compact_history will call _run_full_compaction; stub it
            from services.message_processor import MessageProcessor
            with patch.object(MessageProcessor, "_compact_history", return_value=True) as m:
                MessageProcessor.process("hi", config)
        m.assert_called_once()


# ── D2 — Trail compaction fires at >90% with trail ───────────────────────────


class TestD2TrailCompactionFires:
    """D2: _compact_trail() fires when pct>0.90 AND _has_trail() is True."""

    def test_trail_compaction_fires_at_above_90_with_trail(self, db):
        """When calculate() > 0.90 and there is a trail row, _compact_trail
        is called and iteration restarts."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            compact_trail_calls = []

            def _stub_compact_trail(self):
                compact_trail_calls.append(1)
                return True  # success, restart

            def _stub_has_trail(self):
                return True

            with patch.object(MessageProcessor, "_compact_trail", _stub_compact_trail), \
                 patch.object(MessageProcessor, "_has_trail", _stub_has_trail):
                MessageProcessor.process("hi", config)

        assert len(compact_trail_calls) >= 1, (
            "_compact_trail must be called when pct>0.90 and trail exists (D2)"
        )

    def test_trail_compaction_records_trail_compaction_row(self, db):
        """_compact_trail() must record exactly ONE trail_compaction tool_calls row."""
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False)

        # Directly call _compact_trail on an MP instance with a non-empty trail.
        # The inner MessageProcessor.process() (COMPACTION_CONFIG) is stubbed to
        # return "compacted summary" so the record call fires.
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = None
        mp.current_iteration = 0
        mp.cancel_event = threading.Event()
        mp.deadline = None
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []

        recorded = []

        def _tracking_record(tool_name, params, result, transcript_id, ephemeral=True):
            recorded.append(tool_name)

        with patch.object(MessageProcessor, "_render_act_trail",
                           return_value="[tool] some result"), \
             patch.object(MessageProcessor, "process",
                           return_value="compacted summary"), \
             patch("abilities._base.Ability.record", side_effect=_tracking_record):
            result = mp._compact_trail()

        assert result is True, "_compact_trail must return True on success"
        assert "trail_compaction" in recorded, (
            "_compact_trail must record a 'trail_compaction' row (D2 / AC-23): "
            f"recorded={recorded}"
        )


# ── D3 — No trail compaction when no trail ───────────────────────────────────


class TestD3NoTrailCompactionWhenNoTrail:
    """D3: At >90% with no trail, loop proceeds (falls to 80% check, not trail compact)."""

    def test_no_trail_compaction_when_has_trail_is_false(self, db):
        """_compact_trail is NOT called when _has_trail() returns False."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            compact_trail_calls = []

            def _stub_compact_trail(self):
                compact_trail_calls.append(1)
                return True

            with patch.object(MessageProcessor, "_compact_trail", _stub_compact_trail), \
                 patch.object(MessageProcessor, "_has_trail", return_value=False), \
                 patch.object(MessageProcessor, "_compact_history", return_value=True):
                MessageProcessor.process("hi", config)

        assert len(compact_trail_calls) == 0, (
            "_compact_trail must NOT be called when _has_trail() is False (D3)"
        )

    def test_at_above_90_no_trail_falls_through_to_80_check(self, db):
        """When pct>0.90 and no trail, history compaction path is tried."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            history_calls = []

            def _stub_history(self):
                history_calls.append(1)
                return True

            with patch.object(MessageProcessor, "_has_trail", return_value=False), \
                 patch.object(MessageProcessor, "_compact_history", _stub_history):
                MessageProcessor.process("hi", config)

        assert len(history_calls) >= 1, (
            "At pct>0.90 with no trail, _compact_history must be called (D3/D4)"
        )


# ── D4 — History compaction fires at >80% (and ≤90%) ────────────────────────


class TestD4HistoryCompactionFires:
    """D4: _compact_history fires when pct > 0.80 and pct <= 0.90."""

    def test_history_compaction_fires_at_81_percent(self, db):
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.81, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            history_calls = []

            def _stub_history(self):
                history_calls.append(1)
                return True

            with patch.object(MessageProcessor, "_compact_history", _stub_history):
                MessageProcessor.process("hi", config)

        assert len(history_calls) >= 1, (
            "_compact_history must fire when pct=0.81 (>0.80, ≤0.90) (D4)"
        )

    def test_history_compaction_fires_at_exactly_90_percent(self, db):
        """At exactly 90%, history compaction fires (not trail — strict > 0.90)."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.90, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            trail_calls = []
            history_calls = []

            with patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True):
                MessageProcessor.process("hi", config)

        assert len(trail_calls) == 0, "At exactly 90%, trail compaction must NOT fire (strict >)"
        assert len(history_calls) >= 1, "At exactly 90%, history compaction must fire (D4)"


# ── D5 — 90% checked before 80% ──────────────────────────────────────────────


class TestD5OrderingTrailBeforeHistory:
    """D5: 90% threshold is checked before 80%; both exceeded → trail path taken."""

    def test_both_exceeded_takes_trail_path_not_history(self, db):
        """When pct > 0.90 and has trail, only trail compaction fires — not history."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            trail_calls = []
            history_calls = []

            with patch.object(MessageProcessor, "_has_trail", return_value=True), \
                 patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True):
                MessageProcessor.process("hi", config)

        assert len(trail_calls) >= 1, (
            "At pct>0.90 with trail, trail compaction must fire (D5)"
        )
        assert len(history_calls) == 0, (
            "At pct>0.90 with trail, history compaction must NOT fire (D5 — trail takes priority)"
        )


# ── D6 — No compaction at ≤80% ───────────────────────────────────────────────


class TestD6NoCompactionBelow80:
    """D6: At ≤80%, neither compaction path fires; loop proceeds to send."""

    def test_no_compaction_at_79_percent(self, db):
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate(0.79)
        with ctx:
            from services.message_processor import MessageProcessor
            trail_calls = []
            history_calls = []

            with patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True):
                MessageProcessor.process("hi", config)

        assert len(trail_calls) == 0, "At ≤80%, _compact_trail must NOT fire (D6)"
        assert len(history_calls) == 0, "At ≤80%, _compact_history must NOT fire (D6)"

    def test_no_compaction_at_exactly_80_percent(self, db):
        """Threshold is strict > 0.80, so exactly 0.80 does NOT compact."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate(0.80)
        with ctx:
            from services.message_processor import MessageProcessor
            trail_calls = []
            history_calls = []

            with patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True):
                MessageProcessor.process("hi", config)

        assert len(trail_calls) == 0, "At exactly 80%, _compact_trail must NOT fire (strict >)"
        assert len(history_calls) == 0, "At exactly 80%, _compact_history must NOT fire (strict >)"


# ── D7 — Trail and history compaction independent ────────────────────────────


class TestD7Independence:
    """D7: trail and history compaction are independent; neither touches the other's data."""

    def test_compact_trail_does_not_call_run_full_compaction(self, db):
        """_compact_trail must NOT call _run_full_compaction (history path)."""
        config = _make_flat_config(skip_transcript=False)
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            run_full_calls = []

            def _stub_run_full(self, exclude_id=None):
                run_full_calls.append(1)
                return "summary"

            with patch.object(MessageProcessor, "_has_trail", return_value=True), \
                 patch.object(MessageProcessor, "_render_act_trail",
                               return_value="[memory] result → hi"), \
                 patch.object(MessageProcessor, "_run_full_compaction", _stub_run_full):
                MessageProcessor.process("hi", config)

        assert len(run_full_calls) == 0, (
            "_compact_trail must NOT call _run_full_compaction — they are independent (D7)"
        )

    def test_compact_history_does_not_write_trail_compaction_row(self, db):
        """_compact_history writes a durable 'compaction' row, not 'trail_compaction'."""
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False, suppress_history=False)
        ctx, fake = _patch_providers_calculate([0.85, 0.10])
        with ctx:
            written_names = []

            def _tracking_record(tool_name, params, result, transcript_id, ephemeral=True):
                written_names.append(tool_name)

            with patch.object(MessageProcessor, "_run_full_compaction",
                               return_value="summary"), \
                 patch("abilities._base.Ability.record", side_effect=_tracking_record):
                MessageProcessor.process("hi", config)

        assert "trail_compaction" not in written_names, (
            "_compact_history must NOT write a trail_compaction row (D7)"
        )


# ── D8 — Trail compaction: one row, reset iteration, clear discovered tools ───


class TestD8TrailCompactionStateReset:
    """D8: After trail compaction, iteration resets to 0 and discovered_tools cleared."""

    def test_trail_compaction_resets_iteration_and_clears_tools(self, db):
        """After _compact_trail() succeeds, current_iteration == 0 and discovered_tools == []."""
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False)

        def _stub_compact_trail(self):
            # The real _compact_trail should reset these — we just return True
            # here and verify via post-call inspection in the loop.
            # Use the real implementation; just capture state after.
            return True

        original_loop = MessageProcessor._loop

        def _instrumented_loop(self):
            result = original_loop(self)
            return result

        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            # Use real _compact_trail but stub _has_trail and _render_act_trail
            with patch.object(MessageProcessor, "_has_trail", return_value=True), \
                 patch.object(MessageProcessor, "_render_act_trail",
                               return_value="[tool] params → result"):
                # Patch process to capture state
                original_compact_trail = MessageProcessor._compact_trail

                iteration_after_compact = []
                tools_after_compact = []

                def _spy_compact_trail(self):
                    result = original_compact_trail(self)
                    iteration_after_compact.append(self.current_iteration)
                    tools_after_compact.append(list(self.discovered_tools))
                    return result

                with patch.object(MessageProcessor, "_compact_trail", _spy_compact_trail):
                    MessageProcessor.process("hi", config)

        assert iteration_after_compact, "Spy must have been called"
        assert iteration_after_compact[0] == 0, (
            f"After _compact_trail, current_iteration must be 0 (got {iteration_after_compact[0]}) (D8)"
        )
        assert tools_after_compact[0] == [], (
            f"After _compact_trail, discovered_tools must be [] (got {tools_after_compact[0]}) (D8)"
        )


# ── D9 — Empty trail text → no-op success ────────────────────────────────────


class TestD9EmptyTrailNoOp:
    """D9: When trail text is empty (no rows), _compact_trail returns True without
    recording any row (no-op success)."""

    def test_empty_trail_returns_true_no_row_written(self, db):
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False)
        recorded = []

        def _tracking_record(tool_name, params, result, transcript_id, ephemeral=True):
            recorded.append(tool_name)

        with patch("abilities._base.Ability.record", side_effect=_tracking_record), \
             patch.object(MessageProcessor, "_render_act_trail", return_value=""), \
             patch.object(MessageProcessor, "_has_trail", return_value=False):
            mp = object.__new__(MessageProcessor)
            MessageProcessor.__init__(mp, "hi", None)
            mp.config = config
            mp.uid = None
            mp.current_iteration = 3
            mp.cancel_event = threading.Event()
            mp.deadline = None
            mp.thinking_level = "low"
            mp.thinking_exploration = None
            mp.discovered_tools = []
            result = mp._compact_trail()

        assert result is True, "_compact_trail with empty trail must return True (D9)"
        assert "trail_compaction" not in recorded, (
            "_compact_trail with empty trail must NOT record any row (D9)"
        )


# ── D10 — Empty summary → turn aborts ────────────────────────────────────────


class TestD10EmptySummaryAbortsTurn:
    """D10: When the compaction LLM returns empty text, _compact_trail returns False
    → the loop exits with ''."""

    def test_empty_compaction_output_returns_false(self, db):
        """_compact_trail returns False when MessageProcessor.process returns ''."""
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False)

        with patch.object(MessageProcessor, "_render_act_trail",
                           return_value="[tool] some result"), \
             patch.object(MessageProcessor, "process",
                           side_effect=[""]):
            mp = object.__new__(MessageProcessor)
            MessageProcessor.__init__(mp, "hi", None)
            mp.config = config
            mp.uid = None
            mp.current_iteration = 0
            mp.cancel_event = threading.Event()
            mp.deadline = None
            mp.thinking_level = "low"
            mp.thinking_exploration = None
            mp.discovered_tools = []

            result = mp._compact_trail()

        assert result is False, (
            "_compact_trail must return False when process() returns '' (D10)"
        )

    def test_loop_returns_empty_string_when_compact_trail_returns_false(self, db):
        """When _compact_trail returns False, _loop returns '' (turn aborts)."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor

            with patch.object(MessageProcessor, "_has_trail", return_value=True), \
                 patch.object(MessageProcessor, "_compact_trail", return_value=False):
                result = MessageProcessor.process("hi", config)

        assert result == "", (
            "_loop must return '' when _compact_trail fails (D10)"
        )


# ── D11 — History compaction state reset ─────────────────────────────────────


class TestD11HistoryCompactionStateReset:
    """D11: _compact_history clears discovered_tools and resets thinking_exploration + iteration."""

    def test_compact_history_resets_all_state(self, db):
        from services.message_processor import MessageProcessor

        config = _make_flat_config(skip_transcript=False)

        iteration_after = []
        exploration_after = []
        tools_after = []

        original_compact_history = MessageProcessor._compact_history

        def _spy_compact_history(self):
            result = original_compact_history(self)
            iteration_after.append(self.current_iteration)
            exploration_after.append(self.thinking_exploration)
            tools_after.append(list(self.discovered_tools))
            return result

        ctx, fake = _patch_providers_calculate([0.85, 0.10])
        with ctx:
            with patch.object(MessageProcessor, "_compact_history", _spy_compact_history), \
                 patch.object(MessageProcessor, "_run_full_compaction",
                               return_value="summary"):
                mp = object.__new__(MessageProcessor)
                MessageProcessor.__init__(mp, "hi", None)
                mp.config = config
                mp.uid = None
                mp.current_iteration = 5
                mp.cancel_event = threading.Event()
                mp.deadline = None
                mp.thinking_level = "low"
                mp.thinking_exploration = "some exploration"
                mp.discovered_tools = [{"name": "weather"}]

                result = mp._compact_history()

        assert result is True
        assert iteration_after[0] == 0, (
            f"_compact_history must reset current_iteration to 0 (got {iteration_after[0]}) (D11)"
        )
        assert exploration_after[0] is None, (
            "_compact_history must clear thinking_exploration to None (D11)"
        )
        assert tools_after[0] == [], (
            "_compact_history must clear discovered_tools to [] (D11)"
        )


# ── D12 — History compaction failure aborts turn ─────────────────────────────


class TestD12HistoryCompactionFailureAborts:
    """D12: When _run_full_compaction returns None (failure), _compact_history
    returns False → the loop exits with ''."""

    def test_compact_history_returns_false_on_failure(self, db):
        from services.message_processor import MessageProcessor

        config = _make_flat_config()

        with patch.object(MessageProcessor, "_run_full_compaction", return_value=None):
            mp = object.__new__(MessageProcessor)
            MessageProcessor.__init__(mp, "hi", None)
            mp.config = config
            mp.uid = None
            mp.current_iteration = 0
            mp.cancel_event = threading.Event()
            mp.deadline = None
            mp.thinking_level = "low"
            mp.thinking_exploration = None
            mp.discovered_tools = []

            result = mp._compact_history()

        assert result is False, (
            "_compact_history must return False when _run_full_compaction returns None (D12)"
        )

    def test_loop_returns_empty_when_compact_history_fails(self, db):
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.85, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor

            with patch.object(MessageProcessor, "_compact_history", return_value=False):
                result = MessageProcessor.process("hi", config)

        assert result == "", (
            "_loop must return '' when _compact_history fails (D12)"
        )


# ── D13 — History compaction excludes current input row ──────────────────────


class TestD13ExcludeCurrentRow:
    """D13: _compact_history calls _run_full_compaction(exclude_id=self.uid)
    to exclude the current, not-yet-answered turn."""

    def test_compact_history_passes_uid_as_exclude_id(self, db):
        from services.message_processor import MessageProcessor

        config = _make_flat_config()

        exclude_id_received = []

        def _stub_run_full(self, exclude_id=None):
            exclude_id_received.append(exclude_id)
            return "summary"

        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "hi", None)
        mp.config = config
        mp.uid = 42
        mp.current_iteration = 0
        mp.cancel_event = threading.Event()
        mp.deadline = None
        mp.thinking_level = "low"
        mp.thinking_exploration = None
        mp.discovered_tools = []

        with patch.object(MessageProcessor, "_run_full_compaction", _stub_run_full):
            mp._compact_history()

        assert exclude_id_received == [42], (
            "_compact_history must pass uid as exclude_id to _run_full_compaction (D13): "
            f"got {exclude_id_received}"
        )


# ── D14 — Recursion guard ─────────────────────────────────────────────────────


class TestD14RecursionGuard:
    """D14: Inside a compaction loop (channel='compaction' or 'subagent_compaction'),
    >80% logs a warning and proceeds — compaction never compacts itself."""

    def test_compaction_channel_skips_compact_trail(self, db, caplog):
        """When config.channel == 'compaction', _compact_trail is never called
        even if pct > 0.90."""
        from configs.channels import COMPACTION_CONFIG

        trail_calls = []
        history_calls = []

        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor

            with patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True), \
                 caplog.at_level(logging.WARNING):
                MessageProcessor.process("compact me", COMPACTION_CONFIG)

        assert len(trail_calls) == 0, (
            "_compact_trail must NOT be called inside a compaction loop (D14)"
        )
        assert len(history_calls) == 0, (
            "_compact_history must NOT be called inside a compaction loop (D14)"
        )

    def test_compaction_channel_logs_warning_above_80(self, db, caplog):
        """When inside a compaction loop and pct > 0.80, a warning is logged."""
        from configs.channels import COMPACTION_CONFIG

        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx, caplog.at_level(logging.WARNING, logger="services.message_processor"):
            from services.message_processor import MessageProcessor
            MessageProcessor.process("compact me", COMPACTION_CONFIG)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("compaction" in m.lower() or "recursion" in m.lower()
                   for m in warning_msgs), (
            "A WARNING must be logged when pct>0.80 inside a compaction loop (D14): "
            f"warnings={warning_msgs}"
        )

    def test_subagent_compaction_channel_skips_compaction(self, db):
        """channel='subagent_compaction' also applies the recursion guard."""
        from configs.channels import SUBAGENT_COMPACTION_CONFIG

        trail_calls = []
        history_calls = []

        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor

            with patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True), \
                 patch.object(MessageProcessor, "_compact_history",
                               lambda self: history_calls.append(1) or True):
                MessageProcessor.process("compact me", SUBAGENT_COMPACTION_CONFIG)

        assert len(trail_calls) == 0 and len(history_calls) == 0, (
            "Recursion guard must apply to subagent_compaction channel too (D14)"
        )


# ── D15 — Compaction loops bounded at 30 iterations ──────────────────────────


class TestD15CompactionBounded:
    """D15: Compaction configs (COMPACTION_CONFIG, SUBAGENT_COMPACTION_CONFIG)
    have max_iterations=30."""

    def test_compaction_config_max_iterations(self):
        from configs.channels import COMPACTION_CONFIG, SUBAGENT_COMPACTION_CONFIG

        assert COMPACTION_CONFIG.max_iterations == 30, (
            f"COMPACTION_CONFIG.max_iterations must be 30 (D15), got "
            f"{COMPACTION_CONFIG.max_iterations}"
        )
        assert SUBAGENT_COMPACTION_CONFIG.max_iterations == 30, (
            f"SUBAGENT_COMPACTION_CONFIG.max_iterations must be 30 (D15), got "
            f"{SUBAGENT_COMPACTION_CONFIG.max_iterations}"
        )

    def test_compaction_loop_stops_at_max_iterations(self, db):
        """The compaction loop stops when max_iterations is reached (loop guard)."""
        from configs.channels import COMPACTION_CONFIG
        from services.message_processor import MessageProcessor
        from abilities._base import Ability

        # Patch dispatch to return a non-empty result (keeps looping) without
        # hitting AbilityRegistry (which raises KeyError on unknown tools).
        iteration_count = []

        original_should_stop = MessageProcessor._should_stop

        def _counting_should_stop(self):
            iteration_count.append(self.current_iteration)
            return original_should_stop(self)

        # send_messages returns tool_calls each time to keep looping,
        # dispatch stubbed so AbilityRegistry is never hit.
        from services.llm_service import LLMResponse
        tool_response = LLMResponse(
            text="", model="m", provider="p",
            tool_calls=[{"id": "t1", "name": "some_tool", "input": {}}],
        )

        ctx, fake = _patch_providers_calculate(0.0, send_response=tool_response)
        with ctx:
            with patch.object(MessageProcessor, "_should_stop", _counting_should_stop), \
                 patch.object(Ability, "dispatch", return_value="result"):
                result = MessageProcessor.process("hi", COMPACTION_CONFIG)

        # Loop stopped — current_iteration should not exceed max_iterations
        assert result == "", "Compaction loop must return '' at max_iterations cap (D15)"
        assert len(iteration_count) > 0, "Loop must have run"


# ── D16 — No _overflow_recovered_this_turn guard ─────────────────────────────


class TestD16NoOverflowRecoveredGuard:
    """D16: Each threshold fires at most once naturally (90 fires trail once if
    trail exists; 80 fires history once after reset). There is no
    _overflow_recovered_this_turn guard on the flat path — the two-threshold
    model handles this naturally."""

    def test_flat_loop_has_no_overflow_recovered_guard(self, db):
        """The flat _loop does NOT set or check _overflow_recovered_this_turn."""
        import inspect
        from services.message_processor import MessageProcessor

        loop_source = inspect.getsource(MessageProcessor._loop)
        assert "_overflow_recovered_this_turn" not in loop_source, (
            "_loop must NOT reference _overflow_recovered_this_turn — "
            "the two-threshold model handles this naturally (D16)"
        )

    def test_trail_compaction_fires_once_per_iteration_cycle(self, db):
        """After trail compaction resets iteration to 0, the next calculate()
        call is at 0.10 (below threshold) — compaction doesn't fire twice."""
        config = _make_flat_config()
        ctx, fake = _patch_providers_calculate([0.95, 0.10])
        with ctx:
            from services.message_processor import MessageProcessor
            trail_calls = []

            with patch.object(MessageProcessor, "_has_trail", return_value=True), \
                 patch.object(MessageProcessor, "_compact_trail",
                               lambda self: trail_calls.append(1) or True):
                MessageProcessor.process("hi", config)

        assert len(trail_calls) == 1, (
            "Trail compaction must fire exactly once (D16): "
            f"fired {len(trail_calls)} times"
        )
