"""
Pure-logic tests for the flat MessageProcessor — spec §2/§4a.

Only behaviours that are deterministic functions of plain inputs (no mocked
collaborators) live here:

  * ``_sanitize_llm_args`` — LLM-sentinel stripping (pure string transform).
  * ProcessorConfig hook surface — frozen-dataclass field contract (§2).
  * ``get_previous_messages`` under ``suppress_history`` — short-circuits to ''
    with no DB read (§4a).

The loop-control, transcript-row, thinking-gate, post-turn, and history-read
orchestration that used to live here was mock-collaborator wiring; that
behaviour is covered end-to-end by the scenario suite (scheduled-prompt act
loop, memory pre-act seed, compaction continuity over long history, and
multi-capability single turn).
"""

import pytest

from services.message_processor import _sanitize_llm_args


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_config(
    *,
    channel="dmn",
    role="proactive_thought",
    max_iterations=5,
    skip_transcript=True,
    skip_input_row=False,
    suppress_history=True,
    broadcast_to=None,
    memory_seed=False,
):
    """Return a minimal ProcessorConfig for use in flat-MP tests."""
    from services.processor_config import ProcessorConfig
    from tests.helpers import StubProcessorConfig

    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
        build_user_prompt=lambda _mp: "user body",
        build_user_definition=lambda _mp: "user definition",
        build_system_prompt=lambda _mp: "system",
        always_available=[],
        discoverable=[],
        blocked=frozenset(),
        max_iterations=max_iterations,
        skip_transcript=skip_transcript,
        skip_input_row=skip_input_row,
        suppress_history=suppress_history,
        broadcast_to=broadcast_to,
        memory_seed=memory_seed,
    )


# ---------------------------------------------------------------------------
# suppress_history short-circuit (§2/§4a)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuppressHistory:

    def test_suppress_history_returns_empty_string(self):
        """suppress_history short-circuits previous-messages to '' (no DB read). §2/§4a."""
        from services.message_processor import MessageProcessor
        config = _make_config(suppress_history=True)
        mp = object.__new__(MessageProcessor)
        mp.config = config
        result = mp.get_previous_messages()
        assert result == ""


# ---------------------------------------------------------------------------
# ProcessorConfig hook surface — frozen-dataclass field contract (§2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessorConfigHookSurface:

    def test_post_turn_is_only_optional_hook(self):
        """post_turn_hooks is the ONLY hook surface (no on_narration/
        on_tool_event/pre_act/process_attachments/overflow_strategy). §2 / §4.8."""
        from services.processor_config import ProcessorConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ProcessorConfig)}
        # post_turn_hooks must exist (the post_turn callable became a hook set, §4.8)
        assert "post_turn_hooks" in fields
        # None of the removed hooks should exist
        for removed in ("on_narration", "on_tool_event", "pre_act",
                        "process_attachments", "overflow_strategy"):
            assert removed not in fields, (
                f"ProcessorConfig must not have '{removed}' hook — "
                f"spec §2 / AC-32 prohibit it"
            )


# ---------------------------------------------------------------------------
# sanitize helper — pure string transform
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSanitizeLLMArgs:
    def test_observed_qwen_list_items(self):
        assert _sanitize_llm_args('<|"|milk<|"|') == 'milk'

    def test_well_formed_qwen_sentinels(self):
        assert _sanitize_llm_args('<|im_start|>foo<|im_end|>') == 'foo'

    def test_nested_dict_with_sentinel_items(self):
        result = _sanitize_llm_args({'items': ['<|"|milk<|"|', 'eggs']})
        assert result == {'items': ['milk', 'eggs']}

    def test_clean_inputs_untouched(self):
        value = {'action': 'add', 'items': ['milk']}
        assert _sanitize_llm_args(value) == {'action': 'add', 'items': ['milk']}

    def test_non_string_scalars_untouched(self):
        value = {'limit': 10, 'active': True, 'ratio': 0.5}
        assert _sanitize_llm_args(value) == {'limit': 10, 'active': True, 'ratio': 0.5}

    def test_empty_string_after_strip(self):
        result = _sanitize_llm_args({'x': '<|"|<|"|'})
        assert result == {'x': ''}
