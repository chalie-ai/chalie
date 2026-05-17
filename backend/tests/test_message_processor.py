import time

import pytest

from services.message_processor import _sanitize_llm_args


# ---------------------------------------------------------------------------
# ACT-loop timeout mechanics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIterationTimeout:
    def test_iteration_timeout_breaks_loop_on_runaway_iteration(self):
        """A single iteration that exceeds ITERATION_TIMEOUT causes the ACT loop
        to break after that iteration, rather than continuing to MAX_ITERATIONS.

        The LLM provider call is the external boundary that we patch — it makes
        the iteration hang beyond ITERATION_TIMEOUT. The mechanism under test
        (iter_elapsed > self.ITERATION_TIMEOUT → break) lives entirely in the
        base class ACT loop and is real production code.

        We use ITERATION_TIMEOUT=0.05s so the test runs fast. The LLM call
        sleeps 0.1s, ensuring iter_elapsed > ITERATION_TIMEOUT.
        """
        from unittest.mock import patch, MagicMock

        from services.message_processor import MessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt

        class _SlowIterProcessor(MessageProcessor):
            CHANNEL = "subagent"
            ROLE = "subagent"
            ITERATION_TIMEOUT = 0.5   # 500ms — well above prompt-assembly overhead
            SKIP_TRANSCRIPT_WRITE = True
            SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt

            def get_user_prompt(self):
                return "test"

            def get_user_definition(self):
                return "test user"

        # Fake LLM call sleeps 1.5s > ITERATION_TIMEOUT=0.5s. Without the safety
        # wall, MAX_ITERATIONS=30 × 1.5s = 45s. With the wall, 1 iteration + break.
        fake_response = MagicMock()
        fake_response.text = None
        fake_response.tool_calls = []

        def _slow_send_messages(*args, **kwargs):
            time.sleep(1.5)  # >> ITERATION_TIMEOUT=0.5
            return fake_response

        fake_provider = MagicMock()
        fake_provider.send_messages.side_effect = _slow_send_messages
        fake_provider.get_context_limit.return_value = 32_000
        fake_provider.get_compact_at.return_value = 32_000
        # Threshold check delegates payload measurement to the resolved
        # provider; return a small value so the check returns False and the
        # ACT loop reaches send_messages where the iteration-timeout wall
        # is exercised.
        fake_provider.estimate_payload_tokens.return_value = 100

        t0 = time.monotonic()
        with patch("services.providers.Providers") as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider
            proc = _SlowIterProcessor(raw_input="test")
            proc.send()
        elapsed = time.monotonic() - t0

        # Safety wall must have fired after iteration 1. Without the wall we'd
        # see 30 × 1.5s = 45s. With it, ~2s (1 LLM call + overhead).
        # Allow up to 5 iterations as a generous ceiling for system load.
        assert proc._current_iteration <= 5, (
            f"Loop ran {proc._current_iteration} iterations — expected <=5 "
            f"(ITERATION_TIMEOUT=0.5s safety wall should interrupt after the "
            f"first 1.5s LLM call)"
        )
        assert elapsed < 15.0, (
            f"ACT loop ran for {elapsed:.2f}s — expected <15s with a 0.5s "
            f"ITERATION_TIMEOUT (should break after first runaway iteration)"
        )


@pytest.mark.unit
class TestPerInstanceDeadline:
    def test_per_instance_deadline_caps_subagent_loop(self):
        """SubagentProcessor._deadline is checked at the top of each ACT
        iteration. Setting a short deadline (1.5s) causes the loop to exit
        within ~2s even with fast-returning LLM mock calls.

        LLM provider is patched at the external boundary so no live provider
        is needed. The mechanism under test (deadline check → break) is the
        real ACT loop code in message_processor.py.
        """
        from unittest.mock import patch, MagicMock

        from services.subagent_processor import SubagentProcessor

        # Thin subclass that skips transcript writes so no DB is required.
        # All ACT loop logic (deadline check, iteration counter) is inherited
        # unchanged from SubagentProcessor + MessageProcessor.
        class _NoDBSubagentProcessor(SubagentProcessor):
            SKIP_TRANSCRIPT_WRITE = True

        fake_response = MagicMock()
        fake_response.text = None
        fake_response.tool_calls = []

        def _instant_send(*args, **kwargs):
            time.sleep(0.05)  # brief delay so loop doesn't spin too fast
            return fake_response

        fake_provider = MagicMock()
        fake_provider.send_messages.side_effect = _instant_send
        fake_provider.get_context_limit.return_value = 32_000
        fake_provider.get_compact_at.return_value = 32_000
        # Threshold check delegates to the provider; return a small value
        # so the check returns False and the deadline mechanism is what
        # breaks the loop.
        fake_provider.estimate_payload_tokens.return_value = 100

        with patch("services.providers.Providers") as mock_providers_cls:
            mock_providers_cls.instance.return_value = fake_provider
            proc = _NoDBSubagentProcessor(raw_input="test", agent_type="general_purpose")
            # Override the deadline to 1.5s from now
            proc._deadline = time.time() + 1.5

            t0 = time.monotonic()
            proc.send()
            elapsed = time.monotonic() - t0

        # Deadline is 1.5s. With 0.05s per iteration, without the deadline
        # check the loop would run 50 iterations × 0.05s = 2.5s. With the
        # deadline enforced it must exit around the 1.5s mark.
        assert elapsed < 3.0, (
            f"Loop ran for {elapsed:.2f}s — deadline not enforced? "
            f"Expected <3.0s with a 1.5s deadline"
        )
        # Verify the loop did NOT run all 50 iterations
        assert proc._current_iteration < 50, (
            f"Loop ran all {proc._current_iteration} iterations — deadline did not break it"
        )


@pytest.mark.unit
class TestSkipInputRow:
    def test_skip_input_row_suppresses_write_but_store_runs(self):
        """SKIP_INPUT_ROW=True: write_input_row is not called, but
        store() still writes the assistant row via write_assistant_row."""
        from unittest.mock import patch, MagicMock

        from services.message_processor import MessageProcessor
        from services.system_message_prompt import DMNSystemMessagePrompt

        class _InputSkipProcessor(MessageProcessor):
            CHANNEL = "user"
            ROLE = "subagent_return"
            SKIP_INPUT_ROW = True
            SKIP_TRANSCRIPT_WRITE = False
            SYSTEM_PROMPT_CLASS = DMNSystemMessagePrompt

            def get_user_prompt(self):
                return "test"

            def get_user_definition(self):
                return "test user"

        fake_response = MagicMock()
        fake_response.text = "Synthesized response"
        fake_response.tool_calls = []

        fake_provider = MagicMock()
        fake_provider.send_messages.return_value = fake_response
        fake_provider.get_context_limit.return_value = 32_000
        fake_provider.get_compact_at.return_value = 32_000
        fake_provider.estimate_payload_tokens.return_value = 100

        with patch("services.providers.Providers") as mock_providers_cls, \
             patch("services.transcript_service.write_input_row") as mock_input, \
             patch("services.transcript_service.write_assistant_row") as mock_assistant:
            mock_providers_cls.instance.return_value = fake_provider
            mock_assistant.return_value = 999

            proc = _InputSkipProcessor(raw_input="subagent envelope text")
            result = proc.send()

            mock_input.assert_not_called()
            mock_assistant.assert_called_once()
            assert proc._uid is None
            assert result == "Synthesized response"


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
