"""Unit tests for the Ollama 413 → Stage 2 compaction wire.

Covers:
- ``OllamaService`` raises ``PayloadTooLargeError`` on HTTP 413 (no retry).
- ``FallbackLLMService`` re-raises ``NonRetryableError`` instead of falling
  back — sending the same oversize body to the secondary would just 413 again.
- ``OllamaService.get_context_limit()`` clamps ``:cloud`` models to
  ``OLLAMA_CLOUD_CONTEXT_CAP``; case-insensitive; non-cloud unchanged.
- ``MessageProcessor.send`` catches ``PayloadTooLargeError``, runs a single
  Stage 2 ACT restart, resumes the loop on success.
- A second ``PayloadTooLargeError`` in the same turn breaks to cap exit
  (``final_text=''``) without firing Stage 2 again.
- ``_run_stage2_act_restart`` clears ``_thinking_exploration`` so a large
  exploration block can't re-inflate ``user_body`` after recovery.

Mocking note: the LLM provider is the genuine external boundary for these
paths. Patching ``requests.post`` for OllamaService and substituting the
``Providers`` singleton for MessageProcessor mirrors the existing pattern
in ``test_token_counting.py`` — real branch logic, only the network call
is replaced.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

pytestmark = pytest.mark.unit


# ── OllamaService — 413 raise ──────────────────────────────────────────────

class TestOllama413RaisesPayloadTooLarge:
    def _make_413_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 413
        mock_response.text = '{"error":"Request Entity Too Large"}'
        mock_response.content = b'{"error":"Request Entity Too Large"}'
        http_err = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_err
        return mock_response

    def test_413_raises_payload_too_large_error(self):
        from services.llm_service import PayloadTooLargeError
        from services.ollama_service import OllamaService

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'kimi-k2.6:cloud',
            'max_retries': 0,
        })
        with patch('requests.post', return_value=self._make_413_response()):
            with pytest.raises(PayloadTooLargeError):
                svc.send_messages('sys', [{'role': 'user', 'content': 'hi'}])

    def test_413_does_not_retry(self):
        """413 must short-circuit the retry loop — same body would 413 again."""
        from services.llm_service import PayloadTooLargeError
        from services.ollama_service import OllamaService

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'kimi-k2.6:cloud',
            'max_retries': 3,
        })
        with patch('requests.post', return_value=self._make_413_response()) as mock_post:
            with pytest.raises(PayloadTooLargeError):
                svc.send_messages('sys', [{'role': 'user', 'content': 'hi'}])
            assert mock_post.call_count == 1

    def test_payload_too_large_is_non_retryable_subclass(self):
        from services.llm_service import NonRetryableError, PayloadTooLargeError
        assert issubclass(PayloadTooLargeError, NonRetryableError)


# ── FallbackLLMService — must NOT swallow PayloadTooLargeError ─────────────

class TestFallbackReRaisesNonRetryable:
    def test_fallback_does_not_swallow_payload_too_large(self):
        """If primary 413s, the fallback would 413 too — re-raise so
        MessageProcessor.send can run Stage 2 instead of silently swapping
        provider with the same oversize body."""
        from services.llm_service import FallbackLLMService, PayloadTooLargeError

        primary = MagicMock()
        primary.send_messages.side_effect = PayloadTooLargeError("primary 413")
        fallback = MagicMock()
        fallback.send_messages.return_value = MagicMock()  # would be wrong path

        svc = FallbackLLMService(primary, fallback)
        with pytest.raises(PayloadTooLargeError):
            svc.send_messages('sys', [{'role': 'user', 'content': 'hi'}])
        fallback.send_messages.assert_not_called()

    def test_fallback_still_used_for_generic_exceptions(self):
        """Sanity: non-NonRetryable failures still cascade to fallback."""
        from services.llm_service import FallbackLLMService

        primary = MagicMock()
        primary.send_messages.side_effect = ConnectionError("primary down")
        fallback_response = MagicMock()
        fallback = MagicMock()
        fallback.send_messages.return_value = fallback_response

        svc = FallbackLLMService(primary, fallback)
        result = svc.send_messages('sys', [{'role': 'user', 'content': 'hi'}])
        assert result is fallback_response
        fallback.send_messages.assert_called_once()


# ── OllamaService — get_context_limit clamping ─────────────────────────────

class TestOllamaContextLimitCloudClamp:
    @patch('requests.post')
    def test_cloud_model_clamped_to_cap(self, mock_post):
        from services.ollama_service import OLLAMA_CLOUD_CONTEXT_CAP, OllamaService

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            'model_info': {'kimi-k2.context_length': 262_144},
        }
        mock_post.return_value = mock_resp

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'kimi-k2.6:cloud',
        })
        assert svc.get_context_limit() == OLLAMA_CLOUD_CONTEXT_CAP

    @patch('requests.post')
    def test_cloud_match_is_case_insensitive(self, mock_post):
        from services.ollama_service import OLLAMA_CLOUD_CONTEXT_CAP, OllamaService

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            'model_info': {'kimi-k2.context_length': 262_144},
        }
        mock_post.return_value = mock_resp

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'KIMI-K2.6:CLOUD',
        })
        assert svc.get_context_limit() == OLLAMA_CLOUD_CONTEXT_CAP

    @patch('requests.post')
    def test_non_cloud_model_unchanged(self, mock_post):
        from services.ollama_service import OllamaService

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            'model_info': {'general.context_length': 131_072},
        }
        mock_post.return_value = mock_resp

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'gemma4:31b',
        })
        assert svc.get_context_limit() == 131_072

    @patch('requests.post')
    def test_cloud_model_below_cap_unchanged(self, mock_post):
        """A :cloud model that genuinely reports < cap is not raised."""
        from services.ollama_service import OllamaService

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            'model_info': {'tiny.context_length': 4096},
        }
        mock_post.return_value = mock_resp

        svc = OllamaService({
            'platform': 'ollama',
            'host': 'http://localhost:11434',
            'model': 'tiny:cloud',
        })
        assert svc.get_context_limit() == 4096


# ── MessageProcessor.send — real recovery branch ───────────────────────────

def _build_test_processor_class():
    """Concrete MessageProcessor subclass that runs send() without any I/O.

    - CHANNEL ≠ 'user' → thinking gate is a no-op.
    - SKIP_TRANSCRIPT_WRITE=True → no SQLite writes (transcript or assistant
      rows). _uid stays None and ToolRenderAndRecordService short-circuits.
    - _run_full_compaction overridden to return a fixed string so the
      overflow handler's LLM call is replaced; _handle_overflow runs real.
    """
    from services.message_processor import MessageProcessor

    class _TestProcessor(MessageProcessor):
        CHANNEL = 'test_413'
        ROLE = 'test_413'
        SKIP_TRANSCRIPT_WRITE = True
        LOG_LABEL = 'compaction'

        def __init__(self, raw_input='hello', metadata=None):
            super().__init__(raw_input, metadata)
            self.full_compaction_calls = 0

        def get_user_definition(self) -> str:
            return 'test channel'

        def get_user_prompt(self) -> str:
            return 'short body'

        def get_system_prompt(self) -> str:
            return 'system prompt'

        def get_tools(self) -> list[dict]:
            return []

        def _run_full_compaction(self, exclude_id=None) -> 'str | None':
            # Stub the LLM call inside _handle_overflow; everything around it is real.
            self.full_compaction_calls += 1
            return 'COMPACTED'

    return _TestProcessor


def _llm_response(text='final answer'):
    """Build a minimal LLMResponse stand-in (text, no tool_calls)."""
    from services.llm_service import LLMResponse
    return LLMResponse(
        text=text, model='test', provider='test',
        tokens_input=10, tokens_output=5,
    )


class TestMessageProcessorSend413Recovery:
    def _patch_providers(self, send_messages_side_effects, get_context_limit=200_000):
        """Replace Providers singleton with a fake. Returns the fake provider
        instance so tests can read .send_messages.call_args_list etc.
        """
        from services.providers import Providers
        fake = MagicMock()
        fake.send_messages.side_effect = send_messages_side_effects
        fake.get_context_limit.return_value = get_context_limit
        fake.get_compact_at.return_value = get_context_limit
        # Threshold check delegates to provider.estimate_payload_tokens; the
        # 413 path under test is independent of pre-send threshold checking,
        # so return a value well below compact_at to skip the overflow branch.
        fake.estimate_payload_tokens.return_value = 100
        return patch.object(Providers, 'instance', return_value=fake), fake

    def test_first_413_runs_overflow_handler_then_succeeds(self, db):
        from services.llm_service import PayloadTooLargeError
        processor_cls = _build_test_processor_class()
        proc = processor_cls()

        side_effects = [PayloadTooLargeError("simulated 413"), _llm_response()]
        ctx, fake = self._patch_providers(side_effects)
        with ctx:
            result = proc.send()

        assert result == 'final answer'
        assert proc._overflow_recovered_this_turn is True
        assert proc.full_compaction_calls == 1, "overflow handler must run exactly once"
        assert fake.send_messages.call_count == 2, "one 413 + one success"

    def test_second_413_breaks_to_cap_exit(self, db):
        from services.llm_service import PayloadTooLargeError
        processor_cls = _build_test_processor_class()
        proc = processor_cls()

        # Two 413s back-to-back. Second must NOT trigger overflow handler again.
        side_effects = [
            PayloadTooLargeError("first 413"),
            PayloadTooLargeError("second 413"),
        ]
        ctx, fake = self._patch_providers(side_effects)
        with ctx:
            result = proc.send()

        assert result == ''  # cap exit
        assert proc._overflow_recovered_this_turn is True
        assert proc.full_compaction_calls == 1, "overflow handler must run only once"
        assert fake.send_messages.call_count == 2

    def test_overflow_failure_breaks_to_cap_exit(self, db):
        from services.llm_service import PayloadTooLargeError
        processor_cls = _build_test_processor_class()

        class _FailingOverflow(processor_cls):
            def _run_full_compaction(self, exclude_id=None):
                self.full_compaction_calls += 1
                return None  # overflow failure

        proc = _FailingOverflow()
        side_effects = [PayloadTooLargeError("simulated 413")]
        ctx, fake = self._patch_providers(side_effects)
        with ctx:
            result = proc.send()

        assert result == ''
        assert proc.full_compaction_calls == 1
        assert fake.send_messages.call_count == 1, \
            "Provider must not be called again after overflow failure"

    def test_no_413_path_does_not_set_recovery_flag(self, db):
        processor_cls = _build_test_processor_class()
        proc = processor_cls()

        ctx, _ = self._patch_providers([_llm_response()])
        with ctx:
            result = proc.send()

        assert result == 'final answer'
        assert proc._overflow_recovered_this_turn is False
        assert proc.full_compaction_calls == 0


class TestProactiveThresholdLoopGuard:
    """Regression: threshold trip → compact → threshold STILL trips because
    static system_prompt + tools schema dominate. The loop must NOT run a
    second compaction; it must send anyway and rely on the wire-level 413
    path for genuine overflow.
    """

    def _patch_providers_persistent_overflow(self):
        """Threshold ALWAYS exceeds (estimate > compact_at), regardless of
        how many times it's checked. send_messages returns a clean response
        on first call so we can verify the loop reached the LLM call after
        one compaction."""
        from services.providers import Providers
        fake = MagicMock()
        fake.send_messages.side_effect = [_llm_response()]
        fake.get_context_limit.return_value = 200_000
        fake.get_compact_at.return_value = 200_000
        # Always over threshold — simulates static system+tools dominance.
        fake.estimate_payload_tokens.return_value = 500_000
        return patch.object(Providers, 'instance', return_value=fake), fake

    def test_persistent_threshold_trip_compacts_once_then_sends_anyway(self, db):
        """Compact_at is exceeded on every iteration, but compaction must
        run exactly once per turn. After the one-shot recovery, the loop
        proceeds to send_messages even though threshold still trips —
        otherwise the loop would spin forever recompacting."""
        processor_cls = _build_test_processor_class()
        proc = processor_cls()

        ctx, fake = self._patch_providers_persistent_overflow()
        with ctx:
            result = proc.send()

        assert result == 'final answer'
        # Compaction ran EXACTLY ONCE despite threshold tripping every iter.
        assert proc.full_compaction_calls == 1, \
            f"Expected 1 compaction, got {proc.full_compaction_calls} — runaway loop guard failed"
        # Provider was called once (after the one compaction).
        assert fake.send_messages.call_count == 1
        # Guard flag was set.
        assert proc._overflow_recovered_this_turn is True

    def test_proactive_overflow_returning_false_falls_through_to_send(self, db):
        """When _handle_overflow returns False on the proactive threshold path
        (e.g. SubagentProcessor with an empty trail at iter 0), the loop must
        NOT break to cap exit. Instead it sets the recovery flag and falls
        through to send_messages — the LLM call gets a chance to succeed,
        and the 413 path is the final safety net.

        Regression: scenario 121 was failing because subagent's first iter
        threshold check tripped (static system+tools dominate), the empty
        trail caused _handle_overflow to return False, and the loop broke
        to cap exit before the subagent could even call its tool.
        """
        processor_cls = _build_test_processor_class()

        class _NoOpOverflow(processor_cls):
            # Returns False without calling _run_full_compaction —
            # mimics SubagentProcessor's empty-trail short-circuit.
            def _handle_overflow(self):
                self.full_compaction_calls += 1
                return False

        proc = _NoOpOverflow()
        ctx, fake = self._patch_providers_persistent_overflow()
        with ctx:
            result = proc.send()

        assert result == 'final answer', \
            "Loop must reach send_messages and return its text, not cap-exit"
        assert proc.full_compaction_calls == 1, \
            "Overflow handler called once; subsequent threshold trips must skip it"
        assert fake.send_messages.call_count == 1, \
            "Provider must be called after the failed-overflow fallthrough"
        assert proc._overflow_recovered_this_turn is True, \
            "Recovery flag must be set even when overflow returned False"


class TestHandleOverflowClearsState:
    def test_exploration_cleared_on_overflow_success(self):
        """Large exploration text would re-inflate user_body after overflow.
        Confirm it's nulled when _handle_overflow succeeds."""
        processor_cls = _build_test_processor_class()
        proc = processor_cls()
        proc._thinking_exploration = "X" * 50_000  # large block

        result = proc._handle_overflow()
        assert result is True
        assert proc._thinking_exploration is None
        assert proc._act_trail == []
        assert proc._discovered_tools == []

    def test_exploration_preserved_when_overflow_fails(self):
        """If _run_full_compaction returns None, _handle_overflow returns False
        BEFORE the trail/exploration clears — exploration is preserved (no
        recovery happened, so the existing state must be intact)."""
        processor_cls = _build_test_processor_class()

        class _FailingOverflow(processor_cls):
            def _run_full_compaction(self, exclude_id=None):
                return None

        proc = _FailingOverflow()
        proc._thinking_exploration = "preserved"
        proc._act_trail = ['existing-row']

        result = proc._handle_overflow()
        assert result is False
        assert proc._thinking_exploration == "preserved"
        assert proc._act_trail == ['existing-row']
