"""Tests for LLM rate limit handling — RateLimitError, _call_with_retry, provider wrapping."""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from services.llm_service import (
    RateLimitError,
    _call_with_retry,
    FallbackLLMService,
    LLMResponse,
    AnthropicService,
    OpenAIService,
    GeminiService,
)


pytestmark = pytest.mark.unit


# ── RateLimitError construction ──────────────────────────────────────

class TestRateLimitError:

    def test_fields_set_correctly(self):
        err = RateLimitError("rate limited", retry_after=45.0, provider="anthropic")
        assert str(err) == "rate limited"
        assert err.retry_after == 45.0
        assert err.provider == "anthropic"

    def test_defaults_none(self):
        err = RateLimitError("limited")
        assert err.retry_after is None
        assert err.provider is None

    def test_is_exception(self):
        assert issubclass(RateLimitError, Exception)


# ── _call_with_retry with rate limits ────────────────────────────────

class TestCallWithRetryRateLimit:

    @patch('services.llm_service.time.sleep')
    def test_rate_limit_uses_retry_after(self, mock_sleep):
        """Rate limit with retry_after should sleep for that duration."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("429", retry_after=42.0, provider="openai")
            return "success"

        result = _call_with_retry(fn)
        assert result == "success"
        assert call_count == 2
        mock_sleep.assert_called_once_with(42.0)

    @patch('services.llm_service.time.sleep')
    def test_rate_limit_default_30s_when_no_retry_after(self, mock_sleep):
        """Rate limit without retry_after should default to 30s."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("429", retry_after=None, provider="gemini")
            return "success"

        result = _call_with_retry(fn)
        assert result == "success"
        mock_sleep.assert_called_once_with(30.0)

    @patch('services.llm_service.time.sleep')
    def test_rate_limit_caps_at_120s(self, mock_sleep):
        """retry_after values above 120s should be capped."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("429", retry_after=300.0, provider="anthropic")
            return "success"

        result = _call_with_retry(fn)
        assert result == "success"
        mock_sleep.assert_called_once_with(120.0)

    @patch('services.llm_service.time.sleep')
    def test_rate_limit_exhausted_raises(self, mock_sleep):
        """After 3 rate limit retries, should raise RateLimitError."""
        def fn():
            raise RateLimitError("429", retry_after=1.0, provider="openai")

        with pytest.raises(RateLimitError):
            _call_with_retry(fn)

        # 3 rate limit retries + sleeps
        assert mock_sleep.call_count == 3

    @patch('services.llm_service.time.sleep')
    def test_rate_limit_does_not_consume_generic_retries(self, mock_sleep):
        """Rate limit retries should not consume generic retry budget."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RateLimitError("429", retry_after=1.0, provider="openai")
            if call_count == 3:
                raise ValueError("other error")
            return "success"

        # Should handle 2 rate limits, then 1 generic error + 1 retry → success
        # max_retries=2 means up to 3 generic attempts
        # After 2 rate limits, call_count=3 raises ValueError (attempt 0),
        # then call_count=4 would succeed (attempt 1)
        call_count = 0

        def fn2():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RateLimitError("429", retry_after=1.0, provider="openai")
            if call_count == 3:
                raise ValueError("transient")
            return "success"

        result = _call_with_retry(fn2, max_retries=2)
        assert result == "success"
        assert call_count == 4

    @patch('services.llm_service.time.sleep')
    def test_generic_error_unchanged(self, mock_sleep):
        """Generic errors should still use short exponential backoff."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("transient error")
            return "success"

        result = _call_with_retry(fn, max_retries=2, backoff=1.0)
        assert result == "success"
        assert call_count == 3
        # backoff: 1.0 * 2^0 = 1.0, then 1.0 * 2^1 = 2.0
        assert mock_sleep.call_args_list[0][0][0] == 1.0
        assert mock_sleep.call_args_list[1][0][0] == 2.0


# ── FallbackLLMService propagation ───────────────────────────────────

class TestFallbackPropagatesRateLimit:

    def test_rate_limit_not_caught_by_fallback(self):
        """RateLimitError from primary should propagate, not trigger fallback."""
        primary = MagicMock()
        primary.send_message.side_effect = RateLimitError(
            "429", retry_after=30.0, provider="anthropic"
        )
        fallback = MagicMock()
        fallback.send_message.return_value = LLMResponse(
            text="fallback response", model="test"
        )

        svc = FallbackLLMService(primary, fallback)
        with pytest.raises(RateLimitError):
            svc.send_message("sys", "user")

        # Fallback should NOT have been called
        fallback.send_message.assert_not_called()

    def test_generic_error_still_uses_fallback(self):
        """Non-rate-limit errors should still fall through to fallback."""
        primary = MagicMock()
        primary.send_message.side_effect = ConnectionError("down")
        fallback = MagicMock()
        fallback.send_message.return_value = LLMResponse(
            text="fallback", model="test"
        )

        svc = FallbackLLMService(primary, fallback)
        result = svc.send_message("sys", "user")
        assert result.text == "fallback"
        fallback.send_message.assert_called_once()


# ── Provider wrapping ────────────────────────────────────────────────

class TestAnthropicWrapsRateLimit:

    @patch('services.llm_service._call_with_retry')
    @patch('services.llm_service._resolve_api_key', return_value='test-key')
    def test_anthropic_rate_limit_wrapped(self, mock_key, mock_retry):
        """Anthropic RateLimitError should be converted to our RateLimitError."""
        import anthropic

        # Create a mock anthropic.RateLimitError
        mock_response = MagicMock()
        mock_response.headers = {'retry-after': '45'}
        mock_response.status_code = 429
        anthropic_err = anthropic.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        # The _call() inner function catches anthropic.RateLimitError and re-raises as ours.
        # We need to test the _call() function directly, but it's inside send_message.
        # Instead, set _call_with_retry to call the fn() it receives:
        def call_fn(fn, **kwargs):
            return fn()

        mock_retry.side_effect = call_fn

        svc = AnthropicService({'api_key': 'test', 'model': 'test'})

        with patch('anthropic.Anthropic') as MockClient:
            MockClient.return_value.messages.create.side_effect = anthropic_err
            with pytest.raises(RateLimitError) as exc_info:
                svc.send_message("sys", "user")
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.retry_after == 45.0


class TestOpenAIWrapsRateLimit:

    @patch('services.llm_service._call_with_retry')
    @patch('services.llm_service._resolve_api_key', return_value='test-key')
    def test_openai_rate_limit_wrapped(self, mock_key, mock_retry):
        """OpenAI RateLimitError should be converted to our RateLimitError."""
        import openai as openai_mod

        mock_response = MagicMock()
        mock_response.headers = {'retry-after': '60'}
        mock_response.status_code = 429
        openai_err = openai_mod.RateLimitError(
            message="rate limited",
            response=mock_response,
            body=None,
        )

        def call_fn(fn, **kwargs):
            return fn()

        mock_retry.side_effect = call_fn

        svc = OpenAIService({'api_key': 'test', 'model': 'test'})

        with patch('openai.OpenAI') as MockClient:
            MockClient.return_value.chat.completions.create.side_effect = openai_err
            with pytest.raises(RateLimitError) as exc_info:
                svc.send_message("sys", "user")
            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 60.0
