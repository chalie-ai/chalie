"""Tests for LLM rate limit handling — RateLimitError, _call_with_retry, provider wrapping."""

import pytest

from services.llm_service import (
    RateLimitError,
    _call_with_retry,
    FallbackLLMService,
    LLMResponse,
)


pytestmark = pytest.mark.unit


class TestRateLimitError:

    def test_retry_after_and_provider_are_accessible(self):
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


class TestCallWithRetryRateLimit:

    def test_short_retry_after_retries_once_and_succeeds(self):
        """Rate limit with retry_after <= 10s retries once and returns success."""
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("429", retry_after=0.01, provider="openai")
            return "success"

        result = _call_with_retry(fn)
        assert result == "success"
        assert call_count == 2

    def test_no_retry_after_fails_fast(self):
        def fn():
            raise RateLimitError("429", retry_after=None, provider="gemini")

        with pytest.raises(RateLimitError):
            _call_with_retry(fn)

    def test_long_retry_after_fails_fast(self):
        def fn():
            raise RateLimitError("429", retry_after=42.0, provider="anthropic")

        with pytest.raises(RateLimitError):
            _call_with_retry(fn)

    def test_second_attempt_also_limited_raises(self):
        def fn():
            raise RateLimitError("429", retry_after=0.01, provider="openai")

        with pytest.raises(RateLimitError):
            _call_with_retry(fn)

    def test_generic_error_exponential_backoff_succeeds(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("transient error")
            return "success"

        result = _call_with_retry(fn, max_retries=2, backoff=0.01)
        assert result == "success"
        assert call_count == 3


class TestFallbackHandlesRateLimit:

    def test_fallback_returns_response_on_primary_rate_limit(self):
        class _FailingPrimary:
            def send_message(self, *args, **kwargs):
                raise RateLimitError("429", retry_after=30.0, provider="anthropic")

        class _OkFallback:
            def send_message(self, *args, **kwargs):
                return LLMResponse(text="fallback response", model="test")

        svc = FallbackLLMService(_FailingPrimary(), _OkFallback())
        result = svc.send_message("sys", "user")
        assert result.text == "fallback response"

    def test_fallback_returns_response_on_primary_generic_error(self):
        class _DownPrimary:
            def send_message(self, *args, **kwargs):
                raise ConnectionError("down")

        class _OkFallback:
            def send_message(self, *args, **kwargs):
                return LLMResponse(text="fallback", model="test")

        svc = FallbackLLMService(_DownPrimary(), _OkFallback())
        result = svc.send_message("sys", "user")
        assert result.text == "fallback"

    def test_both_fail_propagates_fallback_error(self):
        class _FailingPrimary:
            def send_message(self, *args, **kwargs):
                raise RateLimitError("429", provider="anthropic")

        class _FailingFallback:
            def send_message(self, *args, **kwargs):
                raise ConnectionError("fallback also down")

        svc = FallbackLLMService(_FailingPrimary(), _FailingFallback())
        with pytest.raises(ConnectionError):
            svc.send_message("sys", "user")
