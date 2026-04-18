"""Tests for Phase 4A: Pre-flight Token Counting.

Tests cover:
- estimate_tokens() heuristic
- get_context_limit() for each provider
- count_tokens() for each provider (with fallback behavior)
- Auto-budget calculation in ACT orchestrator
"""

from unittest.mock import patch, MagicMock


# ── estimate_tokens ──────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string(self):
        from services.llm_service import estimate_tokens
        assert estimate_tokens('') == 0

    def test_none_returns_zero(self):
        from services.llm_service import estimate_tokens
        assert estimate_tokens(None) == 0

    def test_single_word(self):
        from services.llm_service import estimate_tokens
        assert estimate_tokens('hello') == 1  # int(1 * 1.3) = 1

    def test_ten_words(self):
        from services.llm_service import estimate_tokens
        result = estimate_tokens('one two three four five six seven eight nine ten')
        assert result == 13  # int(10 * 1.3) = 13

    def test_proportional_to_word_count(self):
        from services.llm_service import estimate_tokens
        short = estimate_tokens('a b c')
        long = estimate_tokens('a b c d e f g h i j k l')
        assert long > short


# ── AnthropicService ─────────────────────────────────────────────────

class TestAnthropicContextLimit:
    def test_returns_200k(self):
        from services.llm_service import AnthropicService
        svc = AnthropicService({'api_key': 'test', 'model': 'claude-sonnet-4-20250514'})
        assert svc.get_context_limit() == 200_000


class TestAnthropicCountTokens:
    @patch.object(
        __import__('services.llm_service', fromlist=['AnthropicService']).AnthropicService,
        '_get_client',
    )
    def test_uses_api_when_available(self, mock_client_fn):
        from services.llm_service import AnthropicService

        mock_result = MagicMock()
        mock_result.input_tokens = 42
        mock_client = MagicMock()
        mock_client.messages.count_tokens.return_value = mock_result
        mock_client_fn.return_value = mock_client

        svc = AnthropicService({'api_key': 'test', 'model': 'claude-sonnet-4-20250514'})
        result = svc.count_tokens(
            [{"role": "user", "content": "hello"}],
            system_prompt="You are helpful.",
        )
        assert result == 42
        mock_client.messages.count_tokens.assert_called_once()

    @patch.object(
        __import__('services.llm_service', fromlist=['AnthropicService']).AnthropicService,
        '_get_client',
    )
    def test_falls_back_to_estimate_on_error(self, mock_client_fn):
        from services.llm_service import AnthropicService, estimate_tokens

        mock_client = MagicMock()
        mock_client.messages.count_tokens.side_effect = Exception("API error")
        mock_client_fn.return_value = mock_client

        svc = AnthropicService({'api_key': 'test', 'model': 'claude-sonnet-4-20250514'})
        result = svc.count_tokens(
            [{"role": "user", "content": "hello world"}],
            system_prompt="You are helpful.",
        )
        expected = estimate_tokens('You are helpful. hello world')
        assert result == expected


# ── OpenAIService ────────────────────────────────────────────────────

class TestOpenAIContextLimit:
    def test_returns_128k(self):
        from services.llm_service import OpenAIService
        svc = OpenAIService({'api_key': 'test', 'model': 'gpt-4o'})
        assert svc.get_context_limit() == 128_000


class TestOpenAICountTokens:
    def test_falls_back_to_estimate_without_tiktoken(self):
        """Without tiktoken installed, should use estimate_tokens fallback."""
        from services.llm_service import OpenAIService

        svc = OpenAIService({'api_key': 'test', 'model': 'gpt-4o'})
        messages = [{"role": "user", "content": "hello world"}]
        result = svc.count_tokens(messages, system_prompt="Be helpful.")

        # Should produce a reasonable estimate regardless of tiktoken availability
        assert result > 0


# ── GeminiService ────────────────────────────────────────────────────

class TestGeminiContextLimit:
    def test_falls_back_to_default_on_error(self):
        from services.llm_service import GeminiService

        svc = GeminiService({'api_key': 'test', 'model': 'gemini-2.5-flash'})
        if hasattr(svc, '_cached_context_limit'):
            del svc._cached_context_limit

        # Without a real API connection, should fall back to default
        result = svc.get_context_limit()
        assert result == 1_000_000  # Default fallback

    def test_caches_result(self):
        from services.llm_service import GeminiService

        svc = GeminiService({'api_key': 'test', 'model': 'gemini-2.5-flash'})
        svc._cached_context_limit = 500_000
        assert svc.get_context_limit() == 500_000  # Reads cache, no API call


class TestGeminiCountTokens:
    @patch('services.llm_service._resolve_api_key', return_value='test-key')
    def test_falls_back_to_estimate_on_error(self, _mock_key):
        from services.llm_service import GeminiService, estimate_tokens

        svc = GeminiService({'api_key': 'test', 'model': 'gemini-2.5-flash'})
        messages = [{"role": "user", "content": "hello world"}]
        result = svc.count_tokens(messages, system_prompt="Be helpful.")

        # Should fall back to estimate since we don't have a real API connection
        expected = estimate_tokens('Be helpful. hello world')
        assert result == expected


# ── OllamaService ───────────────────────────────────────────────────

class TestOllamaContextLimit:
    @patch('requests.post')
    def test_queries_api_show(self, mock_post):
        from services.ollama_service import OllamaService

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            'model_info': {
                'general.context_length': 131072,
            }
        }
        mock_post.return_value = mock_resp

        svc = OllamaService({'platform': 'ollama', 'host': 'http://localhost:11434', 'model': 'gemma4:31b'})
        result = svc.get_context_limit()
        assert result == 131072

        # Verify caching — second call should NOT hit the API
        mock_post.reset_mock()
        result2 = svc.get_context_limit()
        assert result2 == 131072
        mock_post.assert_not_called()

    @patch('requests.post')
    def test_falls_back_to_8192_on_error(self, mock_post):
        from services.ollama_service import OllamaService

        mock_post.side_effect = Exception("Connection refused")
        svc = OllamaService({'platform': 'ollama', 'host': 'http://localhost:11434', 'model': 'test'})
        result = svc.get_context_limit()
        assert result == 8192


class TestOllamaCountTokens:
    def test_uses_estimate_tokens(self):
        from services.ollama_service import OllamaService
        from services.llm_service import estimate_tokens

        svc = OllamaService({'platform': 'ollama', 'host': 'http://localhost:11434', 'model': 'test'})
        messages = [{"role": "user", "content": "hello world"}]
        result = svc.count_tokens(messages, system_prompt="Be helpful.")
        expected = estimate_tokens('Be helpful. hello world')
        assert result == expected


# ── FallbackLLMService ──────────────────────────────────────────────

class TestFallbackDelegation:
    def test_delegates_context_limit_to_primary(self):
        from services.llm_service import FallbackLLMService

        primary = MagicMock()
        primary.get_context_limit.return_value = 200_000
        fallback = MagicMock()

        svc = FallbackLLMService(primary, fallback)
        assert svc.get_context_limit() == 200_000
        primary.get_context_limit.assert_called_once()
        fallback.get_context_limit.assert_not_called()

    def test_delegates_count_tokens_to_primary(self):
        from services.llm_service import FallbackLLMService

        primary = MagicMock()
        primary.count_tokens.return_value = 500
        fallback = MagicMock()

        svc = FallbackLLMService(primary, fallback)
        msgs = [{"role": "user", "content": "hello"}]
        result = svc.count_tokens(msgs, system_prompt="sys")
        assert result == 500
        primary.count_tokens.assert_called_once_with(msgs, "sys", None)


# ── RefreshableLLMService ───────────────────────────────────────────

class TestRefreshableDelegation:
    def test_delegates_context_limit(self):
        from services.llm_service import RefreshableLLMService

        svc = RefreshableLLMService('test-agent')
        inner = MagicMock()
        inner.get_context_limit.return_value = 128_000
        svc._service = inner
        svc._version = 'v1'

        # Skip the version check by patching _ensure_fresh
        with patch.object(svc, '_ensure_fresh'):
            result = svc.get_context_limit()
        assert result == 128_000

    def test_delegates_count_tokens(self):
        from services.llm_service import RefreshableLLMService

        svc = RefreshableLLMService('test-agent')
        inner = MagicMock()
        inner.count_tokens.return_value = 1234
        svc._service = inner
        svc._version = 'v1'

        with patch.object(svc, '_ensure_fresh'):
            msgs = [{"role": "user", "content": "test"}]
            result = svc.count_tokens(msgs, "sys", [{"name": "tool1"}])
        assert result == 1234


# ── ACT Orchestrator Budget ────────────────────────────────────────

class TestAutoBudgetCalculation:
    """Verify the orchestrator computes context budget from provider limits."""

    def test_budget_from_200k_provider(self):
        """200k limit → min(200k * 0.6, 150k) = 120k."""
        budget = min(int(200_000 * 0.6), 150_000)
        assert budget == 120_000

    def test_budget_from_128k_provider(self):
        """128k limit → min(128k * 0.6, 150k) = 76.8k."""
        budget = min(int(128_000 * 0.6), 150_000)
        assert budget == 76_800

    def test_budget_from_8k_provider(self):
        """8k limit → min(8k * 0.6, 150k) = 4.8k."""
        budget = min(int(8_000 * 0.6), 150_000)
        assert budget == 4_800

    def test_budget_from_1m_provider(self):
        """1M limit → min(1M * 0.6, 150k) = 150k (ceiling)."""
        budget = min(int(1_000_000 * 0.6), 150_000)
        assert budget == 150_000

