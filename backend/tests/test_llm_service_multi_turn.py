"""Tests for send_messages() multi-turn support across LLM providers."""
import pytest
from unittest.mock import patch, MagicMock
from services.llm_service import LLMResponse

pytestmark = pytest.mark.unit

SYSTEM_PROMPT = "You are a helpful assistant."
MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's 2+2?"},
]


class TestAnthropicMultiTurn:
    def _make_svc(self):
        from services.llm_service import AnthropicService
        svc = AnthropicService.__new__(AnthropicService)
        svc._config = {"api_key": "test-key"}
        svc.model = "claude-sonnet-4-20250514"
        svc.timeout = 120
        return svc

    def _mock_response(self, text="4"):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=text)]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=1)
        return mock_response

    def test_send_messages_basic(self):
        svc = self._make_svc()
        mock_response = self._mock_response()

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = mock_response
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

            call_kwargs = mock_get_client.return_value.messages.create.call_args
            assert call_kwargs.kwargs['messages'] == MESSAGES
            assert result.text == "4"
            assert result.provider == 'anthropic'

    def test_cache_prefix_adds_cache_control(self):
        svc = self._make_svc()
        mock_response = self._mock_response("ok")

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = mock_response
            svc.send_messages(SYSTEM_PROMPT, MESSAGES, cache_prefix=True)

            call_kwargs = mock_get_client.return_value.messages.create.call_args
            system_arg = call_kwargs.kwargs['system']
            assert isinstance(system_arg, list)
            assert system_arg[0].get('cache_control') == {"type": "ephemeral"}
            assert system_arg[0].get('text') == SYSTEM_PROMPT

    def test_no_cache_prefix_passes_string(self):
        svc = self._make_svc()
        mock_response = self._mock_response()

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = mock_response
            svc.send_messages(SYSTEM_PROMPT, MESSAGES, cache_prefix=False)

            call_kwargs = mock_get_client.return_value.messages.create.call_args
            assert call_kwargs.kwargs['system'] == SYSTEM_PROMPT

    def test_returns_llm_response(self):
        svc = self._make_svc()
        mock_response = self._mock_response("answer")

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = mock_response
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

        assert isinstance(result, LLMResponse)
        assert result.tokens_input == 10
        assert result.tokens_output == 1


class TestOpenAIMultiTurn:
    def _make_svc(self):
        from services.llm_service import OpenAIService
        svc = OpenAIService.__new__(OpenAIService)
        svc._config = {"api_key": "test-key"}
        svc.model = "gpt-4o"
        svc.timeout = 120
        svc.format = "text"
        return svc

    def _mock_response(self, text="4"):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=1)
        return mock_response

    def test_send_messages_prepends_system(self):
        svc = self._make_svc()
        mock_response = self._mock_response()

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

            call_kwargs = mock_get_client.return_value.chat.completions.create.call_args
            sent_messages = call_kwargs.kwargs['messages']
            assert sent_messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
            assert sent_messages[1:] == MESSAGES
            assert result.text == "4"

    def test_returns_llm_response(self):
        svc = self._make_svc()
        mock_response = self._mock_response("answer")

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

        assert isinstance(result, LLMResponse)
        assert result.provider == 'openai'
        assert result.tokens_input == 10
        assert result.tokens_output == 1

    def test_cache_prefix_ignored(self):
        svc = self._make_svc()
        mock_response = self._mock_response()

        with patch.object(svc, '_get_client') as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES, cache_prefix=True)

        assert result.text == "4"


class TestOllamaMultiTurn:
    def _make_svc(self):
        from services.ollama_service import OllamaService
        svc = OllamaService.__new__(OllamaService)
        svc.model = "qwen3:4b"
        svc.host = "http://localhost:11434"
        svc.keep_alive = "0"
        svc.temperature = 0.5
        svc.timeout = 60
        svc.format = "text"
        svc.max_retries = 2
        return svc

    def test_send_messages_uses_chat_endpoint(self):
        svc = self._make_svc()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "4"},
            "model": "qwen3:4b",
            "prompt_eval_count": 10,
            "eval_count": 1,
        }

        with patch('requests.post', return_value=mock_resp) as mock_post:
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

            call_args = mock_post.call_args
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert '/api/chat' in url

            body = call_args.kwargs.get('json') or call_args[1].get('json')
            assert body['messages'][0]['role'] == 'system'
            assert body['messages'][0]['content'] == SYSTEM_PROMPT
            assert result.text == "4"

    def test_send_messages_appends_user_messages(self):
        svc = self._make_svc()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "result"},
            "model": "qwen3:4b",
            "prompt_eval_count": 5,
            "eval_count": 2,
        }

        with patch('requests.post', return_value=mock_resp) as mock_post:
            svc.send_messages(SYSTEM_PROMPT, MESSAGES)

            body = mock_post.call_args.kwargs.get('json') or mock_post.call_args[1].get('json')
            sent = body['messages']
            assert sent[0] == {"role": "system", "content": SYSTEM_PROMPT}
            assert sent[1:] == MESSAGES

    def test_returns_llm_response(self):
        svc = self._make_svc()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "42"},
            "model": "qwen3:4b",
            "prompt_eval_count": 8,
            "eval_count": 3,
        }

        with patch('requests.post', return_value=mock_resp):
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)

        assert isinstance(result, LLMResponse)
        assert result.provider == 'ollama'
        assert result.tokens_input == 8
        assert result.tokens_output == 3


class TestFallbackMultiTurn:
    def test_delegates_to_primary(self):
        from services.llm_service import FallbackLLMService
        primary = MagicMock()
        fallback = MagicMock()
        svc = FallbackLLMService(primary, fallback)

        expected = LLMResponse(text="ok", model="test", provider="anthropic")
        primary.send_messages.return_value = expected

        result = svc.send_messages(SYSTEM_PROMPT, MESSAGES)
        primary.send_messages.assert_called_once_with(SYSTEM_PROMPT, MESSAGES, False)
        fallback.send_messages.assert_not_called()
        assert result is expected

    def test_falls_back_on_primary_error(self):
        from services.llm_service import FallbackLLMService
        primary = MagicMock()
        fallback = MagicMock()
        svc = FallbackLLMService(primary, fallback)

        primary.send_messages.side_effect = RuntimeError("primary down")
        expected = LLMResponse(text="fallback", model="test", provider="openai")
        fallback.send_messages.return_value = expected

        result = svc.send_messages(SYSTEM_PROMPT, MESSAGES, cache_prefix=True)
        fallback.send_messages.assert_called_once_with(SYSTEM_PROMPT, MESSAGES, True)
        assert result is expected

    def test_rate_limit_propagates_without_fallback(self):
        from services.llm_service import FallbackLLMService, RateLimitError
        primary = MagicMock()
        fallback = MagicMock()
        svc = FallbackLLMService(primary, fallback)

        primary.send_messages.side_effect = RateLimitError("429", provider="anthropic")

        with pytest.raises(RateLimitError):
            svc.send_messages(SYSTEM_PROMPT, MESSAGES)
        fallback.send_messages.assert_not_called()


class TestRefreshableMultiTurn:
    def test_delegates_to_inner_service(self):
        from services.llm_service import RefreshableLLMService
        svc = RefreshableLLMService.__new__(RefreshableLLMService)
        inner = MagicMock()
        svc._service = inner
        svc._agent_name = "test-agent"

        expected = LLMResponse(text="ok", model="test", provider="anthropic")
        inner.send_messages.return_value = expected

        with patch.object(svc, '_ensure_fresh'):
            result = svc.send_messages(SYSTEM_PROMPT, MESSAGES, cache_prefix=True)

        inner.send_messages.assert_called_once_with(SYSTEM_PROMPT, MESSAGES, True)
        assert result is expected
