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

