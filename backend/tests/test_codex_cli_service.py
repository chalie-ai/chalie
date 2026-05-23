"""
Feature tests for the Codex CLI provider (TKT-588).

Split into two sections:
  1. Pure-function unit tests — helpers with no IO, no collaborators.
     Permitted under the zero-mocks rule because they are truly pure.
  2. Feature tests — real production stack (real factory, real detect call).
     The codex binary is not installed on this machine, so every feature test
     exercises the graceful-failure path; none attempt to spawn the subprocess.

What is NOT tested here:
  - Actual subprocess lifecycle (binary absent — would require a real install).
  - Session state after a successful handshake (requires running codex).
  - Mock subprocesses or fake RPC servers (forbidden by project rules).
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Pure-function unit tests
# ---------------------------------------------------------------------------

class TestExtractHelpers:
    """Pure helpers: _extract_user_text, _extract_thread_id, _extract_turn_id."""

    def test_extract_user_text(self):
        """Returns last user-role content; empty string when no user message."""
        from services.codex_cli_service import _extract_user_text
        # Last user message wins over earlier ones and assistant messages.
        messages = [
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'reply'},
            {'role': 'user', 'content': 'second'},
        ]
        assert _extract_user_text(messages) == 'second'
        # No user message → empty string.
        assert _extract_user_text([{'role': 'assistant', 'content': 'hi'}]) == ''
        assert _extract_user_text([]) == ''
        # Non-string content is stringified.
        assert _extract_user_text([{'role': 'user', 'content': 42}]) == '42'

    def test_extract_thread_and_turn_ids(self):
        """Priority chain and error branch for _extract_thread_id and _extract_turn_id."""
        from services.codex_cli_service import _extract_thread_id, _extract_turn_id

        # thread.id wins over all fallback keys when present.
        assert _extract_thread_id({'thread': {'id': 'winner'}, 'sessionId': 'loser', 'threadId': 'also-loser'}) == 'winner'
        # Fallback to thread.sessionId, then top-level sessionId, then threadId.
        assert _extract_thread_id({'thread': {'sessionId': 'sess-456'}}) == 'sess-456'
        assert _extract_thread_id({'sessionId': 'top-789'}) == 'top-789'
        assert _extract_thread_id({'threadId': 'tid-999'}) == 'tid-999'
        # No id at all raises.
        with pytest.raises(RuntimeError, match='thread/start returned no thread id'):
            _extract_thread_id({})

        # Turn id resolution and error.
        assert _extract_turn_id({'turn': {'id': 'turn-abc'}}) == 'turn-abc'
        assert _extract_turn_id({'turnId': 'turn-xyz'}) == 'turn-xyz'
        with pytest.raises(RuntimeError, match='turn/start returned no turn id'):
            _extract_turn_id({})


class TestProcessNotificationAndAuthFailure:
    """Pure helpers: _process_notification and _is_auth_failure."""

    def test_process_notification(self):
        """agentMessage text accumulates; turn/completed returns status; others are ignored."""
        from services.codex_cli_service import _process_notification

        parts = []
        # Non-agent item types are silently ignored.
        assert _process_notification(
            {'method': 'item/completed', 'params': {'item': {'type': 'commandExecution', 'text': 'ignored'}}},
            parts, 'turn-1',
        ) is None
        assert parts == []

        # agentMessage text accumulates across calls.
        _process_notification(
            {'method': 'item/completed', 'params': {'item': {'type': 'agentMessage', 'text': 'foo'}}},
            parts, 'turn-1',
        )
        _process_notification(
            {'method': 'item/completed', 'params': {'item': {'type': 'agentMessage', 'text': 'bar'}}},
            parts, 'turn-1',
        )
        assert ''.join(parts) == 'foobar'

        # turn/completed returns the status string; failed status is also propagated.
        assert _process_notification(
            {'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}},
            parts, 'turn-1',
        ) == 'completed'
        assert _process_notification(
            {'method': 'turn/completed', 'params': {'turn': {'status': 'failed'}}},
            parts, 'turn-1',
        ) == 'failed'

    def test_is_auth_failure(self):
        """Detects OAuth failure patterns in error text and stderr lines."""
        from services.codex_cli_service import _is_auth_failure

        # Various auth failure signals.
        assert _is_auth_failure('invalid_grant received', []) is True
        assert _is_auth_failure('HTTP 401 response', []) is True
        assert _is_auth_failure('', ['Error: 401 unauthorized']) is True
        assert _is_auth_failure('', ['user is not authenticated, please login']) is True

        # Non-auth errors and empty inputs are not false positives.
        assert _is_auth_failure('connection timed out', ['no output']) is False
        assert _is_auth_failure('', []) is False


# ---------------------------------------------------------------------------
# 2. Feature tests — real production stack, codex binary absent
# ---------------------------------------------------------------------------

class TestCheckCodexCliWhenNotInstalled:
    """check_codex_cli (quick check) reports unavailability when binary absent."""

    def test_returns_available_false_with_path_error(self):
        from services.codex_cli_service import check_codex_cli
        result = check_codex_cli('__nonexistent_codex_binary_chalie__')
        assert result['available'] is False
        assert result['models'] == []
        assert 'not found' in result['error'].lower()


class TestDetectCodexCliWhenNotInstalled:
    """detect_codex_cli (full check) reports unavailability when binary absent."""

    def test_returns_available_false_with_path_error(self):
        from services.codex_cli_service import detect_codex_cli
        result = detect_codex_cli('__nonexistent_codex_binary_chalie__')
        assert result['available'] is False
        assert result['models'] == []
        assert result['error'] is not None
        assert 'PATH' in result['error'] or 'not found' in result['error'].lower()


class TestLlmServiceFactoryCodexCliPlatform:
    """_build_service dispatches 'codex_cli' to CodexCliProviderService."""

    def test_factory_returns_codex_cli_service_instance(self):
        """create_llm_service with platform='codex_cli' returns a CodexCliProviderService.

        This does NOT spawn any subprocess — construction only creates the service
        object and touches the module-level session cache; no I/O occurs until
        send_messages() is called.
        """
        from services.llm_service import _build_service
        from services.codex_cli_service import CodexCliProviderService
        config = {'platform': 'codex_cli', 'model': 'o4-mini'}
        service = _build_service(config)
        assert isinstance(service, CodexCliProviderService)
        assert service.model == 'o4-mini'

    def test_codex_cli_service_reports_correct_context_limit(self):
        """CodexCliProviderService.get_context_limit returns 128 000."""
        from services.codex_cli_service import CodexCliProviderService
        svc = CodexCliProviderService({'platform': 'codex_cli', 'model': 'o4-mini'})
        assert svc.get_context_limit() == 128_000

    def test_codex_cli_service_wrong_platform_raises(self):
        """CodexCliProviderService rejects a config with the wrong platform."""
        from services.codex_cli_service import CodexCliProviderService
        with pytest.raises(ValueError, match="does not support platform"):
            CodexCliProviderService({'platform': 'ollama', 'model': 'llama3'})
