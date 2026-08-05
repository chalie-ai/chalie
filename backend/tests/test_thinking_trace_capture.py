# Feature tests for thinking-trace capture in provider clients.
#
# Drives the REAL provider SDKs against a scripted local HTTP server
# (http.server on an ephemeral port returning canned JSON). No mocks of
# production code — the only stand-in is the server whose replies we wrote,
# which is the whole point: we are verifying the client's own parsing, not
# any provider's behaviour.
#
# ProviderApiResponse.thinking_block is populated by the clients.
#   - openai_compatible.py send() reads message.reasoning / reasoning_content
#     extras, else extracts <think>…</think> from content via _strip_think_blocks.
#   - ollama.py _parse_chat_response reads message.thinking.
#
# Five cases covered:
#   1. openai_compatible, reasoning_content field present
#   2. openai_compatible, think-block extraction from content
#   3. openai_compatible, reasoning field wins over think blocks
#   4. ollama, message.thinking field
#   5. no reasoning anywhere → thinking_block is None

import http.server
import json
import threading
from collections.abc import Iterator

import pytest

from configs.enums.thinking_level import ThinkingLevel
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.ollama import OllamaClient
from services.provider_api import ProviderApiRequest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ServerHandle:
    """A threaded HTTP server whose handler class is configured at construction."""

    def __init__(self, handler_cls: type[http.server.BaseHTTPRequestHandler]) -> None:
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self._port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._port

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------------------
# 1. openai_compatible client — reasoning_content field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAICompatibleReasoningContent:

    @pytest.fixture(autouse=True)
    def _server(self) -> Iterator[_ServerHandle]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "reasoning_content": "chain here",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        yield handle
        handle.stop()

    def test_reasoning_content_populates_thinking_block(self, _server: _ServerHandle) -> None:
        """Server returns choices[0].message with reasoning_content='chain here'
        and content='answer' → response.thinking_block == 'chain here',
        response.text == 'answer'."""
        client = OpenAICompatibleClient({
            "platform": "openai_compatible",
            "model": "test-model",
            "host": f"http://127.0.0.1:{_server.port}/v1",
            "api_key": "test",
        })
        resp = client.send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )
        assert resp.thinking_block == "chain here"
        assert resp.text == "answer"


# ---------------------------------------------------------------------------
# 2. openai_compatible client — think-block extraction from content
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAICompatibleThinkBlockExtraction:

    @pytest.fixture(autouse=True)
    def _server(self) -> Iterator[_ServerHandle]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<think>hidden</think>visible",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        yield handle
        handle.stop()

    def test_think_block_extracted_when_no_reasoning_field(self, _server: _ServerHandle) -> None:
        """Server returns content='<think>hidden</think>visible' with no
        reasoning field → thinking_block == 'hidden', text == 'visible'."""
        client = OpenAICompatibleClient({
            "platform": "openai_compatible",
            "model": "test-model",
            "host": f"http://127.0.0.1:{_server.port}/v1",
            "api_key": "test",
        })
        resp = client.send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )
        assert resp.thinking_block == "hidden"
        assert resp.text == "visible"


# ---------------------------------------------------------------------------
# 3. openai_compatible client — reasoning field wins over think blocks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAICompatibleReasoningFieldWins:

    @pytest.fixture(autouse=True)
    def _server(self) -> Iterator[_ServerHandle]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<think>hidden</think>visible",
                            "reasoning_content": "chain from field",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        yield handle
        handle.stop()

    def test_reasoning_field_takes_priority_over_think_blocks(self, _server: _ServerHandle) -> None:
        """Both reasoning field AND <think> in content are present →
        reasoning field wins: thinking_block == 'chain from field',
        text == 'visible'."""
        client = OpenAICompatibleClient({
            "platform": "openai_compatible",
            "model": "test-model",
            "host": f"http://127.0.0.1:{_server.port}/v1",
            "api_key": "test",
        })
        resp = client.send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )
        assert resp.thinking_block == "chain from field"
        assert resp.text == "visible"


# ---------------------------------------------------------------------------
# 4. ollama client — message.thinking field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaThinkingField:

    @pytest.fixture(autouse=True)
    def _server(self) -> Iterator[_ServerHandle]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "model": "test-model",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "thinking": "ollama chain",
                    },
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                    "prompt_eval_duration": 1000000,
                    "eval_duration": 1000000,
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        yield handle
        handle.stop()

    def test_thinking_field_populates_thinking_block(self, _server: _ServerHandle) -> None:
        """Server returns message with thinking='ollama chain' and
        content='hi' → thinking_block == 'ollama chain', text == 'hi'."""
        client = OllamaClient({
            "host": f"http://127.0.0.1:{_server.port}",
            "model": "test-model",
        })
        resp = client.send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )
        assert resp.thinking_block == "ollama chain"
        assert resp.text == "hi"


# ---------------------------------------------------------------------------
# 5. No reasoning anywhere — thinking_block is None
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoReasoningAnywhere:

    @pytest.fixture(autouse=True)
    def _server(self) -> Iterator[_ServerHandle]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "just an answer",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        yield handle
        handle.stop()

    def test_openai_compatible_no_reasoning_yields_none(self, _server: _ServerHandle) -> None:
        """openai_compatible client, no reasoning field and no think blocks
        → thinking_block is None."""
        client = OpenAICompatibleClient({
            "platform": "openai_compatible",
            "model": "test-model",
            "host": f"http://127.0.0.1:{_server.port}/v1",
            "api_key": "test",
        })
        resp = client.send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )
        assert resp.thinking_block is None
        assert resp.text == "just an answer"

    def test_ollama_no_reasoning_yields_none(self) -> None:
        """ollama client, no thinking field and no think blocks in content
        → thinking_block is None."""
        # The Ollama wire shape differs from the OpenAI one the class fixture
        # serves, so this test runs its own server with an Ollama-shaped body.
        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                # Drain the request body before replying — closing the
                # socket with unread bytes RSTs the connection, and the
                # SDK intermittently sees that before the response.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {
                    "model": "test-model",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "no reasoning here"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                    "prompt_eval_duration": 1000000,
                    "eval_duration": 1000000,
                }
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return
        handle = _ServerHandle(_Handler)
        try:
            client = OllamaClient({
                "host": f"http://127.0.0.1:{handle.port}",
                "model": "test-model",
            })
            resp = client.send(
                ProviderApiRequest(
                    system="",
                    messages=[{"role": "user", "content": "hi"}],
                    thinking_mode=ThinkingLevel.MEDIUM,
                )
            )
        finally:
            handle.stop()
        assert resp.thinking_block is None
        assert resp.text == "no reasoning here"
