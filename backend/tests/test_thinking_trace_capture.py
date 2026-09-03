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

import http.server
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from configs.enums.thinking_level import ThinkingLevel
from contracts.provider_client import ProviderClient
from services.llm_clients.openai_compatible import OpenAICompatibleClient
from services.llm_clients.ollama import OllamaClient
from services.provider_api import ProviderApiRequest, ProviderApiResponse

pytestmark = pytest.mark.unit


@contextmanager
def _serving(body: dict[str, object]) -> Iterator[int]:
    """Serve *body* as JSON to every POST on an ephemeral port; yield the port."""
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            # Drain the request body before replying — closing the socket with
            # unread bytes RSTs the connection, and the SDK intermittently
            # sees that before the response.
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()


def _openai_wire(message: dict[str, object]) -> dict[str, object]:
    """A chat.completion body whose choices[0].message carries *message*."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _ollama_wire(message: dict[str, object]) -> dict[str, object]:
    """An /api/chat body whose message carries *message* (its own wire shape)."""
    return {
        "model": "test-model",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", **message},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
        "prompt_eval_duration": 1000000,
        "eval_duration": 1000000,
    }


def _openai_client(port: int) -> ProviderClient:
    return OpenAICompatibleClient({
        "platform": "openai_compatible",
        "model": "test-model",
        "host": f"http://127.0.0.1:{port}/v1",
        "api_key": "test",
    })


def _ollama_client(port: int) -> ProviderClient:
    return OllamaClient({"host": f"http://127.0.0.1:{port}", "model": "test-model"})


_PLATFORMS = {"openai_compatible": (_openai_wire, _openai_client),
              "ollama": (_ollama_wire, _ollama_client)}


def _capture(platform: str, message: dict[str, object]) -> ProviderApiResponse:
    """Serve *message* on *platform*'s wire shape and send one request at it."""
    wire, make_client = _PLATFORMS[platform]
    with _serving(wire(message)) as port:
        return make_client(port).send(
            ProviderApiRequest(
                system="",
                messages=[{"role": "user", "content": "hi"}],
                thinking_mode=ThinkingLevel.MEDIUM,
            )
        )


@pytest.mark.parametrize(
    ("platform", "message", "expected_thinking", "expected_text"),
    [
        pytest.param(
            "openai_compatible",
            {"content": "answer", "reasoning_content": "chain here"},
            "chain here", "answer",
            id="openai-reasoning-content-field",
        ),
        pytest.param(
            "openai_compatible",
            {"content": "<think>hidden</think>visible"},
            "hidden", "visible",
            id="openai-think-block-extracted-from-content",
        ),
        pytest.param(
            "openai_compatible",
            {"content": "<think>hidden</think>visible", "reasoning_content": "chain from field"},
            "chain from field", "visible",
            id="openai-reasoning-field-wins-over-think-blocks",
        ),
        pytest.param(
            "openai_compatible",
            {"content": "just an answer"},
            None, "just an answer",
            id="openai-no-reasoning-yields-none",
        ),
        pytest.param(
            "ollama",
            {"content": "hi", "thinking": "ollama chain"},
            "ollama chain", "hi",
            id="ollama-thinking-field",
        ),
        pytest.param(
            "ollama",
            {"content": "no reasoning here"},
            None, "no reasoning here",
            id="ollama-no-reasoning-yields-none",
        ),
    ],
)
def test_thinking_block_capture(
    platform: str,
    message: dict[str, object],
    expected_thinking: str | None,
    expected_text: str,
) -> None:
    resp = _capture(platform, message)
    assert resp.thinking_block == expected_thinking
    assert resp.text == expected_text
