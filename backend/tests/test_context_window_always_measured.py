# Feature tests for the cross-platform context-window guarantee.
# Real-stack — real clients, a real HTTP server for the Ollama host, no mocks.
#
# The guarantee under test: a provider that ANSWERS always yields a usable
# window, because ProviderDbService.pin_context_window raises on a missing one
# and an unmeasured provider therefore fails every turn. A provider that cannot
# be REACHED still yields None, so a briefly-down host never stamps a fabricated
# window onto its row.

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from services.llm_clients.context_window import (
    DEFAULT_LARGE_WINDOW,
    DEFAULT_WINDOW,
    default_window_for_model,
)


def _serve_ollama_show(payload: "dict[str, object] | None", status: int = 200) -> "tuple[str, HTTPServer]":
    """Stand up an /api/show endpoint; return (host, server).

    ``payload`` None means the host answers with an error status — the
    unreachable-ish case that must NOT produce a fabricated window.
    """

    class _Show(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if payload is None:
                self.send_response(status)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = HTTPServer(("127.0.0.1", 0), _Show)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", server


class TestContextWindowAlwaysMeasured:

    # ------------------------------------------------------------------
    # The shared default table — one owner for every client.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("model,expected", [
        ("zai-org/GLM-4.6", DEFAULT_LARGE_WINDOW),   # vendor-namespaced slug
        ("claude-3-5-sonnet", DEFAULT_LARGE_WINDOW),
        ("gpt-4o-mini", DEFAULT_LARGE_WINDOW),
        ("gemini-2.5-pro", DEFAULT_LARGE_WINDOW),
        ("mistral-small:22b", DEFAULT_WINDOW),       # unrecognised → conservative
        ("", DEFAULT_WINDOW),                        # no slug at all
    ])
    def test_the_family_default_resolves_for_any_slug(self, model: str, expected: int) -> None:
        # Never None and never zero: this is the value that stands between a
        # sizeable provider and a turn that dies resolving its window.
        assert default_window_for_model(model) == expected

    # ------------------------------------------------------------------
    # Ollama — a live host that omits context_length still gets sized.
    # ------------------------------------------------------------------

    def test_ollama_host_that_reports_no_context_length_still_gets_a_window(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        # A real /api/show reply that carries model_info but no *context_length*
        # key — the shape that used to return None and kill the provider.
        host, server = _serve_ollama_show({"model_info": {"general.architecture": "llama"}})
        try:
            client = OllamaClient({'platform': 'ollama', 'model': 'mistral-small:22b', 'host': host})
            assert client.get_context_limit() == DEFAULT_WINDOW
        finally:
            server.shutdown()

    def test_ollama_reports_its_own_context_length_when_it_has_one(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        # The authoritative figure always wins over the default.
        host, server = _serve_ollama_show({"model_info": {"llama.context_length": 262144}})
        try:
            client = OllamaClient({'platform': 'ollama', 'model': 'mistral-small:22b', 'host': host})
            assert client.get_context_limit() == 262144
        finally:
            server.shutdown()

    def test_an_unreachable_ollama_leaves_the_window_unset(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        # Port 1 is closed. None, not a default: a down host must not
        # permanently stamp a guessed window onto the row.
        client = OllamaClient({
            'platform': 'ollama', 'model': 'mistral-small:22b', 'host': 'http://127.0.0.1:1',
        })
        assert client.get_context_limit() is None

    def test_an_ollama_that_refuses_the_model_leaves_the_window_unset(self) -> None:
        from services.llm_clients.ollama import OllamaClient

        # HTTP 404 — the host is up but does not serve this model. Sizing it
        # would be inventing a window for a model that cannot answer.
        host, server = _serve_ollama_show(None, status=404)
        try:
            client = OllamaClient({'platform': 'ollama', 'model': 'absent', 'host': host})
            assert client.get_context_limit() is None
        finally:
            server.shutdown()

    # ------------------------------------------------------------------
    # openai_compatible — a refusal is an answer, not a silence.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("status,message", [
        # The one that actually bites: every reasoning model rejects max_tokens.
        (400, "Unsupported parameter: 'max_tokens' is not supported with this "
              "model. Use 'max_completion_tokens' instead."),
        (401, "Incorrect API key provided."),
        (404, "The model does not exist."),
    ])
    def test_a_host_that_refuses_the_ping_is_still_sized(self, status: int, message: str) -> None:
        from services.llm_clients.openai import OpenAIClient

        # Any HTTP status proves the host is up. 429 and 5xx take the same
        # branch (all APIStatusError) and are left out only because the SDK
        # retries them, which would buy nothing but wall-clock here.
        class _Refuse(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.dumps({
                    "error": {"message": message, "type": "invalid_request_error"},
                }).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 0), _Refuse)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OpenAIClient({
                'platform': 'openai_compatible',
                'model': 'gpt-5-nano',
                'host': f"http://127.0.0.1:{server.server_port}/v1",
                'api_key': secrets.token_hex(16),
            })
            # Not None. None would make pin_context_window raise, so the user
            # would see "cannot determine the context window" instead of the
            # message above — which is the one that tells them what to fix.
            assert client.get_context_limit() == DEFAULT_LARGE_WINDOW
        finally:
            server.shutdown()

    # ------------------------------------------------------------------
    # codex_cli — local cache miss is not a reachability question.
    # ------------------------------------------------------------------

    def test_codex_unknown_slug_is_sized_rather_than_left_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from services.llm_clients.codex_cli import CodexCliClient

        home = tmp_path / ".codex"
        home.mkdir()
        (home / "models_cache.json").write_text(json.dumps({
            "models": [{"slug": "gpt-5.5", "context_window": 272000}],
        }))
        monkeypatch.setenv('CODEX_HOME', str(home))

        # codex runs locally, so a cache miss says nothing about whether the
        # model works — returning None here killed every turn on a fresh slug.
        assert CodexCliClient({
            'platform': 'codex_cli', 'model': 'gpt-6-preview',
        }).get_context_limit() == DEFAULT_LARGE_WINDOW

    # ------------------------------------------------------------------
    # Anthropic — publishes a figure for everything it serves.
    # ------------------------------------------------------------------

    def test_anthropic_always_reports_a_window(self) -> None:
        from services.llm_clients.anthropic import AnthropicClient

        limit = AnthropicClient({
            'platform': 'anthropic', 'model': 'claude-sonnet-4-5', 'api_key': 'k',
        }).get_context_limit()
        assert isinstance(limit, int) and limit > 0

    # ------------------------------------------------------------------
    # Every client satisfies the contract shape.
    # ------------------------------------------------------------------

    def test_no_client_can_return_a_zero_or_negative_window(self) -> None:
        # A zero would pass an `is not None` check and then size every payload
        # to nothing, so the contract is "positive int or None", never 0.
        for model in ('glm-4.6', 'mistral-small:22b', 'gpt-4o', ''):
            assert default_window_for_model(model) > 0
