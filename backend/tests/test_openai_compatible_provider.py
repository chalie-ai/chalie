# Feature tests for the openai_compatible LLM platform.
# Real-stack — no mocks of production code.

import json
import secrets
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast

import pytest
from flask.testing import FlaskClient

import services.vault_service as _vault_mod
from services.vault_service import _vault_state, get_vault_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unlock_vault(password: str = "test-password") -> None:
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


def _unwrap_listing(body: "dict[str, object]") -> "list[dict[str, object]]":
    """Assert the success listing envelope shape and return the result array."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("list[dict[str, object]]", body["result"])


def _unwrap_single(body: "dict[str, object]") -> "dict[str, object]":
    """Assert the success single-resource envelope shape and return the result dict."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("dict[str, object]", body["result"])


def _unwrap_error(body: "dict[str, object]") -> str:
    """Assert the error envelope shape and return the error message."""
    assert body.get("success") is False
    assert body.get("result") == []
    return cast(str, body["error"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAICompatibleProvider:

    @pytest.fixture(autouse=True)
    def _reset_vault_state(self) -> object:
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None
        yield
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

    # ------------------------------------------------------------------
    # 1. Provider is stored and listed back correctly
    # ------------------------------------------------------------------

    def test_post_openai_compatible_stores_and_lists_provider(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _, _store = authed_client
        _unlock_vault()

        resp = client.post('/api/providers/-1', json={
            'name': 'minimax-m2',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': secrets.token_hex(16),
        })

        assert resp.status_code == 201, resp.get_data(as_text=True)
        created = _unwrap_single(cast("dict[str, object]", resp.get_json()))
        assert created['platform'] == 'openai_compatible'
        assert created['model'] == 'MiniMax-M2'
        assert created['host'] == 'https://api.minimax.io/v1'
        assert 'api_key' not in created  # write-only: never on the read shape

        list_resp = client.get('/api/providers/all')
        assert list_resp.status_code == 200
        providers = _unwrap_listing(cast("dict[str, object]", list_resp.get_json()))
        names = [p['name'] for p in providers]
        assert 'minimax-m2' in names

        stored = next(p for p in providers if p['name'] == 'minimax-m2')
        assert stored['platform'] == 'openai_compatible'
        assert stored['host'] == 'https://api.minimax.io/v1'
        assert stored['model'] == 'MiniMax-M2'

    # ------------------------------------------------------------------
    # 2. API key round-trips through vault encryption
    # ------------------------------------------------------------------

    def test_api_key_decrypts_correctly_from_db(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _, _store = authed_client
        _unlock_vault()

        test_key = secrets.token_hex(16)
        resp = client.post('/api/providers/-1', json={
            'name': 'minimax-vault-roundtrip',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': test_key,
        })
        assert resp.status_code == 201
        provider_id = cast(int, _unwrap_single(cast("dict[str, object]", resp.get_json()))['id'])

        # Read back via the DB service (decrypts the key)
        from services.provider_db_service import ProviderDbService
        svc = ProviderDbService()
        provider = svc.get_provider_by_id(provider_id)

        assert provider is not None
        assert provider['api_key'] == test_key
        assert provider['platform'] == 'openai_compatible'
        assert provider['host'] == 'https://api.minimax.io/v1'

    # ------------------------------------------------------------------
    # 3. build_client returns OpenAIClient for openai_compatible
    # ------------------------------------------------------------------

    def test_build_client_returns_openai_client(self) -> None:
        from services.llm_clients.factory import build_client
        from services.llm_clients.openai import OpenAIClient

        config: dict[str, object] = {
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': secrets.token_hex(16),
        }

        client = build_client(config)

        assert isinstance(client, OpenAIClient)

    # ------------------------------------------------------------------
    # 4. _get_client() wires base_url from the host field
    # ------------------------------------------------------------------

    def test_get_client_uses_host_as_base_url(self) -> None:
        from services.llm_clients.openai import OpenAIClient

        config: dict[str, object] = {
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': secrets.token_hex(16),
        }

        client = OpenAIClient(config)
        openai_client = client._get_client()

        # openai.OpenAI stores base_url as httpx.URL (trailing slash normalised)
        from urllib.parse import urlparse
        parsed = urlparse(str(openai_client.base_url))
        assert parsed.hostname == 'api.minimax.io'
        assert parsed.path.startswith('/v1')

    def test_standard_openai_platform_uses_default_base_url(self) -> None:
        """OpenAIClient for platform='openai' (no host) uses the default
        OpenAI base URL — verifies the two code paths are distinct.

        OpenAIService → OpenAIClient in services/llm_clients/openai.py.
        """
        from services.llm_clients.openai import OpenAIClient

        config: dict[str, object] = {
            'platform': 'openai',
            'model': 'gpt-4o-mini',
            'api_key': secrets.token_hex(16),
        }

        client = OpenAIClient(config)
        openai_client = client._get_client()

        from urllib.parse import urlparse
        assert urlparse(str(openai_client.base_url)).hostname == 'api.openai.com'

    # ------------------------------------------------------------------
    # 4a. The window is measured at setup and stored on the row
    #
    # Creating a provider asks the host what the model's context window actually
    # is and writes the answer to the row, so no send has to guess. Without this
    # the column stays NULL for every provider ever created — nothing else
    # populates it — and every turn falls back to a per-platform default that
    # has no relationship to the model actually loaded.
    # ------------------------------------------------------------------

    def _serve_models(self, model: str, n_ctx: object) -> "tuple[str, HTTPServer]":
        """Stand up a llama.cpp-shaped /v1/models endpoint; return (host, server)."""

        class _Models(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = json.dumps({
                    "data": [{"id": model, "meta": {"n_ctx": n_ctx}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

        server = HTTPServer(("127.0.0.1", 0), _Models)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/v1", server

    def _create_and_read_back(self, client: FlaskClient, name: str, model: str,
                              n_ctx: object) -> "dict[str, object]":
        host, server = self._serve_models(model, n_ctx)
        try:
            resp = client.post('/api/providers/-1', json={
                'name': name, 'platform': 'openai_compatible',
                'model': model, 'host': host,
                'api_key': secrets.token_hex(16),
            })
            assert resp.status_code == 201, resp.get_data(as_text=True)
        finally:
            server.shutdown()
        listing = _unwrap_listing(cast("dict[str, object]", client.get('/api/providers/all').get_json()))
        return next(p for p in listing if p['name'] == name)

    def test_create_stores_the_window_reported_by_the_host(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _, _store = authed_client
        _unlock_vault()

        stored = self._create_and_read_back(client, 'measured', 'local-model', 131072)

        assert stored['context_window'] == 131072

    def test_a_window_beyond_the_ceiling_is_clamped_on_the_row(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        from services.provider_api import MAX_CONTEXT_WINDOW

        client, _, _store = authed_client
        _unlock_vault()

        stored = self._create_and_read_back(client, 'huge', 'local-model', 1_000_000)

        # Clamped at write time, so nothing downstream can ever read a window
        # larger than the ceiling off a provider row.
        assert stored['context_window'] == MAX_CONTEXT_WINDOW

    def test_an_unreachable_host_leaves_the_window_unset(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _, _store = authed_client
        _unlock_vault()

        # Port 1 is closed; the probe cannot answer.
        resp = client.post('/api/providers/-1', json={
            'name': 'offline', 'platform': 'openai_compatible',
            'model': 'local-model', 'host': 'http://127.0.0.1:1/v1',
            'api_key': secrets.token_hex(16),
        })
        assert resp.status_code == 201, resp.get_data(as_text=True)

        listing = _unwrap_listing(cast("dict[str, object]", client.get('/api/providers/all').get_json()))
        stored = next(p for p in listing if p['name'] == 'offline')
        # Left NULL rather than seeded with a guess: a temporarily-down host must
        # not permanently stamp a fabricated window onto the row.
        assert stored.get('context_window') is None

    # ------------------------------------------------------------------
    # 4b. Size rejections from a host that is not OpenAI
    #
    # 'context_length_exceeded' is OpenAI's own error code. The self-hosted and
    # third-party servers that share this client are under no obligation to send
    # it, and mostly don't — they just return a 400 saying the input was too
    # long. Those must still raise ContextLimit, or the turn is retried at the
    # same size until attempts run out instead of compacting.
    #
    # Driven through a real HTTP 400 from a local server: the SDK's own error
    # construction and the client's except-branch both run for real.
    # ------------------------------------------------------------------

    def _send_against_400(self, message: str) -> None:
        from services.llm_clients.openai import OpenAIClient
        from services.provider_api import ProviderApiRequest

        class _Reject400(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.dumps({
                    "error": {"message": message, "type": "invalid_request_error"},
                }).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass  # keep the test output clean

        server = HTTPServer(("127.0.0.1", 0), _Reject400)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = OpenAIClient({
                'platform': 'openai_compatible',
                'model': 'local-model',
                'host': f"http://127.0.0.1:{server.server_port}/v1",
                'api_key': secrets.token_hex(16),
            })
            client.send(ProviderApiRequest(
                system="You are helpful.",
                messages=[{"role": "user", "content": "hi"}],
            ))
        finally:
            server.shutdown()

    def test_size_rejection_without_openai_error_code_raises_context_limit(self) -> None:
        from exceptions import ContextLimit

        with pytest.raises(ContextLimit) as caught:
            self._send_against_400(
                "the input exceeds the limit of 4096 tokens for this context length",
            )
        # The platform is reported as its own, not flattened to 'openai' —
        # the recovery path logs which provider hit the wall.
        assert caught.value.provider == 'openai_compatible'
        assert caught.value.model == 'local-model'

    def test_unrelated_400_is_not_mistaken_for_a_size_rejection(self) -> None:
        from exceptions import ContextLimit, ProviderResponseError

        with pytest.raises(ProviderResponseError) as caught:
            self._send_against_400("unknown parameter: 'foo'")
        # Compacting would not help; misreading this would shrink history for nothing.
        assert not isinstance(caught.value, ContextLimit)

    # ------------------------------------------------------------------
    # 5. Missing host → 4xx on POST
    # ------------------------------------------------------------------

    def test_post_openai_compatible_without_host_returns_error(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """POST /api/providers with platform='openai_compatible' but no host
        must be rejected with a 4xx response containing a useful error message."""
        client, _, _store = authed_client
        _unlock_vault()

        resp = client.post('/api/providers/-1', json={
            'name': 'minimax-no-host',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'api_key': secrets.token_hex(16),
            # host is intentionally omitted
        })

        # Legacy returned 422 here; the new base contract's EndpointError default
        # is 400 for the same validation failure surfaced by the provider service.
        assert resp.status_code == 400, (
            f"Expected 400 for missing host, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )
        body = cast("dict[str, object]", resp.get_json())
        error_text = _unwrap_error(body).lower()
        assert 'host' in error_text or 'required' in error_text or 'error' in error_text
