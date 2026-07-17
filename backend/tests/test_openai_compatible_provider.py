# Feature tests for the openai_compatible LLM platform.
# Real-stack — no mocks of production code.

import secrets
import sqlite3
from collections.abc import Iterator
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
    def _reset_vault_state(self) -> Iterator[None]:
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
