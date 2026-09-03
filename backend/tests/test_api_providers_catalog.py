"""Feature tests for the reactive provider-setup wizard's backend contract.

Drives the real Flask app (``create_app`` via ``authed_client``) against a real
SQLite database and the real catalog. No mocks.

The wizard's design: every offered tile IS a platform, backed by its own client
class, and the catalog is derived from the client registry rather than hand-kept
beside it. So the tile carries the platform's own base URL and credential rules,
and the live model-fetch path (``POST /providers/list-models``) dispatches on the
same string the tile prefills. These tests pin that shape end to end — through
the response DTO and the HTTP envelope, which is where a derived catalog can
still lose a field or mangle a value.
"""

import sqlite3
from collections.abc import Iterator
from typing import cast

import pytest
from flask.testing import FlaskClient

import services.vault_service as _vault_mod
from services.llm_clients.registry import PROVIDERS_BY_PLATFORM
from services.vault_service import _vault_state, get_vault_service

# The seven providers the product owner named explicitly. The list is allowed to
# carry more, but never fewer than these, and never the 111-entry dump.
_NAMED_IDS = {"ollama", "anthropic", "gemini", "openai", "deepseek", "minimax", "nvidia"}


def _unlock_vault(password: str = "test-password") -> None:
    """Initialise + unlock the real vault so api_key encryption round-trips."""
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


def _unwrap_listing(body: "dict[str, object]") -> "list[dict[str, object]]":
    """Assert the success listing envelope shape and return the result array."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("list[dict[str, object]]", body["result"])


def _unwrap_error(body: "dict[str, object]") -> str:
    """Assert the error envelope shape and return the error message."""
    assert body.get("success") is False
    assert body.get("result") == []
    return cast(str, body["error"])


@pytest.mark.unit
class TestProviderCatalog:

    @pytest.fixture(autouse=True)
    def _reset_vault_state(self) -> Iterator[None]:
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None
        yield
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

    def test_catalog_is_a_curated_preset_list_not_the_full_dump(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client

        resp = client.get('/api/providers/catalog')

        assert resp.status_code == 200
        body = cast("dict[str, object]", resp.get_json())
        catalog = _unwrap_listing(body)
        assert "pagination" in body
        assert isinstance(catalog, list), "catalog must be an ordered list of presets"
        assert 10 <= len(catalog) <= 40, f"curated list expected, got {len(catalog)}"

        # Every tile carries the full wizard contract and names a platform the
        # send path can actually dispatch. A tile whose platform the registry
        # does not claim is a dead end: the wizard prefills it, the user fills in
        # a key, and the create fails with "Unknown platform".
        for preset in catalog:
            assert set(preset) >= {"id", "name", "platform", "host", "needs_key", "needs_host"}
            assert preset['platform'] in PROVIDERS_BY_PLATFORM
            assert isinstance(preset['needs_key'], bool)
            assert isinstance(preset['needs_host'], bool)
            assert isinstance(preset['host'], str)

        ids = {cast(str, p['id']) for p in catalog}
        assert len(ids) == len(catalog), "preset ids must be unique"
        assert _NAMED_IDS <= ids, f"missing named providers: {_NAMED_IDS - ids}"

    def test_named_presets_prefill_the_right_platform_and_host(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client

        body = cast("dict[str, object]", client.get('/api/providers/catalog').get_json())
        catalog = _unwrap_listing(body)
        by_id = {cast(str, p['id']): p for p in catalog}

        # Every tile is its own platform — the id and the platform are the same
        # string, because the tile IS the client class. A vendor is never filed
        # under another vendor's platform.
        for entry in catalog:
            assert entry['platform'] == entry['id']

        # MiniMax is the spec's worked example: hosted vendor, host pre-filled.
        assert by_id['minimax']['host'] == 'https://api.minimax.io/v1'
        assert by_id['minimax']['needs_key'] is True
        assert by_id['minimax']['needs_host'] is True

        # NVIDIA + DeepSeek are hosted too, each with its own pre-filled base URL.
        assert cast(str, by_id['nvidia']['host']).startswith('https://')
        assert cast(str, by_id['deepseek']['host']).startswith('https://')

        # Native SDK providers need a key but NO host field (host stays empty so
        # the wizard skips straight to the key step).
        for native in ('anthropic', 'openai', 'gemini'):
            assert by_id[native]['host'] == ''
            assert by_id[native]['needs_host'] is False
            assert by_id[native]['needs_key'] is True

        # Self-hosted servers are local: host pre-filled, no API key, so the
        # wizard skips the key step and fetches models immediately.
        assert by_id['ollama']['host'] == 'http://localhost:11434'
        for local in ('ollama', 'llama_cpp', 'vllm'):
            assert by_id[local]['needs_key'] is False
            assert by_id[local]['needs_host'] is True
            assert cast(str, by_id[local]['host']).startswith('http://')

    def test_unregistered_platform_is_rejected(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """A platform no client class claims is refused, not guessed at.

        The listing endpoint dispatches on the registry, so an unknown string has
        nowhere to go. It must say so rather than fall through to a generic
        OpenAI-shaped attempt against a host nobody supplied.
        """
        client, _db, _store = authed_client

        resp = client.post(
            '/api/providers/list-models', json={'platform': 'not-a-real-platform'},
        )

        assert resp.status_code == 400
        body = cast("dict[str, object]", resp.get_json())
        error = _unwrap_error(body)
        assert "Unsupported platform" in error

    def test_preset_save_roundtrips_through_the_existing_create_path(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client
        _unlock_vault()

        body = cast("dict[str, object]", client.get('/api/providers/catalog').get_json())
        catalog = _unwrap_listing(body)
        preset = next(p for p in catalog if p['id'] == 'minimax')

        create = client.post('/api/providers/-1', json={
            'name': 'My MiniMax',
            'platform': preset['platform'],   # 'minimax'
            'host': preset['host'],           # pre-filled base URL
            'api_key': 'sk-test-key-123',
            'model': 'MiniMax-M2',
        })
        assert create.status_code == 201, create.get_json()

        # Cross-step proof: the platform and host the tile pre-filled survived to
        # the DB and are read back on the listing (api_key is write-only, never
        # on the read shape).
        listed_body = cast("dict[str, object]", client.get('/api/providers/all').get_json())
        listed = _unwrap_listing(listed_body)
        row = next(p for p in listed if p['name'] == 'My MiniMax')
        assert row['platform'] == 'minimax'
        assert row['host'] == 'https://api.minimax.io/v1'
        assert row['model'] == 'MiniMax-M2'
        assert 'api_key' not in row
