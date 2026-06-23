"""Feature tests for the reactive provider-setup wizard's backend contract.

Drives the real Flask app (``create_app`` via ``authed_client``) against a real
SQLite database and the real curated catalog. No mocks.

The wizard's design: a curated preset maps a popular provider to one of Chalie's
five existing platforms plus a base URL, so the host is pre-filled and the live
model-fetch path (``POST /providers/list-models``) is reused unchanged. A preset
is NEVER a runtime platform of its own — the old catalog-id-as-platform branch is
gone, and these tests pin both the new catalog shape and the removal of that branch.
"""

import sqlite3
from collections.abc import Iterator

import pytest

import services.vault_service as _vault_mod
from flask.testing import FlaskClient
from services.vault_service import _vault_state, get_vault_service

# The seven providers the product owner named explicitly. The curated list is
# allowed to carry more, but never fewer than these, and never the 111-entry dump.
_NAMED_IDS = {"ollama", "anthropic", "gemini", "openai", "deepseek", "minimax", "nvidia"}
_CHALIE_PLATFORMS = {"ollama", "openai", "anthropic", "gemini", "openai_compatible"}


def _unlock_vault(password: str = "test-password") -> None:
    """Initialise + unlock the real vault so api_key encryption round-trips."""
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


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

        resp = client.get('/providers/catalog')

        assert resp.status_code == 200
        body = resp.get_json()
        catalog = body['catalog']
        assert isinstance(catalog, list), "catalog must be an ordered list of presets"
        assert 10 <= len(catalog) <= 25, f"curated list expected, got {len(catalog)}"

        # Every preset carries the full wizard contract and resolves to a real
        # Chalie platform (never a catalog-id-as-platform).
        for preset in catalog:
            assert set(preset) >= {"id", "name", "platform", "host", "needs_key"}
            assert preset['platform'] in _CHALIE_PLATFORMS
            assert isinstance(preset['needs_key'], bool)
            assert isinstance(preset['host'], str)

        ids = {p['id'] for p in catalog}
        assert len(ids) == len(catalog), "preset ids must be unique"
        assert _NAMED_IDS <= ids, f"missing named providers: {_NAMED_IDS - ids}"

    def test_named_presets_prefill_the_right_platform_and_host(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client

        catalog = client.get('/providers/catalog').get_json()['catalog']
        by_id = {p['id']: p for p in catalog}

        # MiniMax is the spec's worked example: OpenAI-compatible, host pre-filled.
        assert by_id['minimax']['platform'] == 'openai_compatible'
        assert by_id['minimax']['host'] == 'https://api.minimax.io/v1'
        assert by_id['minimax']['needs_key'] is True

        # NVIDIA + DeepSeek are also OpenAI-compatible with a pre-filled base URL.
        assert by_id['nvidia']['platform'] == 'openai_compatible'
        assert by_id['nvidia']['host'].startswith('https://')
        assert by_id['deepseek']['platform'] == 'openai_compatible'
        assert by_id['deepseek']['host'].startswith('https://')

        # Native API providers need a key but NO host field (host stays empty so
        # the wizard skips straight to the key step).
        for native in ('anthropic', 'openai', 'gemini'):
            assert by_id[native]['platform'] == native
            assert by_id[native]['host'] == ''
            assert by_id[native]['needs_key'] is True

        # Ollama is local: host pre-filled, no API key, so the wizard skips the
        # key step and fetches models immediately.
        assert by_id['ollama']['platform'] == 'ollama'
        assert by_id['ollama']['host'] == 'http://localhost:11434'
        assert by_id['ollama']['needs_key'] is False

    def test_preset_id_is_not_a_runtime_platform(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        """Pins removal of the catalog-id-as-platform branch: 'deepseek' was a
        real id in the old dump and must be rejected as a platform, not resolved
        to a static model list."""
        client, _db, _store = authed_client

        resp = client.post('/providers/list-models', json={'platform': 'deepseek'})

        assert resp.status_code == 400
        body = resp.get_json()
        assert body['models'] == []
        assert "Unsupported platform" in body['error']

    def test_preset_save_roundtrips_through_the_existing_create_path(self, authed_client: tuple[FlaskClient, sqlite3.Connection, object]) -> None:
        client, _db, _store = authed_client
        _unlock_vault()

        preset = next(
            p for p in client.get('/providers/catalog').get_json()['catalog']
            if p['id'] == 'minimax'
        )

        create = client.post('/providers', json={
            'name': 'My MiniMax',
            'platform': preset['platform'],   # 'openai_compatible'
            'host': preset['host'],           # pre-filled base URL
            'api_key': 'sk-test-key-123',
            'model': 'MiniMax-M2',
        })
        assert create.status_code == 201, create.get_json()

        # Cross-step proof: the host the preset pre-filled survived to the DB and
        # is read back on the listing (api_key is redacted, never echoed raw).
        listed = client.get('/providers').get_json()['providers']
        row = next(p for p in listed if p['name'] == 'My MiniMax')
        assert row['platform'] == 'openai_compatible'
        assert row['host'] == 'https://api.minimax.io/v1'
        assert row['model'] == 'MiniMax-M2'
        assert row['api_key'] in (None, '***')
