"""
Feature tests for the openai_compatible LLM platform.

What this tests:
  A user configures a third-party OpenAI-compatible provider (e.g., MiniMax) by
  POSTing to /api/providers with platform='openai_compatible' and a custom host.
  The server stores the provider, lists it back correctly, and the LLM service
  factory returns an OpenAIService whose client is wired to the supplied base_url.

Real-stack — no mocks of production code.
  - Flask test client backed by the real authed_client fixture
  - Real SQLite database built from schema.sql via SchemaConvergenceService
  - Real vault (initialized + unlocked) so api_key encryption round-trips
  - Real create_llm_service() factory and OpenAIService._get_client()
  - The openai.OpenAI() constructor never makes network calls on init, so
    asserting on client.base_url is safe and does not require network access.
"""

import pytest

import services.vault_service as _vault_mod
from services.vault_service import _vault_state, get_vault_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unlock_vault(password: str = "test-password") -> None:
    """Initialise and unlock the vault for the current test's DB singleton."""
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAICompatibleProvider:
    """End-to-end feature tests for the openai_compatible platform."""

    @pytest.fixture(autouse=True)
    def _reset_vault_state(self):
        """Guarantee vault DEK is cleared before and after every test."""
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None
        yield
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

    # ------------------------------------------------------------------
    # 1. Provider is stored and listed back correctly
    # ------------------------------------------------------------------

    def test_post_openai_compatible_stores_and_lists_provider(self, authed_client):
        """POST /api/providers with openai_compatible platform stores the row and
        GET /api/providers lists it back with platform and host intact."""
        client, db_conn, _store = authed_client
        _unlock_vault()

        resp = client.post('/providers', json={
            'name': 'minimax-m2',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': 'sk-minimax-test-key',
        })

        assert resp.status_code == 201, resp.get_data(as_text=True)
        created = resp.get_json()['provider']
        assert created['platform'] == 'openai_compatible'
        assert created['model'] == 'MiniMax-M2'
        assert created['host'] == 'https://api.minimax.io/v1'
        assert created['api_key'] == '***'  # masked on create response

        list_resp = client.get('/providers')
        assert list_resp.status_code == 200
        providers = list_resp.get_json()['providers']
        names = [p['name'] for p in providers]
        assert 'minimax-m2' in names

        stored = next(p for p in providers if p['name'] == 'minimax-m2')
        assert stored['platform'] == 'openai_compatible'
        assert stored['host'] == 'https://api.minimax.io/v1'
        assert stored['model'] == 'MiniMax-M2'

    # ------------------------------------------------------------------
    # 2. API key round-trips through vault encryption
    # ------------------------------------------------------------------

    def test_api_key_decrypts_correctly_from_db(self, authed_client):
        """The api_key stored via POST is encrypted in the DB and decrypts to
        the original plaintext when fetched via ProviderDbService."""
        client, db_conn, _store = authed_client
        _unlock_vault()

        resp = client.post('/providers', json={
            'name': 'minimax-vault-roundtrip',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': 'sk-roundtrip-secret',
        })
        assert resp.status_code == 201
        provider_id = resp.get_json()['provider']['id']

        # Read back via the DB service (decrypts the key)
        from services.provider_db_service import ProviderDbService
        import services.database_service as _db_mod
        svc = ProviderDbService(_db_mod._shared_db_service)
        provider = svc.get_provider_by_id(provider_id)

        assert provider is not None
        assert provider['api_key'] == 'sk-roundtrip-secret'
        assert provider['platform'] == 'openai_compatible'
        assert provider['host'] == 'https://api.minimax.io/v1'

    # ------------------------------------------------------------------
    # 3. create_llm_service returns OpenAIService for openai_compatible
    # ------------------------------------------------------------------

    def test_create_llm_service_returns_openai_service(self):
        """create_llm_service with platform='openai_compatible' returns an
        OpenAIService instance, not an unknown-platform error."""
        from services.llm_service import create_llm_service, OpenAIService

        config = {
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': 'sk-minimax-test',
        }

        service = create_llm_service(config)

        assert isinstance(service, OpenAIService)

    # ------------------------------------------------------------------
    # 4. _get_client() wires base_url from the host field
    # ------------------------------------------------------------------

    def test_get_client_uses_host_as_base_url(self):
        """OpenAIService._get_client() for an openai_compatible config returns
        an openai.OpenAI client whose base_url points to the configured host."""
        from services.llm_service import OpenAIService

        config = {
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'host': 'https://api.minimax.io/v1',
            'api_key': 'sk-minimax-test',
        }

        service = OpenAIService(config)
        client = service._get_client()

        # openai.OpenAI stores base_url as httpx.URL (trailing slash normalised)
        from urllib.parse import urlparse
        parsed = urlparse(str(client.base_url))
        assert parsed.hostname == 'api.minimax.io'
        assert parsed.path.startswith('/v1')

    def test_standard_openai_platform_uses_default_base_url(self):
        """OpenAIService for platform='openai' (no host) uses the default
        OpenAI base URL — verifies the two code paths are distinct."""
        from services.llm_service import OpenAIService

        config = {
            'platform': 'openai',
            'model': 'gpt-4o-mini',
            'api_key': 'sk-openai-test',
        }

        service = OpenAIService(config)
        client = service._get_client()

        from urllib.parse import urlparse
        assert urlparse(str(client.base_url)).hostname == 'api.openai.com'

    # ------------------------------------------------------------------
    # 5. Missing host → 4xx on POST
    # ------------------------------------------------------------------

    def test_post_openai_compatible_without_host_returns_error(self, authed_client):
        """POST /api/providers with platform='openai_compatible' but no host
        must be rejected with a 4xx response containing a useful error message."""
        client, db_conn, _store = authed_client
        _unlock_vault()

        resp = client.post('/providers', json={
            'name': 'minimax-no-host',
            'platform': 'openai_compatible',
            'model': 'MiniMax-M2',
            'api_key': 'sk-test',
            # host is intentionally omitted
        })

        assert resp.status_code in (400, 422), (
            f"Expected 4xx for missing host, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )
        body = resp.get_json()
        assert body is not None
        error_text = str(body).lower()
        assert 'host' in error_text or 'required' in error_text or 'error' in error_text
