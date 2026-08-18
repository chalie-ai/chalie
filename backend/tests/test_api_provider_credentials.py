"""Feature tests for the provider edit form's credential and model-list contract.

Drives the real Flask app (``create_app`` via ``authed_client``) against a real
SQLite database, the real vault, and — for the model listing — a real HTTP server
speaking the OpenAI ``/models`` protocol. No mocks.

Three behaviours are pinned here, all of them things the edit form depends on:

- Saving a provider without sending ``api_key`` keeps the stored credential. The
  form hides the key behind a "Change API key" button, so an ordinary edit omits
  the field entirely; if an omitted field were read as "clear it", every rename
  would silently destroy a credential.
- ``GET /api/providers/api-key/<id>`` hands the decrypted key back, and only to a
  caller the rest of the provider admin surface would also accept.
- ``POST /api/providers/list-models`` can list a stored provider's models from the
  credential on its row, so opening the edit page refreshes the list without the
  key ever reaching the browser.
"""

import json
import sqlite3
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

import services.vault_service as _vault_mod
from services.vault_service import _vault_state, get_vault_service

_STORED_KEY = "sk-stored-credential-001"
_NEW_KEY = "sk-replacement-credential-002"


def _unlock_vault(password: str = "test-password") -> None:
    """Initialise + unlock the real vault so api_key encryption round-trips."""
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


def _result(resp: TestResponse) -> "dict[str, object]":
    """Assert the success envelope and return its result object."""
    body = cast("dict[str, object]", resp.get_json())
    assert body.get("success") is True, body
    return cast("dict[str, object]", body["result"])


class _ModelsHandler(BaseHTTPRequestHandler):
    """An OpenAI-protocol host that only lists models for one exact bearer.

    Recording the header is what makes the stored-credential test conclusive: a
    listing coming back proves *a* key worked, but only the recorded value proves
    it was the one on the provider row rather than something the client invented.
    """

    seen_auth: "list[str | None]" = []
    expected_key: str = _STORED_KEY

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's contract
        if not self.path.rstrip('/').endswith('/models'):
            self.send_error(404)
            return
        type(self).seen_auth.append(self.headers.get('Authorization'))
        if self.headers.get('Authorization') != f'Bearer {type(self).expected_key}':
            self.send_error(401)
            return
        payload = json.dumps({"data": [{"id": "abacus-2"}, {"id": "abacus-1"}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's contract
        # The vision probe fires on create; refusing it fast keeps the test quick
        # and leaves supports_vision at 0, which is what an unprobeable host means.
        self.send_error(400)

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request stderr line the default handler prints."""


@pytest.fixture
def models_host() -> Iterator[str]:
    """A real HTTP server on loopback, yielding its base URL."""
    _ModelsHandler.seen_auth = []
    _ModelsHandler.expected_key = _STORED_KEY
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.unit
class TestProviderCredentials:

    @pytest.fixture(autouse=True)
    def _reset_vault_state(self) -> Iterator[None]:
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None
        yield
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

    def _create(self, client: FlaskClient, host: str = '', key: str = _STORED_KEY) -> int:
        resp = client.post('/api/providers/-1', json={
            'name': 'Editable Provider',
            'platform': 'minimax',
            'host': host,
            'api_key': key,
            'model': 'abacus-1',
        })
        assert resp.status_code == 201, resp.get_json()
        return cast(int, _result(resp)['id'])

    def test_saving_without_the_key_field_keeps_the_stored_credential(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    ) -> None:
        """The retain path: an update that omits api_key leaves the row's key alone.

        This is the whole reason the form can hide the credential behind a button.
        The update sends everything the form holds *except* the key — the exact
        body an edit that never pressed "Change API key" produces.
        """
        client, _db, _store = authed_client
        _unlock_vault()
        provider_id = self._create(client)

        update = client.post(f'/api/providers/{provider_id}', json={
            'name': 'Renamed Provider',
            'platform': 'minimax',
            'model': 'abacus-1',
        })
        assert update.status_code == 200, update.get_json()
        assert _result(update)['name'] == 'Renamed Provider'

        revealed = client.get(f'/api/providers/api-key/{provider_id}')
        assert revealed.status_code == 200
        assert _result(revealed)['api_key'] == _STORED_KEY

    def test_an_explicit_new_key_still_replaces_the_stored_one(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    ) -> None:
        """The other half of the retain rule — omission keeps, a value replaces.

        Without this, "keep the existing key" could be implemented by ignoring the
        field altogether and the retain test above would still pass.
        """
        client, _db, _store = authed_client
        _unlock_vault()
        provider_id = self._create(client)

        update = client.post(f'/api/providers/{provider_id}', json={'api_key': _NEW_KEY})
        assert update.status_code == 200, update.get_json()

        revealed = client.get(f'/api/providers/api-key/{provider_id}')
        assert _result(revealed)['api_key'] == _NEW_KEY

    def test_the_stored_key_is_revealed_to_the_edit_form(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    ) -> None:
        """Pressing "Change API key" gets the real key back, not a mask.

        Also pins that the ordinary read shapes still strip it: the reveal is a
        route of its own precisely so the listing cannot start carrying secrets.
        """
        client, _db, _store = authed_client
        _unlock_vault()
        provider_id = self._create(client)

        revealed = client.get(f'/api/providers/api-key/{provider_id}')
        assert revealed.status_code == 200
        result = _result(revealed)
        assert result['api_key'] == _STORED_KEY
        assert result['id'] == provider_id

        one = client.get(f'/api/providers/{provider_id}')
        assert 'api_key' not in _result(one)
        listed = cast("dict[str, object]", client.get('/api/providers/all').get_json())
        for row in cast("list[dict[str, object]]", listed['result']):
            assert 'api_key' not in row

    def test_revealing_an_unknown_or_unaddressed_provider_is_not_found(
        self, authed_client: tuple[FlaskClient, sqlite3.Connection, object],
    ) -> None:
        client, _db, _store = authed_client
        _unlock_vault()

        assert client.get('/api/providers/api-key/999999').status_code == 404
        assert client.get('/api/providers/api-key').status_code == 404

    def test_the_reveal_route_refuses_an_unauthenticated_caller(
        self, db: sqlite3.Connection,
    ) -> None:
        """Built without the session patch, so the real auth decorator decides.

        A route that hands back a decrypted credential has to be behind exactly
        the same gate as the rest of the provider admin surface — this asserts it
        answers the same 401 the CRUD routes do rather than being reachable.
        """
        from api import create_app

        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as anon:
            assert anon.get('/api/providers/api-key/1').status_code == 401
            # The comparison that makes the assertion meaningful: same gate.
            assert anon.get('/api/providers/all').status_code == 401

    def test_the_model_list_refreshes_from_the_stored_credential(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        models_host: str,
    ) -> None:
        """Opening the edit page can list models without holding the key.

        The request carries provider_id and no api_key — exactly what the form
        sends before anyone presses "Change API key" — and the host still sees the
        credential stored on the row.
        """
        client, _db, _store = authed_client
        _unlock_vault()
        provider_id = self._create(client, host=models_host)
        _ModelsHandler.seen_auth = []

        resp = client.post('/api/providers/list-models', json={
            'platform': 'minimax',
            'host': models_host,
            'provider_id': provider_id,
        })

        assert resp.status_code == 200, resp.get_json()
        result = _result(resp)
        assert result['error'] is None, result
        assert [m['id'] for m in cast("list[dict[str, object]]", result['models'])] == [
            'abacus-1', 'abacus-2',
        ]
        assert _ModelsHandler.seen_auth == [f'Bearer {_STORED_KEY}']

    def test_a_typed_key_overrides_the_stored_one_when_listing_models(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        models_host: str,
    ) -> None:
        """A revealed-and-edited key lists against what was typed, not what is stored.

        Otherwise the form would show the old provider's models while the operator
        is pasting a new credential in front of it.
        """
        client, _db, _store = authed_client
        _unlock_vault()
        provider_id = self._create(client, host=models_host)
        _ModelsHandler.seen_auth = []

        resp = client.post('/api/providers/list-models', json={
            'platform': 'minimax',
            'host': models_host,
            'api_key': _NEW_KEY,
            'provider_id': provider_id,
        })

        assert resp.status_code == 200, resp.get_json()
        assert _result(resp)['error'] == 'Invalid API key'
        assert _ModelsHandler.seen_auth == [f'Bearer {_NEW_KEY}']

    def test_listing_models_without_a_provider_id_is_unchanged(
        self,
        authed_client: tuple[FlaskClient, sqlite3.Connection, object],
        models_host: str,
    ) -> None:
        """The pre-existing create-flow path still works on inline credentials alone."""
        client, _db, _store = authed_client

        resp = client.post('/api/providers/list-models', json={
            'platform': 'minimax',
            'host': models_host,
            'api_key': _STORED_KEY,
        })

        assert resp.status_code == 200, resp.get_json()
        result = _result(resp)
        assert result['error'] is None
        assert len(cast("list[object]", result['models'])) == 2
