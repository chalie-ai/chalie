"""Feature tests — the watched-folders endpoint group on the Endpoint/Action contract.

All tests are @pytest.mark.unit — real app via the ``authed_client`` fixture
(real per-test SQLite DB, real in-process MemoryStore). Watched folders were
migrated off the legacy ``/api/documents/watched-folders`` namespace and promoted
to a top-level ``/api/watched-folders`` resource.

Contract notes vs the legacy namespace this suite replaces:
- GET /api/documents/watched-folders → GET /api/watched-folders/all: a bare
  array becomes a listing envelope with pagination.
- Collection POST / item PUT reshaped to the create-sentinel: POST
  /api/watched-folders/-1 creates, POST /api/watched-folders/<id> updates.
- /scan and /browse move to verb Actions: POST /api/watched-folders/scan/<id>
  (id-required) and POST /api/watched-folders/browse (id-less).
- Missing folder_path on create answers 400 (was legacy 422) — the base
  contract's EndpointError path.
"""

import os
import sqlite3
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit

_MISSING = 'no-such-folder'

AuthedClient = tuple[FlaskClient, sqlite3.Connection, MemoryStore]


def _create_folder(client: FlaskClient, path: str, **fields: object) -> dict[str, object]:
    """Create a folder via the create-sentinel and return the created result dict."""
    resp = client.post('/api/watched-folders/-1', json={'folder_path': path, **fields})
    assert resp.status_code == 201, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body['success'] is True
    result: dict[str, object] = body['result']
    return result


@pytest.mark.unit
class TestWatchedFoldersList:
    def test_get_all_returns_listing_envelope(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.get('/api/watched-folders/all')

        assert resp.status_code == 200
        body = resp.get_json()
        assert body['success'] is True
        assert isinstance(body['result'], list)
        assert 'pagination' in body

    def test_get_all_reflects_a_created_folder(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        created = _create_folder(client, str(tmp_path), label='Docs')

        body = client.get('/api/watched-folders/all').get_json()
        result = body['result']
        match = next(f for f in result if f['id'] == created['id'])
        assert match['folder_path'] == str(tmp_path)
        assert match['label'] == 'Docs'
        assert isinstance(match['enabled'], bool)
        assert isinstance(match['recursive'], bool)


@pytest.mark.unit
class TestWatchedFoldersCreate:
    def test_create_returns_201_with_defaults_and_persists(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, db, _store = authed_client
        resp = client.post('/api/watched-folders/-1', json={'folder_path': str(tmp_path)})

        assert resp.status_code == 201
        data = resp.get_json()['result']
        assert data['folder_path'] == str(tmp_path)
        assert data['recursive'] is True       # request default
        assert data['scan_interval'] == 300    # request default
        row = db.execute("SELECT folder_path FROM watched_folders WHERE id = ?", (data['id'],)).fetchone()
        assert row['folder_path'] == str(tmp_path)

    def test_create_missing_folder_path_returns_400(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.post('/api/watched-folders/-1', json={'label': 'no path here'})

        assert resp.status_code == 400
        assert resp.get_json()['success'] is False

    def test_create_nonexistent_path_returns_422(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.post('/api/watched-folders/-1', json={'folder_path': '/no/such/dir/xyzzy'})

        assert resp.status_code == 422
        assert resp.get_json()['success'] is False

    def test_create_duplicate_path_returns_409(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        _create_folder(client, str(tmp_path))

        dup = client.post('/api/watched-folders/-1', json={'folder_path': str(tmp_path)})
        assert dup.status_code == 409
        assert dup.get_json()['success'] is False


@pytest.mark.unit
class TestWatchedFoldersUpdate:
    def test_update_applies_partial_fields(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        created = _create_folder(client, str(tmp_path), label='Old')

        resp = client.post(f"/api/watched-folders/{created['id']}", json={'label': 'New', 'enabled': False})
        assert resp.status_code == 200
        data = resp.get_json()['result']
        assert data['label'] == 'New'
        assert data['enabled'] is False
        assert data['folder_path'] == str(tmp_path)

    def test_update_ignores_folder_path(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        created = _create_folder(client, str(tmp_path))

        # folder_path is create-only; a valid dir here must be ignored, not applied.
        resp = client.post(f"/api/watched-folders/{created['id']}", json={'folder_path': '/etc'})
        assert resp.status_code == 200
        assert resp.get_json()['result']['folder_path'] == str(tmp_path)

    def test_update_unknown_id_returns_404(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.post(f'/api/watched-folders/{_MISSING}', json={'label': 'x'})

        assert resp.status_code == 404
        assert resp.get_json()['success'] is False


@pytest.mark.unit
class TestWatchedFoldersDelete:
    def test_delete_returns_204_and_removes_row(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, db, _store = authed_client
        created = _create_folder(client, str(tmp_path))

        resp = client.delete(f"/api/watched-folders/{created['id']}")
        assert resp.status_code == 204
        assert resp.get_data() == b''
        row = db.execute("SELECT id FROM watched_folders WHERE id = ?", (created['id'],)).fetchone()
        assert row is None

    def test_delete_unknown_id_returns_404(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.delete(f'/api/watched-folders/{_MISSING}')

        assert resp.status_code == 404
        assert resp.get_json()['success'] is False


@pytest.mark.unit
class TestWatchedFoldersNoDetailRoute:
    def test_get_by_id_is_method_not_allowed(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        created = _create_folder(client, str(tmp_path))

        # The legacy namespace had no GET-by-id; the base leaves it a 405 stub.
        resp = client.get(f"/api/watched-folders/{created['id']}")
        assert resp.status_code == 405


@pytest.mark.unit
class TestWatchedFoldersScan:
    def test_scan_existing_folder_returns_204(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        created = _create_folder(client, str(tmp_path))

        resp = client.post(f"/api/watched-folders/scan/{created['id']}")
        assert resp.status_code == 204
        assert resp.get_data() == b''

    def test_scan_unknown_id_returns_404(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.post(f'/api/watched-folders/scan/{_MISSING}')

        assert resp.status_code == 404
        assert resp.get_json()['success'] is False

    def test_scan_sentinel_returns_404(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        # scan is id-required — the create sentinel is not a valid target.
        resp = client.post('/api/watched-folders/scan/-1')

        assert resp.status_code == 404
        assert resp.get_json()['success'] is False


@pytest.mark.unit
class TestWatchedFoldersBrowse:
    def test_browse_valid_path_lists_subdirs(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        (tmp_path / 'sub').mkdir()

        resp = client.post('/api/watched-folders/browse', json={'path': str(tmp_path)})
        assert resp.status_code == 200
        data = resp.get_json()['result']
        assert data['current'] == os.path.realpath(str(tmp_path))
        assert 'sub' in data['directories']
        assert 'parent' in data

    def test_browse_relative_path_returns_422(self, authed_client: AuthedClient) -> None:
        client, _db, _store = authed_client
        resp = client.post('/api/watched-folders/browse', json={'path': 'not/absolute'})

        assert resp.status_code == 422
        assert resp.get_json()['success'] is False

    def test_browse_id_addressed_returns_404(self, authed_client: AuthedClient, tmp_path: Path) -> None:
        client, _db, _store = authed_client
        # browse is id-less only — an id-addressed call is not a valid target.
        resp = client.post('/api/watched-folders/browse/some-id', json={'path': str(tmp_path)})

        assert resp.status_code == 404
        assert resp.get_json()['success'] is False
