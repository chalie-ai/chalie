"""Feature tests — GET /settings/personality and PUT /settings/personality.

All tests are @pytest.mark.unit — real SQLite via the ``authed_client`` fixture,
real personality blueprint, real SettingsService, real voices.jsonl corpus.
Zero mocks for service internals.

Blueprint: api.personality (url_prefix='/settings')
"""

import json
import sqlite3

import pytest

from flask.testing import FlaskClient
from services.file_mapper_service import FileMapperService
from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


# ── Shared corpus helper ───────────────────────────────────────────────────────

_VOICES_PATH = str(FileMapperService.get_backend_path(
    'services', 'personality', 'voices.jsonl',
))


def _load_corpus() -> dict[tuple[int, ...], str]:
    index: dict[tuple[int, ...], str] = {}
    with open(_VOICES_PATH, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index[tuple(obj['tuple'])] = obj['voice']
    return index


# ── TestPersonalityAPIGet ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPersonalityAPIGet:
    def test_get_personality_returns_neutral_when_unset(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        corpus = _load_corpus()
        expected_voice = corpus[(0, 0, 0, 0, 0)]

        client, _db, _store = authed_client
        resp = client.get('/settings/personality')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['tuple'] == [0, 0, 0, 0, 0], (
            f"Expected neutral tuple [0,0,0,0,0], got {data['tuple']!r}"
        )
        assert data['voice'] == expected_voice, (
            f"Expected neutral voice, got {data['voice']!r}"
        )


# ── TestPersonalityAPIPut ──────────────────────────────────────────────────────


@pytest.mark.unit
class TestPersonalityAPIPut:
    def test_put_personality_persists_and_returns_voice(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        corpus = _load_corpus()
        target = [-2, -2, -2, -2, -2]
        expected_voice = corpus[tuple(target)]

        client, _db, _store = authed_client

        put_resp = client.put(
            '/settings/personality',
            json={'tuple': target},
            content_type='application/json',
        )
        assert put_resp.status_code == 200
        put_data = put_resp.get_json()
        assert put_data['voice'] == expected_voice, (
            f"PUT response voice mismatch.\nExpected: {expected_voice!r}\nGot: {put_data['voice']!r}"
        )
        assert put_data['tuple'] == target

        get_resp = client.get('/settings/personality')
        assert get_resp.status_code == 200
        get_data = get_resp.get_json()
        assert get_data['tuple'] == target, (
            f"GET after PUT returned wrong tuple: {get_data['tuple']!r}"
        )
        assert get_data['voice'] == expected_voice, (
            "GET after PUT returned wrong voice"
        )


# ── TestPersonalityAPIPutValidation ───────────────────────────────────────────


@pytest.mark.unit
class TestPersonalityAPIPutValidation:
    def test_put_rejects_out_of_range_step(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.put(
            '/settings/personality',
            json={'tuple': [3, 0, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 400, (
            f"Expected 400 for out-of-range step, got {resp.status_code}"
        )

    def test_put_rejects_tuple_wrong_length(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.put(
            '/settings/personality',
            json={'tuple': [0, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 400, (
            f"Expected 400 for 4-element tuple, got {resp.status_code}"
        )

    def test_put_rejects_non_list_tuple(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.put(
            '/settings/personality',
            json={'tuple': 'not-a-list'},
            content_type='application/json',
        )
        assert resp.status_code == 400, (
            f"Expected 400 for string 'tuple', got {resp.status_code}"
        )

    def test_put_rejects_missing_tuple_field(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.put(
            '/settings/personality',
            json={},
            content_type='application/json',
        )
        assert resp.status_code == 400, (
            f"Expected 400 for missing 'tuple' field, got {resp.status_code}"
        )

    def test_put_rejects_bools_as_integers(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.put(
            '/settings/personality',
            json={'tuple': [True, False, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 400, (
            f"Expected 400 for bool elements, got {resp.status_code}"
        )
