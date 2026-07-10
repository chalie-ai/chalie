"""Feature tests — GET /api/settings/personality and POST /api/settings/personality.

All tests are @pytest.mark.unit — real SQLite via the ``authed_client`` fixture,
real PersonalityAction on the Endpoint/Action base contract, real
personality service, real voices.jsonl corpus. Zero mocks for service internals.

Contract notes vs the legacy namespace this suite originally covered:
- PUT reshaped to POST (the base contract has no PUT).
- Payloads ride the uniform envelope: ``{"success": true, "result": {...}}``.
- Validation semantics are unchanged: request-DTO violations map to 422
  under the base contract, exactly like the legacy namespace.
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
        resp = client.get('/api/settings/personality')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        result = data['result']
        assert result['tuple'] == [0, 0, 0, 0, 0], (
            f"Expected neutral tuple [0,0,0,0,0], got {result['tuple']!r}"
        )
        assert result['voice'] == expected_voice, (
            f"Expected neutral voice, got {result['voice']!r}"
        )


# ── TestPersonalityAPIPost ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestPersonalityAPIPost:
    def test_post_personality_persists_and_returns_voice(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        corpus = _load_corpus()
        target = [-2, -2, -2, -2, -2]
        expected_voice = corpus[tuple(target)]

        client, _db, _store = authed_client

        post_resp = client.post(
            '/api/settings/personality',
            json={'tuple': target},
            content_type='application/json',
        )
        assert post_resp.status_code == 200
        post_data = post_resp.get_json()
        assert post_data['success'] is True
        post_result = post_data['result']
        assert post_result['voice'] == expected_voice, (
            f"POST response voice mismatch.\nExpected: {expected_voice!r}\nGot: {post_result['voice']!r}"
        )
        assert post_result['tuple'] == target

        get_resp = client.get('/api/settings/personality')
        assert get_resp.status_code == 200
        get_result = get_resp.get_json()['result']
        assert get_result['tuple'] == target, (
            f"GET after POST returned wrong tuple: {get_result['tuple']!r}"
        )
        assert get_result['voice'] == expected_voice, (
            "GET after POST returned wrong voice"
        )


# ── TestPersonalityAPIPostValidation ──────────────────────────────────────────
# Structural violations are caught by the request DTO under the base contract
# and answer 422, matching the legacy namespace.


@pytest.mark.unit
class TestPersonalityAPIPostValidation:
    def test_post_rejects_out_of_range_step(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/settings/personality',
            json={'tuple': [3, 0, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 422, (
            f"Expected 422 for out-of-range step, got {resp.status_code}"
        )

    def test_post_rejects_tuple_wrong_length(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/settings/personality',
            json={'tuple': [0, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 422, (
            f"Expected 422 for 4-element tuple, got {resp.status_code}"
        )

    def test_post_rejects_non_list_tuple(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/settings/personality',
            json={'tuple': 'not-a-list'},
            content_type='application/json',
        )
        assert resp.status_code == 422, (
            f"Expected 422 for string 'tuple', got {resp.status_code}"
        )

    def test_post_rejects_missing_tuple_field(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/settings/personality',
            json={},
            content_type='application/json',
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing 'tuple' field, got {resp.status_code}"
        )

    def test_post_rejects_bools_as_integers(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _db, _store = authed_client
        resp = client.post(
            '/api/settings/personality',
            json={'tuple': [True, False, 0, 0, 0]},
            content_type='application/json',
        )
        assert resp.status_code == 422, (
            f"Expected 422 for bool elements, got {resp.status_code}"
        )
