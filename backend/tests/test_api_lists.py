import sqlite3
from typing import cast

import pytest
from flask.testing import FlaskClient

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


def _seed_list(db: sqlite3.Connection, list_id: str = 'abc12345', name: str = 'Shopping List',
               list_type: str = 'checklist') -> None:
    db.execute(
        "INSERT INTO lists (id, name, list_type, created_at, updated_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
        (list_id, name, list_type),
    )
    db.commit()


def _seed_item(db: sqlite3.Connection, item_id: str, list_id: str, content: str, checked: int = 0, position: int = 0) -> None:
    db.execute(
        "INSERT INTO list_items (id, list_id, content, checked, position, "
        "added_at, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (item_id, list_id, content, checked, position),
    )
    db.commit()


def _unwrap_success(body: dict[str, object]) -> object:
    """Assert the success envelope shape and return the bare result payload.

    The payload shape is endpoint-specific (a ``list`` for the collection reads,
    a ``dict`` for the single-item writes), so it is returned as ``object``; each
    caller narrows it with an ``isinstance`` assertion before use."""
    assert body.get("success") is True
    assert "error" not in body
    return body["result"]


def _unwrap_error(body: dict[str, object]) -> str:
    """Assert the error envelope shape and return the error message."""
    assert body.get("success") is False
    assert body.get("result") == []
    return cast("str", body["error"])


# ─── GET /api/lists/all ───────────────────────────────────────────────────

class TestGetLists:
    def test_returns_summaries(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, list_id='aaa11111', name='Groceries')
        _seed_list(db, list_id='bbb22222', name='To-Do')

        resp = client.get('/api/lists/all')
        assert resp.status_code == 200
        body = resp.get_json()
        data = _unwrap_success(body)
        assert isinstance(data, list)
        names = {item['name'] for item in data}
        assert names == {'Groceries', 'To-Do'}
        first = data[0]
        assert {'id', 'name', 'list_type', 'item_count', 'checked_count', 'created_at', 'updated_at'} <= set(first)
        assert "pagination" in body


# ─── POST /api/lists/-1 ──────────────────────────────────────────────────

class TestCreateList:
    def test_creates_and_returns_201(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        resp = client.post('/api/lists/-1', json={"name": "Chores"})
        assert resp.status_code == 201
        data = _unwrap_success(resp.get_json())
        assert isinstance(data, dict)
        assert data['name'] == 'Chores'
        assert data['list_type'] == 'checklist'
        assert len(data['id']) == 8
        assert data['item_count'] == 0
        assert data['checked_count'] == 0
        row = db.execute("SELECT name FROM lists WHERE id = ?", (data['id'],)).fetchone()
        assert row['name'] == 'Chores'

    def test_missing_name_returns_400(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.post('/api/lists/-1', json={})
        assert resp.status_code == 400
        assert _unwrap_error(resp.get_json()) == 'name is required'

    def test_blank_name_returns_422(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.post('/api/lists/-1', json={"name": ""})
        assert resp.status_code == 422
        err = _unwrap_error(resp.get_json())
        assert err.startswith('Invalid request body — name:')

    def test_duplicate_name_returns_409(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, name='Groceries')
        resp = client.post('/api/lists/-1', json={"name": "Groceries"})
        assert resp.status_code == 409
        assert 'already exists' in _unwrap_error(resp.get_json())


# ─── GET /api/lists/<id> ──────────────────────────────────────────────────

class TestGetList:
    def test_returns_list_with_counts(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='Groceries')
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', checked=1, position=1)

        resp = client.get('/api/lists/abc12345')
        assert resp.status_code == 200
        data = _unwrap_success(resp.get_json())
        assert isinstance(data, dict)
        assert data['name'] == 'Groceries'
        assert data['item_count'] == 2
        assert data['checked_count'] == 1
        # Items are a separate sub-resource, never embedded on the list read shape.
        assert 'items' not in data

    def test_missing_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.get('/api/lists/nope0000')
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'


# ─── POST /api/lists/<id> (update) ────────────────────────────────────────

class TestUpdateList:
    def test_renames(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='Old')
        resp = client.post('/api/lists/abc12345', json={"name": "New"})
        assert resp.status_code == 200
        assert cast("dict[str, object]", _unwrap_success(resp.get_json()))['name'] == 'New'
        assert db.execute("SELECT name FROM lists WHERE id = 'abc12345'").fetchone()['name'] == 'New'

    def test_updates_type(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='Chores')
        resp = client.post('/api/lists/abc12345', json={"list_type": "todo"})
        assert resp.status_code == 200
        assert cast("dict[str, object]", _unwrap_success(resp.get_json()))['list_type'] == 'todo'

    def test_rename_collision_returns_409(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='A')
        _seed_list(db, list_id='ddd22222', name='B')
        resp = client.post('/api/lists/abc12345', json={"name": "B"})
        assert resp.status_code == 409
        assert 'already exists' in _unwrap_error(resp.get_json())

    def test_missing_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.post('/api/lists/nope0000', json={"name": "X"})
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'


# ─── DELETE /api/lists/<id> ───────────────────────────────────────────────

class TestDeleteList:
    def test_soft_deletes_204(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.delete('/api/lists/abc12345')
        assert resp.status_code == 204
        assert resp.data == b''
        assert db.execute("SELECT deleted_at FROM lists WHERE id = 'abc12345'").fetchone()['deleted_at'] is not None

    def test_missing_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.delete('/api/lists/nope0000')
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'


# ─── GET /api/lists/items/<list_id> ───────────────────────────────────────

class TestGetItems:
    def test_lists_items_ordered(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        resp = client.get('/api/lists/items/abc12345')
        assert resp.status_code == 200
        body = resp.get_json()
        data = _unwrap_success(body)
        assert isinstance(data, list)
        assert [i['content'] for i in data] == ['milk', 'eggs']
        assert {'id', 'content', 'checked', 'position', 'added_at', 'updated_at'} <= set(data[0])
        assert "pagination" in body

    def test_missing_list_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.get('/api/lists/items/nope0000')
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'


# ─── POST /api/lists/items/<list_id> ──────────────────────────────────────

class TestAddItem:
    def test_adds_and_returns_201(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/api/lists/items/abc12345', json={"content": "Milk"})
        assert resp.status_code == 201
        data = _unwrap_success(resp.get_json())
        assert isinstance(data, dict)
        assert data['content'] == 'Milk'
        assert data['checked'] is False
        assert len(data['id']) == 8
        rows = db.execute(
            "SELECT content FROM list_items WHERE list_id = 'abc12345' AND removed_at IS NULL"
        ).fetchall()
        assert [r['content'] for r in rows] == ['Milk']

    def test_missing_list_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, _, _ = authed_client
        resp = client.post('/api/lists/items/nope0000', json={"content": "Milk"})
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'

    def test_blank_content_returns_422(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/api/lists/items/abc12345', json={"content": ""})
        assert resp.status_code == 422
        err = _unwrap_error(resp.get_json())
        assert err.startswith('Invalid request body — content:')


# ─── POST /api/lists/items/<list_id>:<item_id> (update) ──────────────────

class TestUpdateItem:
    def test_checks_item(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)

        resp = client.post('/api/lists/items/abc12345:i1', json={"checked": True})
        assert resp.status_code == 200
        assert cast("dict[str, object]", _unwrap_success(resp.get_json()))['checked'] is True
        assert db.execute("SELECT checked FROM list_items WHERE id = 'i1'").fetchone()['checked'] == 1

    def test_unchecks_item(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', checked=1, position=0)

        resp = client.post('/api/lists/items/abc12345:i1', json={"checked": False})
        assert resp.status_code == 200
        assert cast("dict[str, object]", _unwrap_success(resp.get_json()))['checked'] is False
        assert db.execute("SELECT checked FROM list_items WHERE id = 'i1'").fetchone()['checked'] == 0

    def test_renames_content(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)

        resp = client.post('/api/lists/items/abc12345:i1', json={"content": "oat milk"})
        assert resp.status_code == 200
        assert cast("dict[str, object]", _unwrap_success(resp.get_json()))['content'] == 'oat milk'

    def test_missing_item_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/api/lists/items/abc12345:none00000', json={"checked": True})
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'


# ─── DELETE /api/lists/items/<list_id>:<item_id> ──────────────────────────

class TestDeleteItem:
    def test_soft_removes_204(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)

        resp = client.delete('/api/lists/items/abc12345:i1')
        assert resp.status_code == 204
        assert resp.data == b''
        assert db.execute("SELECT removed_at FROM list_items WHERE id = 'i1'").fetchone()['removed_at'] is not None

    def test_missing_item_returns_404(self, authed_client: tuple[FlaskClient, sqlite3.Connection, MemoryStore]) -> None:
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.delete('/api/lists/items/abc12345:none00000')
        assert resp.status_code == 404
        assert _unwrap_error(resp.get_json()) == 'Not found'
