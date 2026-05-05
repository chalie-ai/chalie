"""
Tests for backend/api/lists.py — Lists API blueprint.

Real-DB end-to-end tests via the ``authed_client`` fixture: routes hit a real
ListService against a per-test SQLite database. No service-layer mocking.
"""

import pytest


pytestmark = pytest.mark.unit


def _seed_list(db, list_id='abc12345', name='Shopping List',
               list_type='checklist'):
    db.execute(
        "INSERT INTO lists (id, name, list_type, created_at, updated_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now'))",
        (list_id, name, list_type),
    )
    db.commit()


def _seed_item(db, item_id, list_id, content, checked=0, position=0):
    db.execute(
        "INSERT INTO list_items (id, list_id, content, checked, position, "
        "added_at, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
        (item_id, list_id, content, checked, position),
    )
    db.commit()


# ─── GET /lists ───────────────────────────────────────────────────────────

class TestGetLists:
    def test_returns_empty_when_no_lists(self, authed_client):
        client, _, _ = authed_client
        resp = client.get('/lists')
        assert resp.status_code == 200
        assert resp.get_json() == {"items": []}

    def test_returns_summaries(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db, list_id='aaa11111', name='Groceries')
        _seed_list(db, list_id='bbb22222', name='To-Do')

        resp = client.get('/lists')
        assert resp.status_code == 200
        data = resp.get_json()
        names = {item['name'] for item in data['items']}
        assert names == {'Groceries', 'To-Do'}


# ─── POST /lists ──────────────────────────────────────────────────────────

class TestCreateList:
    def test_creates_and_returns_201(self, authed_client):
        client, db, _ = authed_client
        resp = client.post('/lists', json={"name": "Chores"})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data['item']['name'] == 'Chores'
        assert len(data['item']['id']) == 8
        # Verify persisted
        row = db.execute(
            "SELECT name FROM lists WHERE id = ?", (data['item']['id'],)
        ).fetchone()
        assert row['name'] == 'Chores'

    def test_missing_name_returns_400(self, authed_client):
        client, _, _ = authed_client
        resp = client.post('/lists', json={})
        assert resp.status_code == 400

    def test_name_too_long_returns_400(self, authed_client):
        client, _, _ = authed_client
        resp = client.post('/lists', json={"name": "x" * 201})
        assert resp.status_code == 400

    def test_duplicate_name_returns_409(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db, name='Groceries')
        resp = client.post('/lists', json={"name": "Groceries"})
        assert resp.status_code == 409


# ─── GET /lists/<id> ──────────────────────────────────────────────────────

class TestGetList:
    def test_returns_list_with_items(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='Groceries')
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', checked=1, position=1)

        resp = client.get('/lists/abc12345')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['item']['name'] == 'Groceries'
        contents = [i['content'] for i in data['item']['items']]
        assert contents == ['milk', 'eggs']

    def test_unknown_id_returns_404(self, authed_client):
        client, _, _ = authed_client
        resp = client.get('/lists/nonexist')
        assert resp.status_code == 404


# ─── PUT /lists/<id>/rename ───────────────────────────────────────────────

class TestRenameList:
    def test_renames(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db, list_id='abc12345', name='Old')
        resp = client.put('/lists/abc12345/rename', json={"name": "New"})
        assert resp.status_code == 200
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'New'

    def test_empty_name_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.put('/lists/abc12345/rename', json={"name": ""})
        assert resp.status_code == 400

    def test_unknown_id_returns_404(self, authed_client):
        client, _, _ = authed_client
        resp = client.put('/lists/nonexist/rename', json={"name": "Anything"})
        assert resp.status_code == 404


# ─── DELETE /lists/<id> ───────────────────────────────────────────────────

class TestDeleteList:
    def test_soft_deletes(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.delete('/lists/abc12345')
        assert resp.status_code == 200
        row = db.execute(
            "SELECT deleted_at FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['deleted_at'] is not None

    def test_unknown_id_returns_404(self, authed_client):
        client, _, _ = authed_client
        resp = client.delete('/lists/nonexist')
        assert resp.status_code == 404


# ─── POST /lists/<id>/items ───────────────────────────────────────────────

class TestAddItems:
    def test_adds_and_returns_count(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/lists/abc12345/items', json={"items": ["Milk", "Eggs"]})
        assert resp.status_code == 200
        assert resp.get_json()['added'] == 2
        rows = db.execute(
            "SELECT content FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL "
            "ORDER BY position"
        ).fetchall()
        assert [r['content'] for r in rows] == ['Milk', 'Eggs']

    def test_empty_items_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/lists/abc12345/items', json={"items": []})
        assert resp.status_code == 400

    def test_item_too_long_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/lists/abc12345/items', json={"items": ["x" * 501]})
        assert resp.status_code == 400

    def test_non_string_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.post('/lists/abc12345/items', json={"items": [123]})
        assert resp.status_code == 400

    def test_unknown_id_returns_404(self, authed_client):
        client, _, _ = authed_client
        resp = client.post('/lists/nonexist/items', json={"items": ["Milk"]})
        assert resp.status_code == 404


# ─── DELETE /lists/<id>/items (clear) ─────────────────────────────────────

class TestClearItems:
    def test_clears_all(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        resp = client.delete('/lists/abc12345/items')
        assert resp.status_code == 200
        assert resp.get_json()['cleared'] == 2

    def test_unknown_id_returns_404(self, authed_client):
        client, _, _ = authed_client
        resp = client.delete('/lists/nonexist/items')
        assert resp.status_code == 404


# ─── DELETE /lists/<id>/items/batch ───────────────────────────────────────

class TestRemoveItems:
    def test_removes_specified_items(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        resp = client.delete('/lists/abc12345/items/batch', json={"items": ["milk"]})
        assert resp.status_code == 200
        assert resp.get_json()['removed'] == 1

    def test_missing_items_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.delete('/lists/abc12345/items/batch', json={})
        assert resp.status_code == 400


# ─── PUT /lists/<id>/items/check ──────────────────────────────────────────

class TestCheckItems:
    def test_checks_items(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)

        resp = client.put('/lists/abc12345/items/check', json={"items": ["milk"]})
        assert resp.status_code == 200
        assert resp.get_json()['checked'] == 1
        row = db.execute(
            "SELECT checked FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['checked'] == 1

    def test_missing_items_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.put('/lists/abc12345/items/check', json={})
        assert resp.status_code == 400


# ─── PUT /lists/<id>/items/uncheck ────────────────────────────────────────

class TestUncheckItems:
    def test_unchecks_items(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', checked=1, position=0)

        resp = client.put('/lists/abc12345/items/uncheck', json={"items": ["milk"]})
        assert resp.status_code == 200
        assert resp.get_json()['unchecked'] == 1
        row = db.execute(
            "SELECT checked FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['checked'] == 0

    def test_missing_items_returns_400(self, authed_client):
        client, db, _ = authed_client
        _seed_list(db)
        resp = client.put('/lists/abc12345/items/uncheck', json={})
        assert resp.status_code == 400
