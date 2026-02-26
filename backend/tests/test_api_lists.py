"""
Tests for backend/api/lists.py — Lists API blueprint.

All tests mock _get_list_service() to isolate the HTTP layer from the
database-backed ListService. No external dependencies required.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from flask import Flask

from api.lists import lists_bp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_list_dict(
    list_id="abc12345",
    name="Shopping List",
    list_type="checklist",
    item_count=3,
    checked_count=1,
    updated_at=None,
    created_at=None,
    items=None,
):
    """Build a dict that mirrors what ListService returns."""
    now = updated_at or datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)
    base = {
        "id": list_id,
        "name": name,
        "list_type": list_type,
        "updated_at": now,
    }
    if items is not None:
        base["items"] = items
    else:
        base["item_count"] = item_count
        base["checked_count"] = checked_count
    if created_at is not None:
        base["created_at"] = created_at
    return base


def _make_item_dict(
    item_id="item0001",
    content="Milk",
    checked=False,
    position=0,
    added_at=None,
    updated_at=None,
):
    return {
        "id": item_id,
        "content": content,
        "checked": checked,
        "position": position,
        "added_at": added_at or datetime(2026, 2, 26, 10, 0, 0, tzinfo=timezone.utc),
        "updated_at": updated_at or datetime(2026, 2, 26, 10, 0, 0, tzinfo=timezone.utc),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestListsAPI:
    """Test all endpoints on the lists blueprint."""

    @pytest.fixture
    def client(self):
        """Create a minimal Flask test client with only the lists blueprint."""
        app = Flask(__name__)
        app.register_blueprint(lists_bp)
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for every test in this class."""
        with patch("services.auth_session_service.validate_session", return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /lists
    # ------------------------------------------------------------------

    def test_get_lists_returns_serialized_lists(self, client):
        """GET /lists returns a list of serialized list summaries."""
        lists_data = [
            _make_list_dict(list_id="aaa11111", name="Groceries"),
            _make_list_dict(list_id="bbb22222", name="To-Do"),
        ]
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_all_lists.return_value = lists_data

            resp = client.get("/lists")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "items" in data
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "Groceries"
        assert data["items"][1]["name"] == "To-Do"
        # datetime should be serialized as ISO string
        assert isinstance(data["items"][0]["updated_at"], str)

    def test_get_lists_returns_empty_list_when_none(self, client):
        """GET /lists returns an empty items array when no lists exist."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_all_lists.return_value = []

            resp = client.get("/lists")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["items"] == []

    # ------------------------------------------------------------------
    # POST /lists
    # ------------------------------------------------------------------

    def test_create_list_returns_201(self, client):
        """POST /lists with a valid name creates a list and returns 201."""
        created_list = _make_list_dict(
            list_id="new00001",
            name="Chores",
            items=[],
            created_at=datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc),
        )
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.create_list.return_value = "new00001"
            mock_svc.get_list.return_value = created_list

            resp = client.post("/lists", json={"name": "Chores"})

        assert resp.status_code == 201
        data = resp.get_json()
        assert "item" in data
        assert data["item"]["name"] == "Chores"
        assert data["item"]["id"] == "new00001"
        mock_svc.create_list.assert_called_once_with("Chores", list_type="checklist")

    def test_create_list_with_custom_type(self, client):
        """POST /lists passes list_type to the service when provided."""
        created_list = _make_list_dict(
            list_id="new00002",
            name="Watchlist",
            list_type="plain",
            items=[],
        )
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.create_list.return_value = "new00002"
            mock_svc.get_list.return_value = created_list

            resp = client.post("/lists", json={"name": "Watchlist", "list_type": "plain"})

        assert resp.status_code == 201
        mock_svc.create_list.assert_called_once_with("Watchlist", list_type="plain")

    def test_create_list_missing_name_returns_400(self, client):
        """POST /lists without a name returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.post("/lists", json={})

        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_create_list_name_too_long_returns_400(self, client):
        """POST /lists with name > 200 chars returns 400."""
        long_name = "x" * 201
        with patch("api.lists._get_list_service"):
            resp = client.post("/lists", json={"name": long_name})

        assert resp.status_code == 400
        assert "200" in resp.get_json()["error"]

    def test_create_list_duplicate_name_returns_409(self, client):
        """POST /lists with a duplicate name returns 409 when service raises ValueError."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.create_list.side_effect = ValueError("A list named 'Groceries' already exists.")

            resp = client.post("/lists", json={"name": "Groceries"})

        assert resp.status_code == 409
        assert "already exists" in resp.get_json()["error"]

    # ------------------------------------------------------------------
    # GET /lists/<id>
    # ------------------------------------------------------------------

    def test_get_list_returns_list_with_items(self, client):
        """GET /lists/<id> returns the list dict with serialized items."""
        items = [
            _make_item_dict(item_id="i1", content="Milk", position=0),
            _make_item_dict(item_id="i2", content="Bread", position=1, checked=True),
        ]
        list_data = _make_list_dict(list_id="abc12345", name="Groceries", items=items)

        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = list_data

            resp = client.get("/lists/abc12345")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["item"]["name"] == "Groceries"
        assert len(data["item"]["items"]) == 2
        assert data["item"]["items"][0]["content"] == "Milk"
        # datetime fields should be ISO strings
        assert isinstance(data["item"]["items"][0]["added_at"], str)

    def test_get_list_not_found_returns_404(self, client):
        """GET /lists/<id> returns 404 when service returns None."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = None

            resp = client.get("/lists/nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    # ------------------------------------------------------------------
    # PUT /lists/<id>/rename
    # ------------------------------------------------------------------

    def test_rename_list_succeeds(self, client):
        """PUT /lists/<id>/rename with a valid name returns ok."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.rename_list.return_value = True

            resp = client.put("/lists/abc12345/rename", json={"name": "New Name"})

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        mock_svc.rename_list.assert_called_once_with("abc12345", "New Name")

    def test_rename_list_empty_name_returns_400(self, client):
        """PUT /lists/<id>/rename with empty name returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.put("/lists/abc12345/rename", json={"name": ""})

        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_rename_list_not_found_returns_404(self, client):
        """PUT /lists/<id>/rename returns 404 when service returns False."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.rename_list.return_value = False

            resp = client.put("/lists/nonexistent/rename", json={"name": "Valid"})

        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # DELETE /lists/<id>
    # ------------------------------------------------------------------

    def test_delete_list_succeeds(self, client):
        """DELETE /lists/<id> soft-deletes and returns ok."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.delete_list.return_value = True

            resp = client.delete("/lists/abc12345")

        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        mock_svc.delete_list.assert_called_once_with("abc12345")

    def test_delete_list_not_found_returns_404(self, client):
        """DELETE /lists/<id> returns 404 when list does not exist."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.delete_list.return_value = False

            resp = client.delete("/lists/nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    # ------------------------------------------------------------------
    # POST /lists/<id>/items
    # ------------------------------------------------------------------

    def test_add_items_returns_added_count(self, client):
        """POST /lists/<id>/items adds items and returns the added count."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = _make_list_dict(items=[])
            mock_svc.add_items.return_value = 2

            resp = client.post(
                "/lists/abc12345/items",
                json={"items": ["Milk", "Eggs"]},
            )

        assert resp.status_code == 200
        assert resp.get_json()["added"] == 2
        mock_svc.add_items.assert_called_once_with("abc12345", ["Milk", "Eggs"], auto_create=False)

    def test_add_items_empty_array_returns_400(self, client):
        """POST /lists/<id>/items with an empty items array returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.post("/lists/abc12345/items", json={"items": []})

        assert resp.status_code == 400
        assert "non-empty" in resp.get_json()["error"].lower()

    def test_add_items_item_too_long_returns_400(self, client):
        """POST /lists/<id>/items with an item > 500 chars returns 400."""
        long_item = "x" * 501
        with patch("api.lists._get_list_service"):
            resp = client.post("/lists/abc12345/items", json={"items": [long_item]})

        assert resp.status_code == 400
        assert "500" in resp.get_json()["error"]

    def test_add_items_non_string_item_returns_400(self, client):
        """POST /lists/<id>/items with a non-string item returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.post("/lists/abc12345/items", json={"items": [123]})

        assert resp.status_code == 400
        assert "non-empty string" in resp.get_json()["error"].lower()

    def test_add_items_list_not_found_returns_404(self, client):
        """POST /lists/<id>/items returns 404 when the list does not exist."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = None

            resp = client.post(
                "/lists/nonexistent/items",
                json={"items": ["Milk"]},
            )

        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    # ------------------------------------------------------------------
    # DELETE /lists/<id>/items/batch
    # ------------------------------------------------------------------

    def test_remove_items_batch_returns_removed_count(self, client):
        """DELETE /lists/<id>/items/batch removes items and returns count."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = _make_list_dict(items=[])
            mock_svc.remove_items.return_value = 2

            resp = client.delete(
                "/lists/abc12345/items/batch",
                json={"items": ["Milk", "Eggs"]},
            )

        assert resp.status_code == 200
        assert resp.get_json()["removed"] == 2
        mock_svc.remove_items.assert_called_once_with("abc12345", ["Milk", "Eggs"])

    def test_remove_items_batch_missing_items_returns_400(self, client):
        """DELETE /lists/<id>/items/batch without items returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.delete("/lists/abc12345/items/batch", json={})

        assert resp.status_code == 400

    # ------------------------------------------------------------------
    # PUT /lists/<id>/items/check
    # ------------------------------------------------------------------

    def test_check_items_returns_checked_count(self, client):
        """PUT /lists/<id>/items/check checks items and returns count."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = _make_list_dict(items=[])
            mock_svc.check_items.return_value = 1

            resp = client.put(
                "/lists/abc12345/items/check",
                json={"items": ["Milk"]},
            )

        assert resp.status_code == 200
        assert resp.get_json()["checked"] == 1
        mock_svc.check_items.assert_called_once_with("abc12345", ["Milk"])

    def test_check_items_missing_items_returns_400(self, client):
        """PUT /lists/<id>/items/check without items returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.put("/lists/abc12345/items/check", json={})

        assert resp.status_code == 400

    # ------------------------------------------------------------------
    # PUT /lists/<id>/items/uncheck
    # ------------------------------------------------------------------

    def test_uncheck_items_returns_unchecked_count(self, client):
        """PUT /lists/<id>/items/uncheck unchecks items and returns count."""
        with patch("api.lists._get_list_service") as mock_get_svc:
            mock_svc = MagicMock()
            mock_get_svc.return_value = mock_svc
            mock_svc.get_list.return_value = _make_list_dict(items=[])
            mock_svc.uncheck_items.return_value = 2

            resp = client.put(
                "/lists/abc12345/items/uncheck",
                json={"items": ["Milk", "Bread"]},
            )

        assert resp.status_code == 200
        assert resp.get_json()["unchecked"] == 2
        mock_svc.uncheck_items.assert_called_once_with("abc12345", ["Milk", "Bread"])

    def test_uncheck_items_missing_items_returns_400(self, client):
        """PUT /lists/<id>/items/uncheck without items returns 400."""
        with patch("api.lists._get_list_service"):
            resp = client.put("/lists/abc12345/items/uncheck", json={})

        assert resp.status_code == 400
