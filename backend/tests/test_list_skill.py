"""Tests for the list innate skill (list_skill.py).

Uses the real ListService against a fully-migrated SQLite database so
persistence behaviour is verified end-to-end, not mocked.
"""

import pytest

from services.innate_skills.list_skill import handle_list
from tests._tag_helpers import assert_both_markers, extract_body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(params: dict) -> dict | str:
    """Call handle_list and return the JSON-parsed body.

    Result format (new): [list(action=X)]\\n<json_body>\\n[end:list]
    Both markers are always asserted. For string-only bodies (list_all,
    clear, rename), the raw body string is returned instead of a dict.
    """
    result = handle_list("test-topic", params)
    assert_both_markers("list", result)
    body = extract_body("list", result)
    try:
        import json
        return json.loads(body)
    except Exception:
        return body


def _items(result: dict) -> list[str]:
    return [i["content"] for i in result["list"]["items"]]


def _checked(result: dict) -> dict[str, bool]:
    return {i["content"]: i["checked"] for i in result["list"]["items"]}


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAdd:
    def test_create_empty_list(self, db):
        r = _call({"action": "add", "name": "Groceries", "items": []})
        assert r["status"] == "success"
        assert r["list"]["name"] == "Groceries"
        assert r["list"]["items"] == []

    def test_create_and_populate(self, db):
        r = _call({"action": "add", "name": "Groceries", "items": ["Milk", "Eggs"]})
        assert r["status"] == "success"
        assert set(_items(r)) == {"Milk", "Eggs"}

    def test_add_to_existing_list(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        r = _call({"action": "add", "name": "Groceries", "items": ["Eggs"]})
        assert r["status"] == "success"
        assert set(_items(r)) == {"Milk", "Eggs"}

    def test_deduplication(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        r = _call({"action": "add", "name": "Groceries", "items": ["Milk", "Eggs"]})
        assert r["status"] == "success"
        # Milk deduped, Eggs added — list should have exactly 2 items
        assert len(r["list"]["items"]) == 2

    def test_auto_creates_shopping_list_when_no_lists(self, db):
        r = _call({"action": "add", "items": ["Bread"]})
        assert r["status"] == "success"
        assert r["list"]["name"] == "Shopping List"
        assert _items(r) == ["Bread"]

    def test_auto_resolves_single_existing_list(self, db):
        _call({"action": "add", "name": "Tasks", "items": []})
        r = _call({"action": "add", "items": ["Buy coffee"]})
        assert r["status"] == "success"
        assert r["list"]["name"] == "Tasks"

    def test_items_as_json_string_parsed_correctly(self, db):
        # LLM sometimes serialises items as a JSON string instead of a real array
        r = _call({"action": "add", "name": "Groceries", "items": '["milk", "eggs", "bread"]'})
        assert r["status"] == "success"
        assert len(r["list"]["items"]) == 3
        assert set(_items(r)) == {"milk", "eggs", "bread"}

    def test_items_checked_false_by_default(self, db):
        r = _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        assert r["list"]["items"][0]["checked"] is False

    def test_result_contains_name_and_items_keys(self, db):
        r = _call({"action": "add", "name": "Test", "items": []})
        assert "name" in r["list"]
        assert "items" in r["list"]


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRemove:
    def test_remove_specific_items(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk", "Eggs", "Bread"]})
        r = _call({"action": "remove", "name": "Groceries", "items": ["Eggs"]})
        assert r["status"] == "success"
        assert "Eggs" not in _items(r)
        assert "Milk" in _items(r)
        assert "Bread" in _items(r)

    def test_remove_all_items_leaves_empty_list(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        r = _call({"action": "remove", "name": "Groceries", "items": ["Milk"]})
        assert r["status"] == "success"
        assert r["list"]["items"] == []

    def test_no_items_deletes_entire_list(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        r = _call({"action": "remove", "name": "Groceries"})
        assert r["status"] == "success"
        assert r["list"] is None

    def test_empty_items_array_deletes_entire_list(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        r = _call({"action": "remove", "name": "Groceries", "items": []})
        assert r["status"] == "success"
        assert r["list"] is None

    def test_delete_not_found_returns_fail(self, db):
        r = _call({"action": "remove", "name": "NonExistent"})
        assert r["status"] == "fail"
        assert "message" in r

    def test_auto_resolves_to_most_recent_when_multiple_lists(self, db):
        _call({"action": "add", "name": "List A", "items": []})
        _call({"action": "add", "name": "List B", "items": ["x"]})
        r = _call({"action": "remove", "items": ["x"]})
        # resolves to most-recently-updated list — should succeed
        assert r["status"] == "success"


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheck:
    def test_check_item(self, db):
        _call({"action": "add", "name": "Tasks", "items": ["Buy milk", "Write tests"]})
        r = _call({"action": "check", "name": "Tasks", "items": [
            {"content": "Buy milk", "checked": True},
        ]})
        assert r["status"] == "success"
        states = _checked(r)
        assert states["Buy milk"] is True
        assert states["Write tests"] is False

    def test_uncheck_item(self, db):
        _call({"action": "add", "name": "Tasks", "items": ["Buy milk"]})
        _call({"action": "check", "name": "Tasks", "items": [{"content": "Buy milk", "checked": True}]})
        r = _call({"action": "check", "name": "Tasks", "items": [
            {"content": "Buy milk", "checked": False},
        ]})
        assert r["status"] == "success"
        assert _checked(r)["Buy milk"] is False

    def test_mixed_check_and_uncheck_in_one_call(self, db):
        _call({"action": "add", "name": "Tasks", "items": ["A", "B", "C"]})
        # Check A and C, leave B
        _call({"action": "check", "name": "Tasks", "items": [
            {"content": "A", "checked": True},
            {"content": "C", "checked": True},
        ]})
        # Now uncheck A, keep B unchecked, keep C checked
        r = _call({"action": "check", "name": "Tasks", "items": [
            {"content": "A", "checked": False},
            {"content": "B", "checked": False},
            {"content": "C", "checked": True},
        ]})
        assert r["status"] == "success"
        states = _checked(r)
        assert states["A"] is False
        assert states["B"] is False
        assert states["C"] is True

    def test_returns_full_list(self, db):
        _call({"action": "add", "name": "Tasks", "items": ["A", "B"]})
        r = _call({"action": "check", "name": "Tasks", "items": [{"content": "A", "checked": True}]})
        assert r["status"] == "success"
        assert len(r["list"]["items"]) == 2

    def test_empty_items_returns_fail(self, db):
        _call({"action": "add", "name": "Tasks", "items": ["A"]})
        r = _call({"action": "check", "name": "Tasks", "items": []})
        assert r["status"] == "fail"
        assert "message" in r

    def test_list_not_found_returns_fail(self, db):
        r = _call({"action": "check", "name": "Ghost", "items": [
            {"content": "x", "checked": True},
        ]})
        assert r["status"] == "fail"
        assert "message" in r


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestView:
    def test_view_returns_full_list(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk", "Eggs"]})
        r = _call({"action": "view", "name": "Groceries"})
        assert r["status"] == "success"
        assert r["list"]["name"] == "Groceries"
        assert set(_items(r)) == {"Milk", "Eggs"}

    def test_view_empty_list(self, db):
        _call({"action": "add", "name": "Empty", "items": []})
        r = _call({"action": "view", "name": "Empty"})
        assert r["status"] == "success"
        assert r["list"]["items"] == []

    def test_view_not_found_returns_fail(self, db):
        r = _call({"action": "view", "name": "Ghost"})
        assert r["status"] == "fail"
        assert "message" in r

    def test_view_auto_resolves_single_list(self, db):
        _call({"action": "add", "name": "Solo", "items": ["X"]})
        r = _call({"action": "view"})
        assert r["status"] == "success"
        assert r["list"]["name"] == "Solo"

    def test_view_auto_resolves_to_most_recent_when_multiple_lists(self, db):
        _call({"action": "add", "name": "A", "items": []})
        _call({"action": "add", "name": "B", "items": ["x"]})
        r = _call({"action": "view"})
        # resolves to most-recently-updated list — should succeed
        assert r["status"] == "success"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestListAll:
    def test_list_all_empty(self, db):
        r = _call({"action": "list_all"})
        assert "No lists" in r

    def test_list_all_shows_all_lists(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk"]})
        _call({"action": "add", "name": "Tasks", "items": []})
        r = _call({"action": "list_all"})
        assert "Groceries" in r
        assert "Tasks" in r


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClear:
    def test_clear_removes_all_items(self, db):
        _call({"action": "add", "name": "Groceries", "items": ["Milk", "Eggs"]})
        r = _call({"action": "clear", "name": "Groceries"})
        assert "Cleared" in r

        view = _call({"action": "view", "name": "Groceries"})
        assert view["list"]["items"] == []

    def test_clear_missing_name(self, db):
        r = _call({"action": "clear"})
        assert "required" in r

    def test_clear_not_found(self, db):
        r = _call({"action": "clear", "name": "Ghost"})
        assert "not found" in r


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRename:
    def test_rename_list(self, db):
        _call({"action": "add", "name": "Old Name", "items": []})
        r = _call({"action": "rename", "name": "Old Name", "new_name": "New Name"})
        assert "New Name" in r

        view = _call({"action": "view", "name": "New Name"})
        assert view["status"] == "success"

    def test_rename_missing_params(self, db):
        r = _call({"action": "rename", "name": "X"})
        assert "required" in r


# ---------------------------------------------------------------------------
# unknown action
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnknownAction:
    def test_unknown_action_returns_fail_json(self, db):
        r = _call({"action": "fly"})
        assert r["status"] == "fail"
        assert "message" in r
