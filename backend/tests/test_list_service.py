"""
Unit tests for ListService.

Uses real SQLite via the shared `db` fixture -- no mocked database connections.
"""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service(db):
    """ListService wired to the real per-test SQLite database."""
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def _seed_list(db, list_id='abc12345', name='Shopping List',
               list_type='checklist', deleted_at=None):
    """Insert a list row directly into the database."""
    db.execute(
        "INSERT INTO lists (id, name, list_type, created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now'), ?)",
        (list_id, name, list_type, deleted_at),
    )
    db.commit()


def _seed_item(db, item_id, list_id, content, checked=0, position=0,
               removed_at=None):
    """Insert a list_item row directly into the database."""
    db.execute(
        "INSERT INTO list_items (id, list_id, content, checked, position, "
        "added_at, updated_at, removed_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)",
        (item_id, list_id, content, checked, position, removed_at),
    )
    db.commit()


def _seed_event(db, event_id, list_id, event_type, item_content=None,
                details='{}'):
    """Insert a list_event row directly into the database."""
    db.execute(
        "INSERT INTO list_events (id, list_id, event_type, item_content, "
        "details, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (event_id, list_id, event_type, item_content, details),
    )
    db.commit()


# ---------------------------------------------------------------------------
# List CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateList:
    def test_creates_list_and_returns_id(self, service, db):
        list_id = service.create_list("Shopping List")

        assert len(list_id) == 8
        # Verify it was actually written to the database
        row = db.execute(
            "SELECT id, name, list_type FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        assert row is not None
        assert row['name'] == 'Shopping List'
        assert row['list_type'] == 'checklist'

    def test_create_raises_on_duplicate(self, service, db):
        service.create_list("Shopping List")

        with pytest.raises(ValueError, match="already exists"):
            service.create_list("Shopping List")

    def test_create_with_custom_type(self, service, db):
        list_id = service.create_list("Work Tasks", list_type="ordered")

        row = db.execute(
            "SELECT list_type FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        assert row['list_type'] == 'ordered'


@pytest.mark.unit
class TestGetList:
    def test_returns_list_with_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'item0001', 'abc12345', 'milk', checked=0, position=0)
        _seed_item(db, 'item0002', 'abc12345', 'eggs', checked=1, position=1)

        result = service.get_list("Shopping List")

        assert result is not None
        assert result['name'] == 'Shopping List'
        assert len(result['items']) == 2
        assert result['items'][0]['content'] == 'milk'
        assert result['items'][1]['checked'] == 1

    def test_returns_none_for_unknown_list(self, service, db):
        result = service.get_list("Nonexistent")
        assert result is None

    def test_empty_list_returns_empty_items(self, service, db):
        _seed_list(db, name='Empty List')

        result = service.get_list("Empty List")
        assert result['items'] == []


@pytest.mark.unit
class TestDeleteList:
    def test_soft_deletes_list(self, service, db):
        _seed_list(db)

        result = service.delete_list("Shopping List")
        assert result is True

        # Verify it was soft-deleted
        row = db.execute(
            "SELECT deleted_at FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['deleted_at'] is not None

    def test_returns_false_when_not_found(self, service, db):
        result = service.delete_list("Ghost List")
        assert result is False


@pytest.mark.unit
class TestClearList:
    def test_clears_all_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)
        _seed_item(db, 'i3', 'abc12345', 'bread', position=2)

        result = service.clear_list("Shopping List")
        assert result == 3

        # Verify all items are soft-deleted
        active = db.execute(
            "SELECT COUNT(*) FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL"
        ).fetchone()[0]
        assert active == 0

    def test_returns_minus_one_when_not_found(self, service, db):
        result = service.clear_list("Ghost List")
        assert result == -1


@pytest.mark.unit
class TestRenameList:
    def test_renames_successfully(self, service, db):
        _seed_list(db, name='Old Name')

        result = service.rename_list("Old Name", "New Name")
        assert result is True

        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'New Name'

    def test_blocks_rename_on_collision(self, service, db):
        _seed_list(db, list_id='abc12345', name='Old Name')
        _seed_list(db, list_id='def67890', name='New Name')

        result = service.rename_list("Old Name", "New Name")
        assert result is False

        # Verify old name was not changed
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'Old Name'

    def test_rename_to_same_id_is_allowed(self, service, db):
        """Renaming to same ID is not a collision (same list)."""
        _seed_list(db, name='Old Name')

        result = service.rename_list("Old Name", "Old Name v2")
        assert result is True

        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'Old Name v2'


# ---------------------------------------------------------------------------
# Item operations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAddItems:
    def test_adds_new_items(self, service, db):
        _seed_list(db)

        added = service.add_items("Shopping List", ["milk", "eggs"])
        assert added == 2

        rows = db.execute(
            "SELECT content FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL "
            "ORDER BY position"
        ).fetchall()
        assert [r['content'] for r in rows] == ['milk', 'eggs']

    def test_dedupes_by_default(self, service, db):
        _seed_list(db)
        _seed_item(db, 'existing1', 'abc12345', 'milk', position=0)

        added = service.add_items("Shopping List", ["milk", "eggs"])
        assert added == 1  # only eggs added

    def test_deduplication_is_case_insensitive(self, service, db):
        _seed_list(db)
        _seed_item(db, 'existing1', 'abc12345', 'milk', position=0)

        added = service.add_items("Shopping List", ["Milk", "MILK"])
        assert added == 0

    def test_dedupe_false_allows_duplicates(self, service, db):
        _seed_list(db)

        added = service.add_items("Shopping List", ["milk", "milk"], dedupe=False)
        assert added == 2

    def test_auto_creates_list(self, service, db):
        added = service.add_items("New List", ["item1"], auto_create=True)
        assert added == 1

        # Verify the list was auto-created
        row = db.execute(
            "SELECT name FROM lists WHERE LOWER(name) = LOWER('New List') "
            "AND deleted_at IS NULL"
        ).fetchone()
        assert row is not None
        assert row['name'] == 'New List'

    def test_restores_soft_deleted_item(self, service, db):
        _seed_list(db)
        _seed_item(db, 'olditemid', 'abc12345', 'milk', position=0,
                   removed_at=datetime.now(timezone.utc).isoformat())

        added = service.add_items("Shopping List", ["milk"])
        assert added == 1

        # Verify the item was restored (removed_at cleared)
        row = db.execute(
            "SELECT removed_at, checked FROM list_items WHERE id = 'olditemid'"
        ).fetchone()
        assert row['removed_at'] is None
        assert row['checked'] == 0


@pytest.mark.unit
class TestRemoveItems:
    def test_removes_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)

        removed = service.remove_items("Shopping List", ["milk"])
        assert removed == 1

        row = db.execute(
            "SELECT removed_at FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['removed_at'] is not None

    def test_returns_zero_when_list_not_found(self, service, db):
        removed = service.remove_items("Ghost", ["milk"])
        assert removed == 0


@pytest.mark.unit
class TestCheckUncheck:
    def test_checks_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        count = service.check_items("Shopping List", ["milk", "eggs"])
        assert count == 2

        rows = db.execute(
            "SELECT checked FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL "
            "ORDER BY position"
        ).fetchall()
        assert all(r['checked'] == 1 for r in rows)

    def test_unchecks_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', checked=1, position=0)

        count = service.uncheck_items("Shopping List", ["milk"])
        assert count == 1

        row = db.execute(
            "SELECT checked FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['checked'] == 0


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveList:
    def test_resolves_by_id(self, service, db):
        _seed_list(db)

        result = service._resolve_list('abc12345')
        assert result['id'] == 'abc12345'

    def test_resolves_by_name_case_insensitive(self, service, db):
        _seed_list(db)

        result = service._resolve_list('shopping list')
        assert result['name'] == 'Shopping List'

    def test_returns_none_when_not_found(self, service, db):
        result = service._resolve_list('nonexistent')
        assert result is None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetHistory:
    def test_returns_events_for_list(self, service, db):
        _seed_list(db)
        _seed_event(db, 'evt00001', 'abc12345', 'item_added', 'milk')
        _seed_event(db, 'evt00002', 'abc12345', 'item_checked', 'milk')

        events = service.get_history("Shopping List")
        assert len(events) == 2
        # Events are returned in DESC order, so most recent first
        event_types = {e['event_type'] for e in events}
        assert 'item_added' in event_types
        assert 'item_checked' in event_types
        # Verify item_content is present
        contents = {e['item_content'] for e in events}
        assert 'milk' in contents

    def test_returns_empty_when_list_not_found(self, service, db):
        events = service.get_history("Ghost List")
        assert events == []


# ---------------------------------------------------------------------------
# Soft delete behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSoftDelete:
    def test_deleted_list_not_resolvable_by_name(self, service, db):
        """Once deleted, the list should not appear in resolution."""
        _seed_list(db, deleted_at=datetime.now(timezone.utc).isoformat())

        result = service._resolve_list("Shopping List")
        assert result is None

    def test_removed_item_excluded_from_get_list(self, service, db):
        _seed_list(db)
        _seed_item(db, 'item0001', 'abc12345', 'eggs', position=0)
        _seed_item(db, 'item0002', 'abc12345', 'milk', position=1,
                   removed_at=datetime.now(timezone.utc).isoformat())

        result = service.get_list("Shopping List")
        assert len(result['items']) == 1
        assert result['items'][0]['content'] == 'eggs'


# ---------------------------------------------------------------------------
# get_lists_for_prompt
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetListsForPrompt:
    def test_returns_empty_when_no_lists(self, service):
        with patch.object(service, 'get_all_lists', return_value=[]):
            result = service.get_lists_for_prompt()
        assert result == ""

    def test_formats_single_list(self, service):
        updated = datetime.now(timezone.utc) - timedelta(hours=2)
        with patch.object(service, 'get_all_lists', return_value=[
            {'id': 'abc12345', 'name': 'Shopping List', 'list_type': 'checklist',
             'item_count': 5, 'checked_count': 2, 'updated_at': updated}
        ]):
            result = service.get_lists_for_prompt()

        assert "## Active Lists" in result
        assert "Shopping List" in result
        assert "5 items" in result
        assert "2 checked" in result
        assert "ago" in result

    def test_formats_multiple_lists(self, service):
        updated1 = datetime.now(timezone.utc) - timedelta(minutes=30)
        updated2 = datetime.now(timezone.utc) - timedelta(days=3)
        with patch.object(service, 'get_all_lists', return_value=[
            {'id': 'aaa', 'name': 'Shopping List', 'list_type': 'checklist',
             'item_count': 3, 'checked_count': 0, 'updated_at': updated1},
            {'id': 'bbb', 'name': 'Chores', 'list_type': 'checklist',
             'item_count': 4, 'checked_count': 1, 'updated_at': updated2},
        ]):
            result = service.get_lists_for_prompt()

        assert "Shopping List" in result
        assert "Chores" in result

    def test_no_checked_count_omitted(self, service):
        updated = datetime.now(timezone.utc) - timedelta(hours=1)
        with patch.object(service, 'get_all_lists', return_value=[
            {'id': 'abc', 'name': 'To Do', 'list_type': 'checklist',
             'item_count': 3, 'checked_count': 0, 'updated_at': updated}
        ]):
            result = service.get_lists_for_prompt()

        assert "checked" not in result
        assert "3 items" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    def test_add_empty_items_list_returns_zero(self, service, db):
        _seed_list(db)

        added = service.add_items("Shopping List", [])
        assert added == 0

    def test_add_whitespace_only_items_skipped(self, service, db):
        _seed_list(db)

        added = service.add_items("Shopping List", ["  ", "\t", ""])
        assert added == 0

    def test_check_items_empty_list_returns_zero(self, service, db):
        _seed_list(db)

        count = service.check_items("Shopping List", [])
        assert count == 0

    def test_get_most_recent_list(self, service, db):
        _seed_list(db)

        result = service.get_most_recent_list()
        assert result['name'] == 'Shopping List'

    def test_get_most_recent_list_none_when_empty(self, service, db):
        result = service.get_most_recent_list()
        assert result is None
