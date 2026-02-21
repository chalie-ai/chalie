"""
Unit tests for ListService.

All tests mock the database connection — no external dependencies required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    """Mock DatabaseService with a context-managed connection."""
    db = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    db.connection.return_value.__enter__ = MagicMock(return_value=conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)
    return db, conn, cursor


@pytest.fixture
def service(mock_db):
    from services.list_service import ListService
    db, _, _ = mock_db
    return ListService(db)


# ---------------------------------------------------------------------------
# List CRUD
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateList:
    def test_creates_list_and_returns_id(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.rowcount = 1

        list_id = service.create_list("Shopping List")

        assert len(list_id) == 8
        assert cursor.execute.called

    def test_create_raises_on_duplicate(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.execute.side_effect = Exception("unique constraint violated")

        with pytest.raises(ValueError, match="already exists"):
            service.create_list("Shopping List")

    def test_create_with_custom_type(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.rowcount = 1
        service.create_list("Work Tasks", list_type="ordered")
        # Verify list_type was passed
        call_args = cursor.execute.call_args_list[0]
        assert "ordered" in str(call_args)


@pytest.mark.unit
class TestGetList:
    def _make_list_row(self):
        return ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))

    def _make_item_rows(self):
        return [
            ('item0001', 'milk', False, 0, datetime.now(timezone.utc), datetime.now(timezone.utc)),
            ('item0002', 'eggs', True, 1, datetime.now(timezone.utc), datetime.now(timezone.utc)),
        ]

    def test_returns_list_with_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [
            None,                       # ID lookup
            self._make_list_row(),      # name lookup
        ]
        cursor.fetchall.return_value = self._make_item_rows()

        result = service.get_list("Shopping List")

        assert result is not None
        assert result['name'] == 'Shopping List'
        assert len(result['items']) == 2
        assert result['items'][0]['content'] == 'milk'
        assert result['items'][1]['checked'] is True

    def test_returns_none_for_unknown_list(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None
        result = service.get_list("Nonexistent")
        assert result is None

    def test_empty_list_returns_empty_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Empty List', 'checklist', datetime.now(timezone.utc))]
        cursor.fetchall.return_value = []
        result = service.get_list("Empty List")
        assert result['items'] == []


@pytest.mark.unit
class TestDeleteList:
    def test_soft_deletes_list(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]
        cursor.rowcount = 1

        result = service.delete_list("Shopping List")
        assert result is True

    def test_returns_false_when_not_found(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None
        result = service.delete_list("Ghost List")
        assert result is False


@pytest.mark.unit
class TestClearList:
    def test_clears_all_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]
        cursor.rowcount = 5

        result = service.clear_list("Shopping List")
        assert result == 5

    def test_returns_minus_one_when_not_found(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None
        result = service.clear_list("Ghost List")
        assert result == -1


@pytest.mark.unit
class TestRenameList:
    def test_renames_successfully(self, service, mock_db):
        _, conn, cursor = mock_db
        # First resolve: finds old list; second resolve: no collision
        cursor.fetchone.side_effect = [
            None, ('abc12345', 'Old Name', 'checklist', datetime.now(timezone.utc)),  # resolve old
            None, None,  # resolve new name → not found (no collision)
        ]
        cursor.rowcount = 1

        result = service.rename_list("Old Name", "New Name")
        assert result is True

    def test_blocks_rename_on_collision(self, service, mock_db):
        _, conn, cursor = mock_db
        # First resolve: old list; second resolve: collision found
        cursor.fetchone.side_effect = [
            None, ('abc12345', 'Old Name', 'checklist', datetime.now(timezone.utc)),
            None, ('def67890', 'New Name', 'checklist', datetime.now(timezone.utc)),  # collision
        ]

        result = service.rename_list("Old Name", "New Name")
        assert result is False

    def test_rename_to_same_id_is_allowed(self, service, mock_db):
        """Renaming to same ID is not a collision (same list)."""
        _, conn, cursor = mock_db
        list_row = ('abc12345', 'Old Name', 'checklist', datetime.now(timezone.utc))
        cursor.fetchone.side_effect = [
            None, list_row,  # resolve old
            None, list_row,  # resolve new — same list_id → no collision
        ]
        cursor.rowcount = 1

        result = service.rename_list("Old Name", "Old Name v2")
        assert result is True


# ---------------------------------------------------------------------------
# Item operations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAddItems:
    def _list_row(self):
        return ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))

    def test_adds_new_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row(), (2,), None, None, None]
        cursor.fetchall.return_value = []  # no existing items
        cursor.rowcount = 1

        added = service.add_items("Shopping List", ["milk", "eggs"])
        assert added == 2

    def test_dedupes_by_default(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row(), (0,), None, None]
        cursor.fetchall.return_value = [("milk",)]  # milk already exists
        cursor.rowcount = 1

        added = service.add_items("Shopping List", ["milk", "eggs"])
        assert added == 1  # only eggs added

    def test_deduplication_is_case_insensitive(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row(), (0,), None]
        cursor.fetchall.return_value = [("milk",)]  # "milk" already exists
        cursor.rowcount = 1

        added = service.add_items("Shopping List", ["Milk", "MILK"])
        assert added == 0

    def test_dedupe_false_allows_duplicates(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row(), (0,), None, None]
        cursor.fetchall.return_value = []
        cursor.rowcount = 1

        added = service.add_items("Shopping List", ["milk", "milk"], dedupe=False)
        assert added == 2

    def test_auto_creates_list(self, service, mock_db):
        _, conn, cursor = mock_db
        # First _resolve_list returns None → triggers create_list
        # Then create_list inserts and _log_event runs
        # Then add_items resolves again
        cursor.fetchone.side_effect = [
            None, None,  # resolve → not found
            # create_list → insert (no fetchone needed)
            None, None,  # _log_event resolve (not called here in create path)
            # After create, add_items re-resolves:
            None, ('newid01', 'New List', 'checklist', datetime.now(timezone.utc)),
            (- 1,),  # max position
            # fetchall for existing items
        ]
        cursor.fetchall.return_value = []
        cursor.rowcount = 1

        # Patch create_list to return a predictable id
        with patch.object(service, 'create_list', return_value='newid01') as mock_create:
            with patch.object(service, '_resolve_list') as mock_resolve:
                mock_resolve.side_effect = [
                    None,  # first call → list not found → triggers create
                    {'id': 'newid01', 'name': 'New List', 'updated_at': datetime.now(timezone.utc)},
                ]
                added = service.add_items("New List", ["item1"], auto_create=True)
            mock_create.assert_called_once_with("New List", user_id='primary')

    def test_restores_soft_deleted_item(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [
            None, self._list_row(),
            (0,),   # max position
            ('olditemid',),  # found a previously removed row
        ]
        cursor.fetchall.return_value = []
        cursor.rowcount = 1

        added = service.add_items("Shopping List", ["milk"])
        assert added == 1
        # Should UPDATE not INSERT for restore
        update_calls = [c for c in cursor.execute.call_args_list if 'removed_at = NULL' in str(c)]
        assert len(update_calls) >= 1


@pytest.mark.unit
class TestRemoveItems:
    def test_removes_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]
        cursor.rowcount = 1

        removed = service.remove_items("Shopping List", ["milk"])
        assert removed == 1

    def test_returns_zero_when_list_not_found(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None

        removed = service.remove_items("Ghost", ["milk"])
        assert removed == 0


@pytest.mark.unit
class TestCheckUncheck:
    def _list_row(self):
        return ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))

    def test_checks_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row()]
        cursor.rowcount = 1  # 1 row affected per item

        count = service.check_items("Shopping List", ["milk", "eggs"])
        assert count == 2

    def test_unchecks_items(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, self._list_row()]
        cursor.rowcount = 1

        count = service.uncheck_items("Shopping List", ["milk"])
        assert count == 1


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveList:
    def test_resolves_by_id(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))

        result = service._resolve_list('abc12345')
        assert result['id'] == 'abc12345'

    def test_resolves_by_name_case_insensitive(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [
            None,  # ID lookup returns nothing
            ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc)),  # name lookup
        ]

        result = service._resolve_list('shopping list')
        assert result['name'] == 'Shopping List'

    def test_returns_none_when_not_found(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None

        result = service._resolve_list('nonexistent')
        assert result is None


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetHistory:
    def test_returns_events_for_list(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]
        cursor.fetchall.return_value = [
            ('evt00001', 'abc12345', 'item_added', 'milk', {}, datetime.now(timezone.utc)),
            ('evt00002', 'abc12345', 'item_checked', 'milk', {}, datetime.now(timezone.utc)),
        ]

        events = service.get_history("Shopping List")
        assert len(events) == 2
        assert events[0]['event_type'] == 'item_added'
        assert events[0]['item_content'] == 'milk'

    def test_returns_empty_when_list_not_found(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None

        events = service.get_history("Ghost List")
        assert events == []


# ---------------------------------------------------------------------------
# Soft delete behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSoftDelete:
    def test_deleted_list_not_resolvable_by_name(self, service, mock_db):
        """Once deleted, the list should not appear in resolution."""
        _, conn, cursor = mock_db
        # Simulate no active lists with this name (deleted_at is set)
        cursor.fetchone.return_value = None

        result = service._resolve_list("Deleted List")
        assert result is None

    def test_removed_item_excluded_from_get_list(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [
            None,
            ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc)),
        ]
        # Only active items returned (removed_at IS NULL filter in SQL)
        cursor.fetchall.return_value = [
            ('item0001', 'eggs', False, 0, datetime.now(timezone.utc), datetime.now(timezone.utc)),
        ]

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
    def test_add_empty_items_list_returns_zero(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]

        added = service.add_items("Shopping List", [])
        assert added == 0

    def test_add_whitespace_only_items_skipped(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]
        cursor.fetchone.return_value = None
        cursor.fetchall.return_value = []
        cursor.rowcount = 0

        added = service.add_items("Shopping List", ["  ", "\t", ""])
        assert added == 0

    def test_check_items_empty_list_returns_zero(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.side_effect = [None, ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))]

        count = service.check_items("Shopping List", [])
        assert count == 0

    def test_get_most_recent_list(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = ('abc12345', 'Shopping List', 'checklist', datetime.now(timezone.utc))

        result = service.get_most_recent_list()
        assert result['name'] == 'Shopping List'

    def test_get_most_recent_list_none_when_empty(self, service, mock_db):
        _, conn, cursor = mock_db
        cursor.fetchone.return_value = None

        result = service.get_most_recent_list()
        assert result is None
