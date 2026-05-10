"""
Unit tests for ListService.

Strict id-only addressing: existing lists are looked up by 8-char hex id;
``name`` is used only for ``create_list`` and ``rename_list``. Uses real
SQLite via the shared ``db`` fixture — no mocked database connections.
"""

import pytest
from services.time_utils import utc_now


pytestmark = pytest.mark.unit


@pytest.fixture
def service(db):
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def _seed_list(db, list_id='abc12345', name='Shopping List',
               list_type='checklist', deleted_at=None):
    db.execute(
        "INSERT INTO lists (id, name, list_type, created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now'), ?)",
        (list_id, name, list_type, deleted_at),
    )
    db.commit()


def _seed_item(db, item_id, list_id, content, checked=0, position=0,
               removed_at=None):
    db.execute(
        "INSERT INTO list_items (id, list_id, content, checked, position, "
        "added_at, updated_at, removed_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)",
        (item_id, list_id, content, checked, position, removed_at),
    )
    db.commit()


# ─── create_list ──────────────────────────────────────────────────────────

class TestCreateList:
    def test_creates_list_and_returns_id(self, service, db):
        list_id = service.create_list("Shopping List")

        assert len(list_id) == 8
        row = db.execute(
            "SELECT name, list_type FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        assert row['name'] == 'Shopping List'
        assert row['list_type'] == 'checklist'

    def test_create_raises_on_duplicate_name(self, service):
        service.create_list("Shopping List")
        with pytest.raises(ValueError, match="already exists"):
            service.create_list("Shopping List")

    def test_create_with_custom_type(self, service, db):
        list_id = service.create_list("Work", list_type="ordered")
        row = db.execute(
            "SELECT list_type FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        assert row['list_type'] == 'ordered'


# ─── get_list ─────────────────────────────────────────────────────────────

class TestGetList:
    def test_returns_list_with_items_by_id(self, service, db):
        _seed_list(db)
        _seed_item(db, 'item0001', 'abc12345', 'milk', checked=0, position=0)
        _seed_item(db, 'item0002', 'abc12345', 'eggs', checked=1, position=1)

        result = service.get_list('abc12345')

        assert result is not None
        assert result['id'] == 'abc12345'
        assert result['name'] == 'Shopping List'
        assert [i['content'] for i in result['items']] == ['milk', 'eggs']
        assert result['items'][1]['checked'] == 1

    def test_returns_none_for_unknown_id(self, service):
        assert service.get_list('nonexist') is None

    def test_does_not_fall_back_to_name(self, service, db):
        _seed_list(db)
        # Passing the name (not id) must not resolve.
        assert service.get_list('Shopping List') is None

    def test_excludes_removed_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'eggs', position=0)
        _seed_item(db, 'i2', 'abc12345', 'milk', position=1,
                   removed_at=utc_now().isoformat())

        result = service.get_list('abc12345')
        assert [i['content'] for i in result['items']] == ['eggs']

    def test_deleted_list_returns_none(self, service, db):
        _seed_list(db, deleted_at=utc_now().isoformat())
        assert service.get_list('abc12345') is None


# ─── delete_list ──────────────────────────────────────────────────────────

class TestDeleteList:
    def test_soft_deletes_by_id(self, service, db):
        _seed_list(db)
        assert service.delete_list('abc12345') is True
        row = db.execute(
            "SELECT deleted_at FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['deleted_at'] is not None

    def test_returns_false_when_not_found(self, service):
        assert service.delete_list('nonexist') is False

    def test_does_not_fall_back_to_name(self, service, db):
        _seed_list(db)
        assert service.delete_list('Shopping List') is False


# ─── clear_list ───────────────────────────────────────────────────────────

class TestClearList:
    def test_clears_all_items_by_id(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        assert service.clear_list('abc12345') == 2
        active = db.execute(
            "SELECT COUNT(*) FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL"
        ).fetchone()[0]
        assert active == 0

    def test_returns_minus_one_when_not_found(self, service):
        assert service.clear_list('nonexist') == -1


# ─── rename_list ──────────────────────────────────────────────────────────

class TestRenameList:
    def test_renames_by_id(self, service, db):
        _seed_list(db, name='Old Name')
        assert service.rename_list('abc12345', 'New Name') is True
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'New Name'

    def test_blocks_rename_on_name_collision(self, service, db):
        _seed_list(db, list_id='abc12345', name='Old')
        _seed_list(db, list_id='def67890', name='Taken')
        assert service.rename_list('abc12345', 'Taken') is False
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'Old'

    def test_returns_false_when_id_not_found(self, service):
        assert service.rename_list('nonexist', 'Anything') is False


# ─── add_items ────────────────────────────────────────────────────────────

class TestAddItems:
    def test_adds_new_items_by_id(self, service, db):
        _seed_list(db)
        assert service.add_items('abc12345', ['milk', 'eggs']) == 2
        rows = db.execute(
            "SELECT content FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL "
            "ORDER BY position"
        ).fetchall()
        assert [r['content'] for r in rows] == ['milk', 'eggs']

    def test_dedupes_case_insensitive(self, service, db):
        _seed_list(db)
        _seed_item(db, 'existing1', 'abc12345', 'milk', position=0)
        assert service.add_items('abc12345', ['Milk', 'MILK']) == 0

    def test_dedupe_false_allows_duplicates(self, service, db):
        _seed_list(db)
        assert service.add_items('abc12345', ['milk', 'milk'], dedupe=False) == 2

    def test_does_not_auto_create(self, service, db):
        # No list seeded — must not create one and must return 0.
        assert service.add_items('nonexist', ['milk']) == 0
        rows = db.execute("SELECT COUNT(*) FROM lists").fetchone()[0]
        assert rows == 0

    def test_restores_soft_deleted_item(self, service, db):
        _seed_list(db)
        _seed_item(db, 'olditemid', 'abc12345', 'milk', position=0,
                   removed_at=utc_now().isoformat())
        assert service.add_items('abc12345', ['milk']) == 1
        row = db.execute(
            "SELECT removed_at, checked FROM list_items WHERE id = 'olditemid'"
        ).fetchone()
        assert row['removed_at'] is None
        assert row['checked'] == 0

    def test_skips_whitespace_and_empty(self, service, db):
        _seed_list(db)
        assert service.add_items('abc12345', ['  ', '\t', '']) == 0


# ─── remove_items ─────────────────────────────────────────────────────────

class TestRemoveItems:
    def test_removes_items_by_content(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        assert service.remove_items('abc12345', ['Milk']) == 1
        row = db.execute(
            "SELECT removed_at FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['removed_at'] is not None

    def test_returns_zero_when_list_not_found(self, service):
        assert service.remove_items('nonexist', ['milk']) == 0

    def test_returns_zero_when_no_match(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        assert service.remove_items('abc12345', ['bread']) == 0


# ─── check / uncheck ──────────────────────────────────────────────────────

class TestCheckUncheck:
    def test_checks_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)
        assert service.check_items('abc12345', ['milk', 'eggs']) == 2

    def test_unchecks_items(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', checked=1, position=0)
        assert service.uncheck_items('abc12345', ['milk']) == 1

    def test_check_unknown_item_returns_zero(self, service, db):
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        assert service.check_items('abc12345', ['bread']) == 0

    def test_check_empty_input_returns_zero(self, service, db):
        _seed_list(db)
        assert service.check_items('abc12345', []) == 0


# ─── _get_list_row ────────────────────────────────────────────────────────

class TestGetListRow:
    def test_resolves_by_id(self, service, db):
        _seed_list(db)
        result = service._get_list_row('abc12345')
        assert result['id'] == 'abc12345'
        assert result['name'] == 'Shopping List'

    def test_does_not_resolve_by_name(self, service, db):
        _seed_list(db)
        assert service._get_list_row('Shopping List') is None
        assert service._get_list_row('shopping list') is None

    def test_excludes_deleted(self, service, db):
        _seed_list(db, deleted_at=utc_now().isoformat())
        assert service._get_list_row('abc12345') is None

    def test_returns_none_for_empty(self, service):
        assert service._get_list_row('') is None
