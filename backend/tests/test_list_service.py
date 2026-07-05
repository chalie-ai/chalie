import sqlite3
from typing import Optional, cast

import pytest

from services.list_service import ListService
from services.time_utils import utc_now

pytestmark = pytest.mark.unit


@pytest.fixture
def service(db: sqlite3.Connection) -> ListService:
    return ListService()


def _seed_list(db: sqlite3.Connection, list_id: str = 'abc12345', name: str = 'Shopping List',
               list_type: str = 'checklist', deleted_at: Optional[str] = None) -> None:
    db.execute(
        "INSERT INTO lists (id, name, list_type, created_at, updated_at, deleted_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now'), ?)",
        (list_id, name, list_type, deleted_at),
    )
    db.commit()


def _seed_item(db: sqlite3.Connection, item_id: str, list_id: str, content: str,
               checked: int = 0, position: int = 0,
               removed_at: Optional[str] = None) -> None:
    db.execute(
        "INSERT INTO list_items (id, list_id, content, checked, position, "
        "added_at, updated_at, removed_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)",
        (item_id, list_id, content, checked, position, removed_at),
    )
    db.commit()


# ─── create_list ──────────────────────────────────────────────────────────

class TestCreateList:
    def test_creates_list_and_returns_id(self, service: ListService, db: sqlite3.Connection) -> None:
        list_id = service.create_list("Shopping List")

        assert len(list_id) == 8
        row = db.execute(
            "SELECT name, list_type FROM lists WHERE id = ?", (list_id,)
        ).fetchone()
        assert row['name'] == 'Shopping List'
        assert row['list_type'] == 'checklist'

    def test_create_raises_on_duplicate_name(self, service: ListService) -> None:
        service.create_list("Shopping List")
        with pytest.raises(ValueError, match="already exists"):
            service.create_list("Shopping List")


# ─── get_list ─────────────────────────────────────────────────────────────

class TestGetList:
    def test_returns_list_with_items_by_id(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'item0001', 'abc12345', 'milk', checked=0, position=0)
        _seed_item(db, 'item0002', 'abc12345', 'eggs', checked=1, position=1)

        result = service.get_list('abc12345')

        assert result is not None
        assert result['id'] == 'abc12345'
        assert result['name'] == 'Shopping List'
        assert [cast(dict[str, object], i)['content'] for i in cast(list[object], result['items'])] == ['milk', 'eggs']
        assert cast(dict[str, object], cast(list[object], result['items'])[1])['checked'] == 1

    def test_excludes_removed_items(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'eggs', position=0)
        _seed_item(db, 'i2', 'abc12345', 'milk', position=1,
                   removed_at=utc_now().isoformat())

        result = service.get_list('abc12345')
        assert [cast(dict[str, object], i)['content'] for i in cast(list[object], cast("dict[str, object]", result)['items'])] == ['eggs']


# ─── delete_list ──────────────────────────────────────────────────────────

class TestDeleteList:
    def test_soft_deletes_by_id(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        assert service.delete_list('abc12345') is True
        row = db.execute(
            "SELECT deleted_at FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['deleted_at'] is not None

# ─── clear_list ───────────────────────────────────────────────────────────

class TestClearList:
    def test_clears_all_items_by_id(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)

        assert service.clear_list('abc12345') == 2
        active = db.execute(
            "SELECT COUNT(*) FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL"
        ).fetchone()[0]
        assert active == 0



# ─── rename_list ──────────────────────────────────────────────────────────

class TestRenameList:
    def test_renames_by_id(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db, name='Old Name')
        assert service.rename_list('abc12345', 'New Name') is True
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'New Name'

    def test_blocks_rename_on_name_collision(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db, list_id='abc12345', name='Old')
        _seed_list(db, list_id='def67890', name='Taken')
        assert service.rename_list('abc12345', 'Taken') is False
        row = db.execute(
            "SELECT name FROM lists WHERE id = 'abc12345'"
        ).fetchone()
        assert row['name'] == 'Old'

# ─── add_items ────────────────────────────────────────────────────────────

class TestAddItems:
    def test_adds_new_items_by_id(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        assert service.add_items('abc12345', ['milk', 'eggs']) == 2
        rows = db.execute(
            "SELECT content FROM list_items "
            "WHERE list_id = 'abc12345' AND removed_at IS NULL "
            "ORDER BY position"
        ).fetchall()
        assert [r['content'] for r in rows] == ['milk', 'eggs']

    def test_dedupes_case_insensitive(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'existing1', 'abc12345', 'milk', position=0)
        assert service.add_items('abc12345', ['Milk', 'MILK']) == 0

    def test_restores_soft_deleted_item(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'olditemid', 'abc12345', 'milk', position=0,
                   removed_at=utc_now().isoformat())
        assert service.add_items('abc12345', ['milk']) == 1
        row = db.execute(
            "SELECT removed_at, checked FROM list_items WHERE id = 'olditemid'"
        ).fetchone()
        assert row['removed_at'] is None
        assert row['checked'] == 0



# ─── remove_items ─────────────────────────────────────────────────────────

class TestRemoveItems:
    def test_removes_items_by_content(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        assert service.remove_items('abc12345', ['Milk']) == 1
        row = db.execute(
            "SELECT removed_at FROM list_items WHERE id = 'i1'"
        ).fetchone()
        assert row['removed_at'] is not None

# ─── check / uncheck ──────────────────────────────────────────────────────

class TestCheckUncheck:
    def test_checks_items(self, service: ListService, db: sqlite3.Connection) -> None:
        _seed_list(db)
        _seed_item(db, 'i1', 'abc12345', 'milk', position=0)
        _seed_item(db, 'i2', 'abc12345', 'eggs', position=1)
        assert service.check_items('abc12345', ['milk', 'eggs']) == 2







