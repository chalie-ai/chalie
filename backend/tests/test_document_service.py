# Tests for DocumentService migrated from mock_db to real SQLite via shared `db` fixture.

import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import pytest

from services.document_service import DocumentService
from services.database_service import get_shared_db_service
from services.write_queue_service import WriteQueueService

if TYPE_CHECKING:
    pass


class _InlineWriteQueue:
    """Minimal write queue that executes closures inline (same thread).

    Avoids thread-local SQLite connection issues in tests by running
    the DocumentService's write closures on the calling thread instead
    of a background daemon.
    """

    def submit_sync(self, fn: Callable[..., object], *args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    def submit(self, fn: Callable[..., object], *args: object, **kwargs: object) -> None:
        fn(*args, **kwargs)


def _insert_document(db: sqlite3.Connection, doc_id: str = 'abc123', original_name: str = 'test.pdf',
                     mime_type: str = 'application/pdf', file_size: int = 1024,
                     file_path: str = 'abc123/test.pdf', file_hash: str = 'sha256hash',
                     source_type: str = 'upload', status: str = 'pending', **extra: object) -> str:
    """Seed a document row directly for read-path tests."""
    cols: dict[str, object] = dict(
        id=doc_id,
        original_name=original_name,
        mime_type=mime_type,
        file_size_bytes=file_size,
        file_path=file_path,
        file_hash=file_hash,
        source_type=source_type,
        status=status,
    )
    cols.update(extra)
    col_names = ', '.join(cols.keys())
    placeholders = ', '.join(['?'] * len(cols))
    db.execute(
        f"INSERT INTO documents ({col_names}) VALUES ({placeholders})",
        list(cols.values()),
    )
    db.commit()
    return doc_id


@pytest.fixture
def doc_service(db: sqlite3.Connection) -> DocumentService:
    """DocumentService wired to the test database with inline write queue."""
    svc = DocumentService(get_shared_db_service())
    svc._write_queue = cast(WriteQueueService, _InlineWriteQueue())
    return svc


@pytest.mark.unit
class TestCreateDocument:
    def test_creates_document_and_returns_id(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        doc_id = doc_service.create_document(
            original_name='test.pdf',
            mime_type='application/pdf',
            file_size=1024,
            file_path='abc123/test.pdf',
            file_hash='sha256hash',
            source_type='upload',
        )

        assert doc_id is not None
        assert len(doc_id) == 8  # secrets.token_hex(4)

        # Verify row exists in DB
        row = db.execute(
            "SELECT id, original_name, mime_type FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
        assert row is not None
        assert row['original_name'] == 'test.pdf'


@pytest.mark.unit
class TestGetDocument:
    def test_returns_none_when_not_found(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        result = doc_service.get_document('nonexistent')
        assert result is None

    def test_returns_dict_when_found(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='abc123', status='ready')

        result = doc_service.get_document('abc123')
        assert result is not None
        assert result['id'] == 'abc123'
        assert result['original_name'] == 'test.pdf'
        assert result['status'] == 'ready'


@pytest.mark.unit
class TestSoftDelete:
    def test_soft_delete_sets_deleted_at(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='abc123')

        result = doc_service.soft_delete('abc123')
        assert result is True

        row = db.execute(
            "SELECT deleted_at FROM documents WHERE id = 'abc123'"
        ).fetchone()
        assert row['deleted_at'] is not None


@pytest.mark.unit
class TestRestore:
    def test_restore_clears_deleted_at(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='abc123')
        doc_service.soft_delete('abc123')

        result = doc_service.restore('abc123')
        assert result is True

        row = db.execute(
            "SELECT deleted_at FROM documents WHERE id = 'abc123'"
        ).fetchone()
        assert row['deleted_at'] is None



@pytest.mark.unit
class TestGetAllDocuments:
    def test_excludes_deleted_by_default(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='live1')
        _insert_document(db, doc_id='dead1')
        doc_service.soft_delete('dead1')

        result = doc_service.get_all_documents(include_deleted=False)
        ids = [d['id'] for d in result]
        assert 'live1' in ids
        assert 'dead1' not in ids


@pytest.mark.unit
class TestFindDuplicates:
    def test_finds_exact_hash_match(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='dup1', original_name='existing.pdf',
                         file_hash='same_hash')

        results = doc_service.find_duplicates('same_hash', None, 0)
        assert len(results) == 1
        assert results[0]['match_type'] == 'exact'


@pytest.mark.unit
class TestUpdateStatus:
    def test_updates_status(self, db: sqlite3.Connection, doc_service: DocumentService) -> None:
        _insert_document(db, doc_id='abc123', status='pending')

        doc_service.update_status('abc123', 'processing')

        row = db.execute(
            "SELECT status FROM documents WHERE id = 'abc123'"
        ).fetchone()
        assert row['status'] == 'processing'
