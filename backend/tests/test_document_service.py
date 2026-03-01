"""
Tests for DocumentService — CRUD, chunk storage, soft delete, purge, search.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone, timedelta

from services.document_service import DocumentService


@pytest.fixture
def mock_db():
    db = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    db.connection.return_value.__enter__ = MagicMock(return_value=conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return db, cursor


@pytest.mark.unit
class TestCreateDocument:
    def test_creates_document_and_returns_id(self, mock_db):
        db, cursor = mock_db
        service = DocumentService(db)

        doc_id = service.create_document(
            original_name='test.pdf',
            mime_type='application/pdf',
            file_size=1024,
            file_path='abc123/test.pdf',
            file_hash='sha256hash',
            source_type='upload',
        )

        assert doc_id is not None
        assert len(doc_id) == 8  # secrets.token_hex(4)
        cursor.execute.assert_called_once()
        cursor.close.assert_called_once()

    def test_creates_document_with_camera_source(self, mock_db):
        db, cursor = mock_db
        service = DocumentService(db)

        doc_id = service.create_document(
            original_name='scan.jpg',
            mime_type='image/jpeg',
            file_size=2048,
            file_path='def456/scan.jpg',
            file_hash='sha256hash2',
            source_type='camera',
        )

        assert doc_id is not None
        # Verify source_type is passed to SQL
        sql_args = cursor.execute.call_args[0][1]
        assert 'camera' in sql_args


@pytest.mark.unit
class TestGetDocument:
    def test_returns_none_when_not_found(self, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = None
        service = DocumentService(db)

        result = service.get_document('nonexistent')
        assert result is None

    def test_returns_dict_when_found(self, mock_db):
        db, cursor = mock_db
        now = datetime.now(timezone.utc)
        cursor.fetchone.return_value = (
            'abc123', 'test.pdf', 'application/pdf', 1024, 'abc123/test.pdf',
            'hash', 5, 'ready', None, 10,
            'upload', ['tag1'], 'summary text', {'key': 'val'}, None,
            'clean text', 'en', 'fingerprint123',
            now, now, None, None,
        )
        service = DocumentService(db)

        result = service.get_document('abc123')
        assert result is not None
        assert result['id'] == 'abc123'
        assert result['original_name'] == 'test.pdf'
        assert result['status'] == 'ready'
        assert result['tags'] == ['tag1']


@pytest.mark.unit
class TestSoftDelete:
    def test_soft_delete_sets_deleted_at(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 1
        service = DocumentService(db)

        result = service.soft_delete('abc123')
        assert result is True
        cursor.execute.assert_called_once()

    def test_soft_delete_returns_false_when_not_found(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 0
        service = DocumentService(db)

        result = service.soft_delete('nonexistent')
        assert result is False


@pytest.mark.unit
class TestRestore:
    def test_restore_clears_deleted_at(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 1
        service = DocumentService(db)

        result = service.restore('abc123')
        assert result is True

    def test_restore_returns_false_when_not_deleted(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 0
        service = DocumentService(db)

        result = service.restore('abc123')
        assert result is False


@pytest.mark.unit
class TestStoreChunks:
    def test_stores_multiple_chunks(self, mock_db):
        db, cursor = mock_db
        service = DocumentService(db)

        chunks = [
            {'chunk_index': 0, 'content': 'Hello', 'page_number': 1,
             'section_title': 'Intro', 'token_count': 5, 'embedding': [0.1] * 768},
            {'chunk_index': 1, 'content': 'World', 'page_number': 1,
             'section_title': 'Body', 'token_count': 5, 'embedding': [0.2] * 768},
        ]

        service.store_chunks('abc123', chunks)
        assert cursor.execute.call_count == 2

    def test_stores_nothing_when_empty(self, mock_db):
        db, cursor = mock_db
        service = DocumentService(db)

        service.store_chunks('abc123', [])
        cursor.execute.assert_not_called()


@pytest.mark.unit
class TestGetAllDocuments:
    def test_returns_empty_list_when_no_docs(self, mock_db):
        db, cursor = mock_db
        cursor.fetchall.return_value = []
        service = DocumentService(db)

        result = service.get_all_documents()
        assert result == []

    def test_excludes_deleted_by_default(self, mock_db):
        db, cursor = mock_db
        cursor.fetchall.return_value = []
        service = DocumentService(db)

        service.get_all_documents(include_deleted=False)
        sql = cursor.execute.call_args[0][0]
        assert 'deleted_at IS NULL' in sql


@pytest.mark.unit
class TestFindDuplicates:
    def test_finds_exact_hash_match(self, mock_db):
        db, cursor = mock_db
        now = datetime.now(timezone.utc)
        cursor.fetchall.return_value = [('dup1', 'existing.pdf', now)]
        service = DocumentService(db)

        results = service.find_duplicates('same_hash', None, 0)
        assert len(results) == 1
        assert results[0]['match_type'] == 'exact'

    def test_skips_semantic_for_short_text(self, mock_db):
        db, cursor = mock_db
        cursor.fetchall.return_value = []
        service = DocumentService(db)

        results = service.find_duplicates('hash', [0.1] * 768, 50)
        # Should only do hash check, not semantic (text_length < 200)
        assert cursor.execute.call_count == 1


@pytest.mark.unit
class TestUpdateStatus:
    def test_updates_status(self, mock_db):
        db, cursor = mock_db
        service = DocumentService(db)

        service.update_status('abc123', 'processing')
        cursor.execute.assert_called_once()
        args = cursor.execute.call_args[0][1]
        assert 'processing' in args
