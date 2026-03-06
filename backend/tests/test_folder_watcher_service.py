"""
Tests for FolderWatcherService — CRUD, directory browsing, scan logic.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

from services.folder_watcher_service import (
    FolderWatcherService,
    MAX_ENQUEUE_PER_SCAN,
    MISSING_THRESHOLD,
    MIN_SCAN_INTERVAL,
    ALLOWED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    db = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    db.connection.return_value.__enter__ = MagicMock(return_value=conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return db, cursor


@pytest.fixture
def service(mock_db):
    db, _ = mock_db
    return FolderWatcherService(db)


def _make_folder_row(
    folder_id="abc12345",
    folder_path="/test/watched",
    label="TestFolder",
    source_type="filesystem",
    enabled=1,
    file_patterns='["*"]',
    ignore_patterns='[".git","node_modules"]',
    recursive=1,
    scan_interval=300,
    last_scan_at=None,
    last_scan_files=0,
    last_scan_error=None,
    source_config="{}",
    created_at="2026-03-06 12:00:00",
    updated_at="2026-03-06 12:00:00",
):
    """Build a tuple mimicking a watched_folders DB row."""
    return (
        folder_id, folder_path, label, source_type, enabled,
        file_patterns, ignore_patterns, recursive, scan_interval,
        last_scan_at, last_scan_files, last_scan_error, source_config,
        created_at, updated_at,
    )


_FOLDER_COLS = [
    'id', 'folder_path', 'label', 'source_type', 'enabled',
    'file_patterns', 'ignore_patterns', 'recursive', 'scan_interval',
    'last_scan_at', 'last_scan_files', 'last_scan_error', 'source_config',
    'created_at', 'updated_at',
]


def _make_folder_dict(**overrides):
    """Build a folder dict as returned by service methods."""
    d = {
        'id': 'abc12345',
        'folder_path': '/test/watched',
        'label': 'TestFolder',
        'source_type': 'filesystem',
        'enabled': 1,
        'file_patterns': ['*'],
        'ignore_patterns': ['.git', 'node_modules'],
        'recursive': 1,
        'scan_interval': 300,
        'last_scan_at': None,
        'last_scan_files': 0,
        'last_scan_error': None,
        'source_config': {},
        'created_at': '2026-03-06 12:00:00',
        'updated_at': '2026-03-06 12:00:00',
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateFolder:
    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/real/test/folder')
    def test_creates_folder_and_returns_dict(self, mock_real, mock_isdir, mock_access, mock_db):
        db, cursor = mock_db
        # get_folder call after insert
        cursor.fetchone.return_value = _make_folder_row(folder_path='/real/test/folder')
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.create_folder('/test/folder', label='My Docs')
        assert result is not None
        assert result['folder_path'] == '/real/test/folder'
        assert result['file_patterns'] == ['*']

    @patch('services.folder_watcher_service.os.path.isdir', return_value=False)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/nonexistent')
    def test_rejects_nonexistent_path(self, mock_real, mock_isdir, service):
        with pytest.raises(ValueError, match="not a directory"):
            service.create_folder('/nonexistent')

    @patch('services.folder_watcher_service.os.access', return_value=False)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/unreadable')
    def test_rejects_unreadable_path(self, mock_real, mock_isdir, mock_access, service):
        with pytest.raises(PermissionError, match="not readable"):
            service.create_folder('/unreadable')

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/real/test')
    def test_enforces_min_scan_interval(self, mock_real, mock_isdir, mock_access, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = _make_folder_row(scan_interval=MIN_SCAN_INTERVAL)
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        svc.create_folder('/test', scan_interval=10)
        insert_args = cursor.execute.call_args_list[0][0][1]
        # scan_interval is the 8th positional arg in the INSERT
        assert insert_args[7] == MIN_SCAN_INTERVAL


@pytest.mark.unit
class TestGetFolder:
    def test_returns_none_when_not_found(self, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = None
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.get_folder('nonexistent')
        assert result is None

    def test_returns_dict_when_found(self, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = _make_folder_row()
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.get_folder('abc12345')
        assert result['id'] == 'abc12345'
        assert result['label'] == 'TestFolder'
        assert isinstance(result['file_patterns'], list)
        assert isinstance(result['source_config'], dict)


@pytest.mark.unit
class TestGetAllFolders:
    def test_returns_empty_list(self, mock_db):
        db, cursor = mock_db
        cursor.fetchall.return_value = []
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.get_all_folders()
        assert result == []

    def test_returns_multiple_folders(self, mock_db):
        db, cursor = mock_db
        cursor.fetchall.return_value = [
            _make_folder_row(folder_id='f1'),
            _make_folder_row(folder_id='f2'),
        ]
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.get_all_folders()
        assert len(result) == 2


@pytest.mark.unit
class TestUpdateFolder:
    def test_updates_allowed_fields(self, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = _make_folder_row(label='Updated')
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.update_folder('abc12345', label='Updated')
        # Should have called UPDATE
        update_sql = cursor.execute.call_args_list[0][0][0]
        assert 'UPDATE watched_folders' in update_sql
        assert 'label' in update_sql

    def test_ignores_unknown_fields(self, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = _make_folder_row()
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        result = svc.update_folder('abc12345', unknown_field='bad')
        # No UPDATE should be executed — only the get_folder SELECT
        sql_calls = [c[0][0] for c in cursor.execute.call_args_list]
        assert all('UPDATE' not in s for s in sql_calls)

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/new/path')
    def test_validates_new_folder_path(self, mock_real, mock_isdir, mock_access, mock_db):
        db, cursor = mock_db
        cursor.fetchone.return_value = _make_folder_row(folder_path='/new/path')
        cursor.description = [(col, None) for col in _FOLDER_COLS]
        svc = FolderWatcherService(db)

        svc.update_folder('abc12345', folder_path='/new/path')
        mock_isdir.assert_called_with('/new/path')


@pytest.mark.unit
class TestDeleteFolder:
    def test_deletes_folder(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 1
        svc = FolderWatcherService(db)

        with patch(_P_MEMSTORE):
            result = svc.delete_folder('abc12345')
        assert result is True

    def test_returns_false_when_not_found(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 0
        svc = FolderWatcherService(db)

        with patch(_P_MEMSTORE):
            result = svc.delete_folder('nonexistent')
        assert result is False

    def test_soft_deletes_documents_when_requested(self, mock_db):
        db, cursor = mock_db
        cursor.rowcount = 1
        svc = FolderWatcherService(db)

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [
            {'id': 'd1', 'deleted_at': None},
            {'id': 'd2', 'deleted_at': '2026-03-01'},  # already deleted
        ]

        with patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_MEMSTORE):
            svc.delete_folder('abc12345', delete_documents=True)

        # Only d1 should be soft-deleted (d2 already was)
        mock_doc_svc.soft_delete.assert_called_once_with('d1')


# ---------------------------------------------------------------------------
# Directory browsing tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBrowseDirectory:
    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/test')
    def test_returns_directory_structure(self, mock_real, mock_isdir, mock_access, service):
        mock_entries = []
        for name in ['Documents', 'Desktop', '.hidden']:
            entry = MagicMock()
            entry.name = name
            entry.path = f'/test/{name}'
            entry.is_dir.return_value = True
            mock_entries.append(entry)

        with patch('services.folder_watcher_service.os.scandir', return_value=mock_entries), \
             patch('services.folder_watcher_service.os.listdir'):
            result = service.browse_directory('/test')

        assert result['current'] == '/test'
        # Hidden dirs should be excluded
        assert '.hidden' not in result['directories']
        assert 'Documents' in result['directories']
        assert 'Desktop' in result['directories']

    @patch('services.folder_watcher_service.os.path.isdir', return_value=False)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/not_a_dir')
    def test_rejects_non_directory(self, mock_real, mock_isdir, service):
        with pytest.raises(ValueError, match="Not a directory"):
            service.browse_directory('/not_a_dir')

    @patch('services.folder_watcher_service.os.access', return_value=False)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/unreadable')
    def test_rejects_unreadable_directory(self, mock_real, mock_isdir, mock_access, service):
        with pytest.raises(PermissionError):
            service.browse_directory('/unreadable')

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath')
    @patch('services.folder_watcher_service.os.path.expanduser')
    def test_defaults_to_home_dir(self, mock_expand, mock_real, mock_isdir, mock_access, service):
        mock_expand.return_value = '/Users/testuser'
        mock_real.return_value = '/Users/testuser'

        with patch('services.folder_watcher_service.os.scandir', return_value=[]):
            result = service.browse_directory(None)

        mock_expand.assert_called_with("~")
        assert result['current'] == '/Users/testuser'


# ---------------------------------------------------------------------------
# Scan scheduling tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScanScheduling:
    def test_scan_due_when_never_scanned(self, service):
        folder = _make_folder_dict(last_scan_at=None)
        assert service.is_scan_due(folder) is True

    def test_scan_due_when_interval_elapsed(self, service):
        from datetime import datetime, timedelta
        past = (datetime.utcnow() - timedelta(seconds=400)).isoformat()
        folder = _make_folder_dict(last_scan_at=past, scan_interval=300)
        assert service.is_scan_due(folder) is True

    def test_scan_not_due_when_recent(self, service):
        from datetime import datetime
        just_now = datetime.utcnow().isoformat()
        folder = _make_folder_dict(last_scan_at=just_now, scan_interval=300)
        assert service.is_scan_due(folder) is False

    def test_trigger_scan_sets_memorystore_key(self, service):
        with patch(_P_MEMSTORE) as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            service.trigger_scan('abc12345')
        mock_store.set.assert_called_once_with('watcher:scan_now:abc12345', '1', ex=600)

    def test_is_scan_requested_returns_true_and_clears(self, service):
        with patch(_P_MEMSTORE) as MockStore:
            mock_store = MagicMock()
            mock_store.get.return_value = '1'
            MockStore.return_value = mock_store
            result = service.is_scan_requested('abc12345')

        assert result is True
        mock_store.delete.assert_called_once_with('watcher:scan_now:abc12345')

    def test_is_scan_requested_returns_false(self, service):
        with patch(_P_MEMSTORE) as MockStore:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            MockStore.return_value = mock_store
            result = service.is_scan_requested('abc12345')

        assert result is False


# ---------------------------------------------------------------------------
# Scan logic tests
# ---------------------------------------------------------------------------

_P_MEMSTORE = 'services.memory_store.MemoryStore'
_P_DOCSVC = 'services.document_service.DocumentService'
_P_ENQUEUE = 'services.document_queue.enqueue_document_processing'


@pytest.mark.unit
class TestScanFolder:
    def test_skips_if_already_scanning(self, service):
        with patch(_P_MEMSTORE) as MockStore:
            mock_store = MagicMock()
            mock_store.get.return_value = '1'  # lock held
            MockStore.return_value = mock_store

            result = service.scan_folder(_make_folder_dict())

        assert result == {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

    @patch('services.folder_watcher_service.os.path.isdir', return_value=False)
    def test_handles_missing_folder(self, mock_isdir, mock_db):
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        with patch(_P_MEMSTORE) as MockStore:
            mock_store = MagicMock()
            mock_store.get.side_effect = lambda k: None  # no lock
            MockStore.return_value = mock_store

            result = svc.scan_folder(_make_folder_dict())

        assert result['new'] == 0
        # Should have recorded scan error
        update_calls = [c for c in cursor.execute.call_args_list
                        if 'last_scan_error' in str(c)]
        assert len(update_calls) > 0


@pytest.mark.unit
class TestScanNewFiles:
    """Test detection of new files during scan."""

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.getsize', return_value=1024)
    @patch('services.folder_watcher_service.mimetypes')
    def test_detects_new_file(self, mock_mt, mock_size, mock_isdir, mock_db):
        db, cursor = mock_db
        mock_mt.guess_type.return_value = ('application/pdf', None)
        svc = FolderWatcherService(db)

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = []
        mock_doc_svc.create_document.return_value = 'new_doc_1'

        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: None  # no lock, no cache

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/doc.pdf', 1709712000.0),
             ]), \
             patch.object(svc, '_compute_hash', return_value='hash_abc'):

            result = svc.scan_folder(folder)

        assert result['new'] == 1
        mock_doc_svc.create_document.assert_called_once()
        mock_enqueue.assert_called_once_with('new_doc_1')

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.getsize', return_value=1024)
    @patch('services.folder_watcher_service.mimetypes')
    def test_rate_limits_new_files(self, mock_mt, mock_size, mock_isdir, mock_db):
        """Caps enqueuing at MAX_ENQUEUE_PER_SCAN."""
        db, cursor = mock_db
        mock_mt.guess_type.return_value = ('text/plain', None)
        svc = FolderWatcherService(db)

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = []
        mock_doc_svc.create_document.side_effect = lambda **kw: f'doc_{kw["original_name"]}'

        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: None

        # Generate more files than the limit
        file_count = MAX_ENQUEUE_PER_SCAN + 10
        discovered = [(f'/test/watched/file{i}.txt', 1709712000.0) for i in range(file_count)]

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=discovered), \
             patch.object(svc, '_compute_hash', return_value='unique'):

            result = svc.scan_folder(folder)

        assert result['new'] == MAX_ENQUEUE_PER_SCAN
        assert result['skipped'] == 10
        assert mock_enqueue.call_count == MAX_ENQUEUE_PER_SCAN


@pytest.mark.unit
class TestScanModifiedFiles:
    """Test detection of modified files during scan."""

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.getsize', return_value=2048)
    @patch('services.folder_watcher_service.mimetypes')
    def test_detects_modified_file(self, mock_mt, mock_size, mock_isdir, mock_db):
        db, cursor = mock_db
        mock_mt.guess_type.return_value = ('application/pdf', None)
        svc = FolderWatcherService(db)

        existing_doc = {
            'id': 'old_doc', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'old_hash', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]
        mock_doc_svc.create_document.return_value = 'new_doc'

        # Cache says mtime was 100, now it's 200 → changed
        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'old_doc'}
        })
        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: cache_data if 'state' in k else None

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/doc.pdf', 200.0),
             ]), \
             patch.object(svc, '_compute_hash', return_value='new_hash'):

            result = svc.scan_folder(folder)

        assert result['updated'] == 1
        mock_doc_svc.soft_delete.assert_called_once_with('old_doc')
        mock_doc_svc.set_supersedes.assert_called_once_with('new_doc', 'old_doc')
        mock_enqueue.assert_called_once_with('new_doc')

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_skips_when_mtime_unchanged(self, mock_isdir, mock_db):
        """Files with same mtime should be fast-skipped (no hash computation)."""
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'hash1', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'doc1'}
        })
        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: cache_data if 'state' in k else None

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE), \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/doc.pdf', 100.0),  # same mtime
             ]), \
             patch.object(svc, '_compute_hash') as mock_hash:

            result = svc.scan_folder(folder)

        assert result['skipped'] == 1
        mock_hash.assert_not_called()

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_skips_when_content_unchanged(self, mock_isdir, mock_db):
        """Files with different mtime but same hash (touch only) → skip."""
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'same_hash', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'doc1'}
        })
        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: cache_data if 'state' in k else None

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/doc.pdf', 200.0),  # different mtime
             ]), \
             patch.object(svc, '_compute_hash', return_value='same_hash'):

            result = svc.scan_folder(folder)

        assert result['skipped'] == 1
        mock_enqueue.assert_not_called()


@pytest.mark.unit
class TestScanRenamedFiles:
    """Test detection of renamed files during scan."""

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_detects_renamed_file(self, mock_isdir, mock_db):
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        # Doc exists at old path with known hash
        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/old_name.pdf',
            'file_hash': 'hash_abc', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: None

        folder = _make_folder_dict()

        # File appears at new path (old path gone), same hash
        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/new_name.pdf', 100.0),
             ]), \
             patch.object(svc, '_compute_hash', return_value='hash_abc'):

            result = svc.scan_folder(folder)

        assert result['renamed'] == 1
        assert result['new'] == 0
        mock_doc_svc.update_file_path.assert_called_once_with('doc1', '/test/watched/new_name.pdf')
        mock_enqueue.assert_not_called()  # No reprocessing for renames


@pytest.mark.unit
class TestScanDeletedFiles:
    """Test detection and tolerance of deleted files."""

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_tolerates_temporary_absence(self, mock_isdir, mock_db):
        """Files missing for fewer than MISSING_THRESHOLD scans are not deleted."""
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/missing.pdf',
            'file_hash': 'hash1', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        # Cache has missing_count = 1 (first absence was last scan)
        cache_data = json.dumps({
            '/test/watched/missing.pdf': {'mtime': 100.0, 'doc_id': 'doc1', 'missing_count': 1}
        })
        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: cache_data if 'state' in k else None

        folder = _make_folder_dict()

        # File not on disk (empty walk)
        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE), \
             patch.object(svc, '_walk_folder', return_value=[]):

            result = svc.scan_folder(folder)

        assert result['deleted'] == 0
        mock_doc_svc.soft_delete.assert_not_called()

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_deletes_after_threshold_scans(self, mock_isdir, mock_db):
        """Files missing for MISSING_THRESHOLD scans get soft-deleted."""
        db, cursor = mock_db
        svc = FolderWatcherService(db)

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/gone.pdf',
            'file_hash': 'hash1', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        # missing_count is at threshold - 1 (this scan will push it over)
        cache_data = json.dumps({
            '/test/watched/gone.pdf': {
                'mtime': 100.0, 'doc_id': 'doc1',
                'missing_count': MISSING_THRESHOLD - 1,
            }
        })
        mock_store = MagicMock()
        mock_store.get.side_effect = lambda k: cache_data if 'state' in k else None

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=mock_store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE), \
             patch.object(svc, '_walk_folder', return_value=[]):

            result = svc.scan_folder(folder)

        assert result['deleted'] == 1
        mock_doc_svc.soft_delete.assert_called_once_with('doc1')


# ---------------------------------------------------------------------------
# Environment tagging tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnvironmentTags:
    def test_derives_tags_from_label_and_subfolders(self, service):
        folder = _make_folder_dict(label='Finance', folder_path='/Users/dylan/Finance')
        tags = service._derive_environment_tags(folder, '/Users/dylan/Finance/invoices/2024/receipt.pdf')
        assert tags == ['Finance', 'invoices', '2024']

    def test_root_level_file_gets_label_only(self, service):
        folder = _make_folder_dict(label='Finance', folder_path='/Users/dylan/Finance')
        tags = service._derive_environment_tags(folder, '/Users/dylan/Finance/receipt.pdf')
        assert tags == ['Finance']

    def test_no_label_returns_subfolders_only(self, service):
        folder = _make_folder_dict(label=None, folder_path='/test/watched')
        tags = service._derive_environment_tags(folder, '/test/watched/sub/deep/file.txt')
        assert tags == ['sub', 'deep']

    def test_hidden_segments_excluded(self, service):
        folder = _make_folder_dict(label='Code', folder_path='/test/repo')
        tags = service._derive_environment_tags(folder, '/test/repo/.config/hidden/file.py')
        # .config starts with '.' so it's excluded
        assert 'Code' in tags
        assert '.config' not in tags


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseJsonList:
    def test_parses_json_string(self, service):
        assert service._parse_json_list('["*.pdf","*.docx"]') == ['*.pdf', '*.docx']

    def test_returns_list_as_is(self, service):
        assert service._parse_json_list(['*.pdf']) == ['*.pdf']

    def test_returns_empty_for_invalid_json(self, service):
        assert service._parse_json_list('not json') == []

    def test_returns_empty_for_non_list_json(self, service):
        assert service._parse_json_list('{"key": "val"}') == []

    def test_returns_empty_for_none(self, service):
        assert service._parse_json_list(None) == []


@pytest.mark.unit
class TestRowToDict:
    def test_converts_row_with_json_fields(self, service):
        row = _make_folder_row()
        result = service._row_to_dict(row, _FOLDER_COLS)
        assert result['file_patterns'] == ['*']
        assert result['ignore_patterns'] == ['.git', 'node_modules']
        assert result['source_config'] == {}
        assert result['id'] == 'abc12345'
