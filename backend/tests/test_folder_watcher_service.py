"""
Tests for FolderWatcherService — CRUD, directory browsing, scan logic.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.memory_store import MemoryStore
from services.folder_watcher_service import (
    FolderWatcherService,
    MAX_ENQUEUE_PER_SCAN,
    MISSING_THRESHOLD,
    MIN_SCAN_INTERVAL,
)
from services.database_service import get_shared_db_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service(db):
    """FolderWatcherService backed by the real test DB."""
    return FolderWatcherService(get_shared_db_service())


def _seed_folder(db, folder_id="abc12345", folder_path="/test/watched",
                 label="TestFolder", source_type="filesystem", enabled=1,
                 file_patterns='["*"]',
                 ignore_patterns='[".git","node_modules"]',
                 recursive=1, scan_interval=300, source_config="{}"):
    """Insert a watched_folders row directly into the test DB."""
    db.execute("""
        INSERT INTO watched_folders
            (id, folder_path, label, source_type, enabled,
             file_patterns, ignore_patterns, recursive,
             scan_interval, source_config, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    """, (
        folder_id, folder_path, label, source_type, enabled,
        file_patterns, ignore_patterns, recursive,
        scan_interval, source_config,
    ))
    db.commit()


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


_FOLDER_COLS = [
    'id', 'folder_path', 'label', 'source_type', 'enabled',
    'file_patterns', 'ignore_patterns', 'recursive', 'scan_interval',
    'last_scan_at', 'last_scan_files', 'last_scan_error', 'source_config',
    'created_at', 'updated_at',
]


# ---------------------------------------------------------------------------
# CRUD tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateFolder:
    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/real/test/folder')
    def test_creates_folder_and_returns_dict(self, mock_real, mock_isdir, _mock_access, db):
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.create_folder('/test/folder', label='My Docs')
        assert result is not None
        assert result['folder_path'] == '/real/test/folder'
        assert result['file_patterns'] == ['*']

        # Verify the row was actually persisted
        row = db.execute(
            "SELECT * FROM watched_folders WHERE folder_path = '/real/test/folder'"
        ).fetchone()
        assert row is not None
        assert row['label'] == 'My Docs'

    @patch('services.folder_watcher_service.os.path.isdir', return_value=False)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/nonexistent')
    def test_rejects_nonexistent_path(self, mock_real, mock_isdir, service):
        with pytest.raises(ValueError, match="not a directory"):
            service.create_folder('/nonexistent')

    @patch('services.folder_watcher_service.os.access', return_value=False)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/unreadable')
    def test_rejects_unreadable_path(self, mock_real, mock_isdir, _mock_access, service):
        with pytest.raises(PermissionError, match="not readable"):
            service.create_folder('/unreadable')

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/real/test')
    def test_enforces_min_scan_interval(self, mock_real, mock_isdir, _mock_access, db):
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.create_folder('/test', scan_interval=10)
        assert result is not None
        assert result['scan_interval'] == MIN_SCAN_INTERVAL

        # Verify in DB directly
        row = db.execute(
            "SELECT scan_interval FROM watched_folders WHERE folder_path = '/real/test'"
        ).fetchone()
        assert row['scan_interval'] == MIN_SCAN_INTERVAL


@pytest.mark.unit
class TestGetFolder:
    def test_returns_none_when_not_found(self, db):
        svc = FolderWatcherService(get_shared_db_service())
        result = svc.get_folder('nonexistent')
        assert result is None

    def test_returns_dict_when_found(self, db):
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.get_folder('abc12345')
        assert result['id'] == 'abc12345'
        assert result['label'] == 'TestFolder'
        assert isinstance(result['file_patterns'], list)
        assert isinstance(result['source_config'], dict)


@pytest.mark.unit
class TestGetAllFolders:
    def test_returns_multiple_folders(self, db):
        _seed_folder(db, folder_id='f1', folder_path='/test/path1')
        _seed_folder(db, folder_id='f2', folder_path='/test/path2')
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.get_all_folders()
        assert len(result) == 2


@pytest.mark.unit
class TestUpdateFolder:
    def test_updates_allowed_fields(self, db):
        _seed_folder(db, label='Original')
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.update_folder('abc12345', label='Updated')
        assert result['label'] == 'Updated'

        # Verify in DB
        row = db.execute(
            "SELECT label FROM watched_folders WHERE id = 'abc12345'"
        ).fetchone()
        assert row['label'] == 'Updated'

    def test_ignores_unknown_fields(self, db):
        _seed_folder(db, label='Original')
        svc = FolderWatcherService(get_shared_db_service())

        result = svc.update_folder('abc12345', unknown_field='bad')
        # Label should be unchanged
        assert result['label'] == 'Original'

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath', return_value='/new/path')
    def test_validates_new_folder_path(self, mock_real, mock_isdir, _mock_access, db):
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        svc.update_folder('abc12345', folder_path='/new/path')
        mock_isdir.assert_called_with('/new/path')

        # Verify path was updated in DB
        row = db.execute(
            "SELECT folder_path FROM watched_folders WHERE id = 'abc12345'"
        ).fetchone()
        assert row['folder_path'] == '/new/path'


@pytest.mark.unit
class TestDeleteFolder:
    def test_deletes_folder(self, db):
        """Verify delete_folder removes the DB row and clears the MemoryStore scan cache."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        store = MemoryStore()
        with patch(_P_MEMSTORE, return_value=store):
            result = svc.delete_folder('abc12345')
        assert result is True

        # Verify row is gone
        row = db.execute(
            "SELECT * FROM watched_folders WHERE id = 'abc12345'"
        ).fetchone()
        assert row is None

    def test_returns_false_when_not_found(self, db):
        """Verify delete_folder returns False for a non-existent folder ID."""
        svc = FolderWatcherService(get_shared_db_service())

        store = MemoryStore()
        with patch(_P_MEMSTORE, return_value=store):
            result = svc.delete_folder('nonexistent')
        assert result is False

    def test_soft_deletes_documents_when_requested(self, db):
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [
            {'id': 'd1', 'deleted_at': None},
            {'id': 'd2', 'deleted_at': '2026-03-01'},  # already deleted
        ]

        store = MemoryStore()
        with patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_MEMSTORE, return_value=store):
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
    def test_returns_directory_structure(self, mock_real, mock_isdir, _mock_access, service):
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
    def test_rejects_unreadable_directory(self, mock_real, mock_isdir, _mock_access, service):
        with pytest.raises(PermissionError):
            service.browse_directory('/unreadable')

    @patch('services.folder_watcher_service.os.access', return_value=True)
    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.realpath')
    @patch('services.folder_watcher_service.os.path.expanduser')
    def test_defaults_to_home_dir(self, mock_expand, mock_real, mock_isdir, _mock_access, service):
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
        from datetime import timedelta
        from services.time_utils import utc_now
        past = (utc_now() - timedelta(seconds=400)).isoformat()
        folder = _make_folder_dict(last_scan_at=past, scan_interval=300)
        assert service.is_scan_due(folder) is True

    def test_scan_not_due_when_recent(self, service):
        from services.time_utils import utc_now
        just_now = utc_now().isoformat()
        folder = _make_folder_dict(last_scan_at=just_now, scan_interval=300)
        assert service.is_scan_due(folder) is False

    def test_trigger_scan_sets_memorystore_key(self, service):
        """trigger_scan writes a scan-now flag to the MemoryStore with a 600s TTL."""
        store = MemoryStore()
        with patch(_P_MEMSTORE, return_value=store):
            service.trigger_scan('abc12345')
        assert store.get('watcher:scan_now:abc12345') == '1'

    def test_is_scan_requested_returns_true_and_clears(self, service):
        """is_scan_requested returns True and removes the flag key when the flag is set."""
        store = MemoryStore()
        store.set('watcher:scan_now:abc12345', '1')
        with patch(_P_MEMSTORE, return_value=store):
            result = service.is_scan_requested('abc12345')

        assert result is True
        assert store.get('watcher:scan_now:abc12345') is None

    def test_is_scan_requested_returns_false(self, service):
        """is_scan_requested returns False when no scan-now flag is present."""
        store = MemoryStore()
        with patch(_P_MEMSTORE, return_value=store):
            result = service.is_scan_requested('abc12345')

        assert result is False


# ---------------------------------------------------------------------------
# Scan logic tests
# ---------------------------------------------------------------------------

_P_MEMSTORE = 'services.memory_store.MemoryStore'
_P_DOCSVC = 'services.document_service.DocumentService'
_P_ENQUEUE = 'services.folder_watcher_service.FolderWatcherService._process_watched_document'


@pytest.mark.unit
class TestScanFolder:
    def test_skips_if_already_scanning(self, service):
        """scan_folder returns empty result immediately when the scan lock is already held."""
        store = MemoryStore()
        store.set('watcher:scanning:abc12345', '1')  # pre-seed: lock held
        with patch(_P_MEMSTORE, return_value=store):
            result = service.scan_folder(_make_folder_dict())

        assert result == {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

    @patch('services.folder_watcher_service.os.path.isdir', return_value=False)
    def test_handles_missing_folder(self, mock_isdir, db):
        """scan_folder records an error in the DB when the watched folder no longer exists."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        store = MemoryStore()  # empty: no lock, no cache
        with patch(_P_MEMSTORE, return_value=store):
            result = svc.scan_folder(_make_folder_dict())

        assert result['new'] == 0
        # Verify scan error was recorded in the real DB
        row = db.execute(
            "SELECT last_scan_error FROM watched_folders WHERE id = 'abc12345'"
        ).fetchone()
        assert row is not None
        assert row['last_scan_error'] is not None
        assert 'no longer accessible' in row['last_scan_error']


@pytest.mark.unit
class TestScanNewFiles:
    """Test detection of new files during scan."""

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.getsize', return_value=1024)
    @patch('services.folder_watcher_service.mimetypes')
    def test_detects_new_file(self, mock_mt, _mock_size, mock_isdir, db):
        """scan_folder creates a document record and enqueues it for a brand-new file."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())
        mock_mt.guess_type.return_value = ('application/pdf', None)

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = []
        mock_doc_svc.create_document.return_value = 'new_doc_1'

        store = MemoryStore()  # empty: no lock, no cache

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE) as mock_enqueue, \
             patch.object(svc, '_walk_folder', return_value=[
                 ('/test/watched/doc.pdf', 1709712000.0),
             ]), \
             patch.object(svc, '_compute_hash', return_value='hash_abc'):

            result = svc.scan_folder(folder)

        assert result['new'] == 1
        mock_doc_svc.create_document.assert_called_once()
        mock_enqueue.assert_called_once_with('new_doc_1', '/test/watched/doc.pdf')

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    @patch('services.folder_watcher_service.os.path.getsize', return_value=1024)
    @patch('services.folder_watcher_service.mimetypes')
    def test_rate_limits_new_files(self, mock_mt, _mock_size, mock_isdir, db):
        """scan_folder caps enqueuing at MAX_ENQUEUE_PER_SCAN new files per cycle."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())
        mock_mt.guess_type.return_value = ('text/plain', None)

        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = []
        mock_doc_svc.create_document.side_effect = lambda **kw: f'doc_{kw["original_name"]}'

        store = MemoryStore()  # empty: no lock, no cache

        # Generate more files than the limit
        file_count = MAX_ENQUEUE_PER_SCAN + 10
        discovered = [(f'/test/watched/file{i}.txt', 1709712000.0) for i in range(file_count)]

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
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
    def test_detects_modified_file(self, mock_mt, _mock_size, mock_isdir, db):
        """scan_folder supersedes a document when the file content changes on disk."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())
        mock_mt.guess_type.return_value = ('application/pdf', None)

        existing_doc = {
            'id': 'old_doc', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'old_hash', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]
        mock_doc_svc.create_document.return_value = 'new_doc'

        # Cache says mtime was 100, now it's 200 -> changed
        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'old_doc'}
        })
        store = MemoryStore()
        store.set('watcher:state:abc12345', cache_data)  # pre-seed scan-state cache

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
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
        mock_enqueue.assert_called_once_with('new_doc', '/test/watched/doc.pdf')

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_skips_when_mtime_unchanged(self, mock_isdir, db):
        """Files with same mtime should be fast-skipped (no hash computation)."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'hash1', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'doc1'}
        })
        store = MemoryStore()
        store.set('watcher:state:abc12345', cache_data)  # pre-seed scan-state cache

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
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
    def test_skips_when_content_unchanged(self, mock_isdir, db):
        """Files with different mtime but same hash (touch only) -> skip."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/doc.pdf',
            'file_hash': 'same_hash', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        cache_data = json.dumps({
            '/test/watched/doc.pdf': {'mtime': 100.0, 'doc_id': 'doc1'}
        })
        store = MemoryStore()
        store.set('watcher:state:abc12345', cache_data)  # pre-seed scan-state cache

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
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
    def test_detects_renamed_file(self, mock_isdir, db):
        """scan_folder updates the DB path (no reprocessing) when a file is renamed."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

        # Doc exists at old path with known hash
        existing_doc = {
            'id': 'doc1', 'file_path': '/test/watched/old_name.pdf',
            'file_hash': 'hash_abc', 'deleted_at': None,
        }
        mock_doc_svc = MagicMock()
        mock_doc_svc.get_documents_by_watched_folder.return_value = [existing_doc]

        store = MemoryStore()  # empty: no lock, no cache

        folder = _make_folder_dict()

        # File appears at new path (old path gone), same hash
        with patch(_P_MEMSTORE, return_value=store), \
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
    def test_tolerates_temporary_absence(self, mock_isdir, db):
        """Files missing for fewer than MISSING_THRESHOLD scans are not deleted."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

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
        store = MemoryStore()
        store.set('watcher:state:abc12345', cache_data)  # pre-seed: one prior absence

        folder = _make_folder_dict()

        # File not on disk (empty walk)
        with patch(_P_MEMSTORE, return_value=store), \
             patch(_P_DOCSVC, return_value=mock_doc_svc), \
             patch(_P_ENQUEUE), \
             patch.object(svc, '_walk_folder', return_value=[]):

            result = svc.scan_folder(folder)

        assert result['deleted'] == 0
        mock_doc_svc.soft_delete.assert_not_called()

    @patch('services.folder_watcher_service.os.path.isdir', return_value=True)
    def test_deletes_after_threshold_scans(self, mock_isdir, db):
        """Files missing for MISSING_THRESHOLD scans get soft-deleted."""
        _seed_folder(db)
        svc = FolderWatcherService(get_shared_db_service())

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
        store = MemoryStore()
        store.set('watcher:state:abc12345', cache_data)  # pre-seed: at threshold minus one

        folder = _make_folder_dict()

        with patch(_P_MEMSTORE, return_value=store), \
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
