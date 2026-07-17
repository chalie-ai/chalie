# Tests for DocumentService migrated from mock_db to real SQLite via shared `db` fixture.

import sqlite3
from collections.abc import Callable
from typing import cast

import pytest

from models.document import DocumentRow
from services.document_service import DocumentService
from services.memory_recall_service import recall
from services.write_queue_service import WriteQueueService


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
    cols: dict[str, object] = {
        "id": doc_id,
        "original_name": original_name,
        "mime_type": mime_type,
        "file_size_bytes": file_size,
        "file_path": file_path,
        "file_hash": file_hash,
        "source_type": source_type,
        "status": status,
    }
    cols.update(extra)
    col_names = ', '.join(cols.keys())
    placeholders = ', '.join(['?'] * len(cols))
    db.execute(
        f"INSERT INTO documents ({col_names}) VALUES ({placeholders})",
        list(cols.values()),
    )
    db.commit()
    return doc_id


def _drain_search_index() -> None:
    """Drive the REAL async search-expander pipeline synchronously against the
    bound test DB — the exact production code path, no mocks. In prod the
    search_expander_worker daemon does this continuously; a test must do it
    explicitly because no worker runs under pytest."""
    from services.search_expander_service import SearchExpanderService
    svc = SearchExpanderService()
    svc._self_heal()
    item = svc._dequeue()
    while item is not None:
        svc._process(item)
        item = svc._dequeue()


@pytest.fixture
def doc_service(db: sqlite3.Connection) -> DocumentService:
    """DocumentService wired to the test database with inline write queue."""
    svc = DocumentService()
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
class TestFragmentIntegrity:
    """The ``data_graph`` fragment side of documents (``models/document.py``),
    not just the ``documents`` metadata row TestSoftDelete/TestRestore above
    cover. Drives the real DocumentService soft-delete/restore/hard-delete
    paths and the real DocumentRow fragment primitives against real SQLite —
    the pre-rewrite cascade left orphaned fragments + FTS/vec shadow rows on
    hard-delete and re-ingest; these prove that bug stays fixed."""

    def test_soft_delete_excludes_fragments_from_reads_and_recall_restore_reinstates_them(
        self, db: sqlite3.Connection, doc_service: DocumentService,
    ) -> None:
        _insert_document(db, doc_id='softdoc', original_name='softdoc.txt', status='ready')
        DocumentRow.create_fragment('softdoc', 0, 'The zyxqplote fox jumped over the lazy dog.')
        _drain_search_index()

        # Live before soft-delete: direct read and cross-kind recall both surface it.
        assert len(DocumentRow.for_document_id('softdoc').get()) == 1
        hits = recall('zyxqplote', kinds=['document'], limit=10)
        assert any(h['source'] == 'document:softdoc' for h in hits), hits

        assert doc_service.soft_delete('softdoc') is True

        # RECALL-STRICT: an active=0 fragment must not surface anywhere.
        assert DocumentRow.for_document_id('softdoc').get() == []
        hits_after_delete = recall('zyxqplote', kinds=['document'], limit=10)
        assert not any(h['source'] == 'document:softdoc' for h in hits_after_delete), hits_after_delete

        assert doc_service.restore('softdoc') is True
        assert len(DocumentRow.for_document_id('softdoc').get()) == 1

    def test_purge_by_document_id_removes_fragments_and_shadow_rows_no_orphans(
        self, db: sqlite3.Connection,
    ) -> None:
        f1 = DocumentRow.create_fragment('purgedoc', 0, 'Alpha fragment about zqxvbnwerty giraffes.')
        f2 = DocumentRow.create_fragment('purgedoc', 1, 'Beta fragment about zqxvbnwerty giraffes too.')
        ids = [f1.id, f2.id]
        _drain_search_index()

        # FTS posting is really there before purge (a real MATCH hit, not just a
        # row count) — the thing that must leave no orphan behind.
        hits_before = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH ?", ('zqxvbnwerty',)
        ).fetchall()
        assert len(hits_before) == 2, hits_before

        # The key/value-vec shadow rows are written by the real production
        # backfill (_backfill_key_value_vec) during the drain above — asserted
        # here so the purge below is proven to actually delete something.
        for rowid in ids:
            assert db.execute(
                "SELECT 1 FROM data_graph_key_vec WHERE rowid = ?", (rowid,)
            ).fetchone() is not None
            assert db.execute(
                "SELECT 1 FROM data_graph_value_vec WHERE rowid = ?", (rowid,)
            ).fetchone() is not None

        count = DocumentRow.purge_by_document_id('purgedoc')
        assert count == 2
        assert DocumentRow.for_document_id('purgedoc').get() == []

        hits_after = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH ?", ('zqxvbnwerty',)
        ).fetchall()
        assert hits_after == []
        for rowid in ids:
            assert db.execute(
                "SELECT 1 FROM data_graph_key_vec WHERE rowid = ?", (rowid,)
            ).fetchone() is None
            assert db.execute(
                "SELECT 1 FROM data_graph_value_vec WHERE rowid = ?", (rowid,)
            ).fetchone() is None

    def test_hard_delete_removes_fragments_through_the_production_cascade(
        self, db: sqlite3.Connection, doc_service: DocumentService,
    ) -> None:
        _insert_document(db, doc_id='harddoc', original_name='harddoc.txt', status='ready',
                         file_path='')
        DocumentRow.create_fragment('harddoc', 0, 'Gamma fragment about qzxdfghjkl elephants.')
        _drain_search_index()

        assert len(DocumentRow.for_document_id('harddoc').get()) == 1

        # Through DocumentService.hard_delete (services/document_service.py:520-521),
        # not by calling purge_by_document_id directly — proves the production
        # cascade wiring, not just the model primitive in isolation.
        assert doc_service.hard_delete('harddoc') is True

        assert DocumentRow.for_document_id('harddoc').get() == []
        hits_after = db.execute(
            "SELECT rowid FROM data_graph_fts WHERE data_graph_fts MATCH ?", ('qzxdfghjkl',)
        ).fetchall()
        assert hits_after == []


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
