"""
Tests for MomentService — pinned messages stored as documents.

Covers: create, duplicate detection, get, list, forget,
enrichment (gist merge, summary, seal), salience boosting.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timezone


# Patch targets
_P_DOC_SVC = 'services.document_service.DocumentService'
_P_EMBED = 'services.embedding_service.get_embedding_service'
_P_ENQUEUE = 'services.document_queue.enqueue_document_processing'
_P_BG_LLM = 'services.background_llm_queue.create_background_llm_proxy'


def _make_moment_doc(doc_id='abc123', title='Test Moment', text='Some pinned text',
                     status='enriching', topic=None, gists=None):
    """Build a mock document dict that looks like a moment."""
    meta = {
        'moment_status': status,
        'moment_title': title,
        'moment_gists': gists or [],
        'moment_pinned_at': '2026-01-15T10:00:00+00:00',
        'moment_exchange_id': None,
        'moment_topic': topic,
        'moment_thread_id': None,
        'moment_user_id': 'primary',
        'boosted_episodes': [],
    }
    return {
        'id': doc_id,
        'original_name': f'Memory — {title}.md',
        'mime_type': 'text/markdown',
        'file_size_bytes': len(text),
        'file_path': f'{doc_id}/Memory — {title}.md',
        'file_hash': 'fakehash',
        'page_count': None,
        'status': 'ready',
        'error_message': None,
        'chunk_count': 1,
        'source_type': 'moment',
        'tags': [],
        'summary': '',
        'extracted_metadata': meta,
        'supersedes_id': None,
        'clean_text': text,
        'language': 'en',
        'fingerprint': None,
        'doc_category': 'Memory',
        'doc_project': None,
        'doc_date': '2026-01-15',
        'meta_locked': False,
        'watched_folder_id': None,
        'created_at': '2026-01-15T10:00:00',
        'updated_at': '2026-01-15T10:00:00',
        'deleted_at': None,
        'purge_after': None,
    }


@pytest.mark.unit
class TestMomentServiceCreate:
    """Test moment creation via document pipeline."""

    @patch(_P_ENQUEUE)
    @patch(_P_EMBED)
    @patch(_P_DOC_SVC)
    def test_create_moment_creates_document(self, MockDocSvc, mock_embed, mock_enqueue):
        from services.moment_service import MomentService

        mock_embed_svc = MagicMock()
        mock_embed_svc.generate_embedding.return_value = [0.1] * 384
        mock_embed.return_value = mock_embed_svc

        mock_doc_instance = MockDocSvc.return_value
        mock_doc_instance.create_document_from_text.return_value = 'doc123'
        mock_doc_instance.get_document.return_value = _make_moment_doc('doc123')

        # No duplicate found
        db = MagicMock()
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        cursor_mock.fetchone.return_value = None
        conn_mock.cursor.return_value = cursor_mock
        db.connection.return_value.__enter__ = MagicMock(return_value=conn_mock)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        svc = MomentService(db)

        with patch.object(svc, '_generate_title', return_value='Test Title'):
            result = svc.create_moment(
                message_text='Remember this',
                topic='cooking',
            )

        assert result['id'] == 'doc123'
        mock_doc_instance.create_document_from_text.assert_called_once()
        mock_doc_instance.update_classification.assert_called_once()
        mock_enqueue.assert_called_once_with('doc123')

    @patch(_P_EMBED)
    @patch(_P_DOC_SVC)
    def test_create_moment_detects_duplicate(self, MockDocSvc, mock_embed):
        from services.moment_service import MomentService

        mock_embed_svc = MagicMock()
        mock_embed_svc.generate_embedding.return_value = [0.1] * 384
        mock_embed.return_value = mock_embed_svc

        existing_doc = _make_moment_doc('existing123', title='Already Pinned')
        mock_doc_instance = MockDocSvc.return_value
        mock_doc_instance.get_document.return_value = existing_doc

        db = MagicMock()
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        # Return a close match (distance=0.05 < threshold=0.15)
        cursor_mock.fetchone.return_value = ('existing123', 0.05)
        conn_mock.cursor.return_value = cursor_mock
        db.connection.return_value.__enter__ = MagicMock(return_value=conn_mock)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        svc = MomentService(db)
        result = svc.create_moment(message_text='Already Pinned')

        assert result['duplicate'] is True
        assert result['existing_id'] == 'existing123'


@pytest.mark.unit
class TestMomentServiceRead:
    """Test moment read operations."""

    def test_get_moment_returns_none_for_non_moment(self):
        from services.moment_service import MomentService

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = {
                'id': 'doc1', 'source_type': 'upload',
                'deleted_at': None, 'extracted_metadata': {},
            }

            svc = MomentService(MagicMock())
            result = svc.get_moment('doc1')
            assert result is None

    def test_get_moment_returns_moment_for_moment_doc(self):
        from services.moment_service import MomentService

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = _make_moment_doc('m1')

            svc = MomentService(MagicMock())
            result = svc.get_moment('m1')
            assert result is not None
            assert result['id'] == 'm1'
            assert result['title'] == 'Test Moment'

    def test_get_moment_returns_none_for_deleted(self):
        from services.moment_service import MomentService

        doc = _make_moment_doc('m1')
        doc['deleted_at'] = '2026-01-20T00:00:00'

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = doc

            svc = MomentService(MagicMock())
            result = svc.get_moment('m1')
            assert result is None


@pytest.mark.unit
class TestMomentServiceForget:
    """Test moment forget (soft-delete + salience reversal)."""

    def test_forget_soft_deletes_document(self):
        from services.moment_service import MomentService

        doc = _make_moment_doc('m1')
        doc['extracted_metadata']['boosted_episodes'] = [
            {'episode_id': 'ep1', 'pre_boost_salience': 3.0},
        ]

        db = MagicMock()
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        conn_mock.cursor.return_value = cursor_mock
        db.connection.return_value.__enter__ = MagicMock(return_value=conn_mock)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = doc

            svc = MomentService(db)
            result = svc.forget_moment('m1')

            assert result is True
            mock_instance.soft_delete.assert_called_once_with('m1')
            # Salience reversal should have executed
            cursor_mock.execute.assert_called()

    def test_forget_non_moment_returns_false(self):
        from services.moment_service import MomentService

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = {
                'id': 'doc1', 'source_type': 'upload',
                'extracted_metadata': {},
            }

            svc = MomentService(MagicMock())
            result = svc.forget_moment('doc1')
            assert result is False


@pytest.mark.unit
class TestMomentServiceEnrichment:
    """Test gist merging and summary generation."""

    def test_enrich_merges_new_gists(self):
        from services.moment_service import MomentService

        doc = _make_moment_doc('m1', gists=['Existing gist'])

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = doc

            svc = MomentService(MagicMock())
            result = svc.enrich_moment('m1', ['New gist about cooking'])

            assert result is True
            # Should have called update_extracted_metadata with merged gists
            call_args = mock_instance.update_extracted_metadata.call_args
            updated_meta = call_args[1]['metadata'] if 'metadata' in call_args[1] else call_args[0][1]
            assert len(updated_meta['moment_gists']) == 2

    def test_enrich_deduplicates_similar_gists(self):
        from services.moment_service import MomentService

        doc = _make_moment_doc('m1', gists=['The chicken recipe was great'])

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = doc

            svc = MomentService(MagicMock())
            # Very similar gist should be deduplicated
            result = svc.enrich_moment('m1', ['The chicken recipe was really great'])

            assert result is False  # No new gists added

    @patch(_P_BG_LLM)
    def test_generate_summary_updates_document(self, mock_bg_llm):
        from services.moment_service import MomentService

        mock_llm = MagicMock()
        mock_llm.send_message.return_value = MagicMock(text='Great chicken recipe for quick meals')
        mock_bg_llm.return_value = mock_llm

        doc = _make_moment_doc('m1', gists=['gist1', 'gist2'])

        with patch(_P_DOC_SVC) as MockDocSvc:
            mock_instance = MockDocSvc.return_value
            mock_instance.get_document.return_value = doc

            svc = MomentService(MagicMock())
            result = svc.generate_summary('m1')

            assert result == 'Great chicken recipe for quick meals'
            mock_instance.update_extracted_metadata.assert_called_once()


@pytest.mark.unit
class TestMomentServiceHelpers:
    """Test helper methods."""

    def test_is_duplicate_gist_detects_similar(self):
        from services.moment_service import MomentService
        svc = MomentService(MagicMock())

        assert svc._is_duplicate_gist(
            'the quick brown fox jumps',
            ['the quick brown fox jumps over'],
            threshold=0.7,
        ) is True

    def test_is_duplicate_gist_allows_different(self):
        from services.moment_service import MomentService
        svc = MomentService(MagicMock())

        assert svc._is_duplicate_gist(
            'completely different topic here',
            ['the quick brown fox jumps over'],
            threshold=0.7,
        ) is False

    def test_doc_to_moment_extracts_fields(self):
        from services.moment_service import MomentService
        svc = MomentService(MagicMock())

        doc = _make_moment_doc('m1', title='My Moment', text='Hello world')
        result = svc._doc_to_moment(doc)

        assert result['id'] == 'm1'
        assert result['title'] == 'My Moment'
        assert result['message_text'] == 'Hello world'
        assert result['status'] == 'enriching'
        assert result['gists'] == []

    def test_doc_to_moment_handles_string_metadata(self):
        from services.moment_service import MomentService
        svc = MomentService(MagicMock())

        doc = _make_moment_doc('m1')
        doc['extracted_metadata'] = json.dumps(doc['extracted_metadata'])
        result = svc._doc_to_moment(doc)

        assert result['id'] == 'm1'
        assert result['title'] == 'Test Moment'
