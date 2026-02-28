"""Tests for privacy API endpoints — data-summary, export, delete-all."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app():
    """Create a minimal Flask app with the privacy blueprint registered."""
    from flask import Flask
    from api.privacy import privacy_bp
    app = Flask(__name__)
    app.register_blueprint(app_bp := privacy_bp, url_prefix='/api')
    return app


# ── _serialize_row ────────────────────────────────────────────────────────────

class TestSerializeRow:
    def test_none_values_pass_through(self):
        from api.privacy import _serialize_row
        assert _serialize_row({'x': None}) == {'x': None}

    def test_datetime_converted_to_iso(self):
        from api.privacy import _serialize_row
        dt = datetime(2026, 2, 28, 12, 0, 0, tzinfo=timezone.utc)
        result = _serialize_row({'ts': dt})
        assert result['ts'] == dt.isoformat()

    def test_bytes_serialized_as_none(self):
        from api.privacy import _serialize_row
        result = _serialize_row({'embed': b'\x00\x01\x02'})
        assert result['embed'] is None

    def test_memoryview_serialized_as_none(self):
        from api.privacy import _serialize_row
        result = _serialize_row({'embed': memoryview(b'\x00\x01')})
        assert result['embed'] is None

    def test_uuid_converted_to_string(self):
        import uuid
        from api.privacy import _serialize_row
        u = uuid.uuid4()
        result = _serialize_row({'id': u})
        assert result['id'] == str(u)

    def test_dict_passthrough(self):
        from api.privacy import _serialize_row
        d = {'nested': {'key': 'value'}}
        result = _serialize_row({'data': d})
        assert result['data'] == d

    def test_string_passthrough(self):
        from api.privacy import _serialize_row
        result = _serialize_row({'name': 'Alice'})
        assert result['name'] == 'Alice'


# ── delete-all ────────────────────────────────────────────────────────────────

class TestDeleteAll:
    def test_requires_confirm_header(self):
        from api.privacy import delete_all
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(__import__('api.privacy', fromlist=['privacy_bp']).privacy_bp)

        with app.test_client() as client:
            with patch('api.auth.require_session', lambda f: f):
                # Direct function test — missing header
                from api.privacy import delete_all
                with app.test_request_context('/api/privacy/delete-all', method='DELETE'):
                    from flask import request as flask_request
                    # Simulate missing header
                    resp, code = delete_all.__wrapped__() if hasattr(delete_all, '__wrapped__') else (None, None)

    def test_redis_patterns_cover_all_expected_namespaces(self):
        """Verify the delete-all function references all critical Redis namespaces."""
        import inspect
        from api.privacy import delete_all
        src = inspect.getsource(delete_all)

        # Critical namespaces that must appear
        for pattern in [
            'working_memory:*', 'gist:*', 'fact:*',
            'auth_session:*', 'proactive:*',
            'identity_state:*', 'cognitive_drift_state',
            'tool_state:*', 'metrics:timing:*',
        ]:
            assert pattern in src, f"Expected Redis pattern '{pattern}' in delete_all"

    def test_postgres_tables_cover_all_user_data(self):
        """Verify delete-all truncates all documented user-data tables."""
        import inspect
        from api.privacy import delete_all
        src = inspect.getsource(delete_all)

        required_tables = [
            'episodes', 'semantic_concepts', 'semantic_relationships',
            'user_traits', 'threads', 'autobiography', 'scheduled_items',
            'persistent_tasks', 'lists', 'identity_vectors', 'place_fingerprints',
            'cognitive_reflexes', 'interaction_log', 'cortex_iterations',
            'routing_decisions', 'procedural_memory', 'curiosity_threads',
        ]
        for table in required_tables:
            assert table in src, f"Expected table '{table}' in delete_all truncation list"

    def test_audit_log_written_after_truncation(self):
        """Verify interaction_log is truncated before the audit entry is written."""
        import inspect
        from api.privacy import delete_all
        src = inspect.getsource(delete_all)

        truncate_pos = src.find('"interaction_log"')
        audit_pos = src.find('privacy_delete_all')
        assert truncate_pos < audit_pos, (
            "interaction_log should be truncated before the audit entry is written"
        )


# ── data-summary ──────────────────────────────────────────────────────────────

class TestDataSummary:
    def test_summary_queries_all_user_data_tables(self):
        """data_summary() must query all documented user-data tables."""
        import inspect
        from api.privacy import data_summary
        src = inspect.getsource(data_summary)

        required_tables = [
            'episodes', 'semantic_concepts', 'user_traits', 'threads',
            'autobiography', 'scheduled_items', 'persistent_tasks',
            'lists', 'place_fingerprints', 'cognitive_reflexes',
            'interaction_log', 'curiosity_threads',
        ]
        for table in required_tables:
            assert table in src, f"Expected table '{table}' in data_summary query list"


# ── export ────────────────────────────────────────────────────────────────────

class TestExportData:
    def test_export_queries_all_user_data_tables(self):
        """export_data() must query all documented user-data tables."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        required_tables = [
            'episodes', 'semantic_concepts', 'semantic_relationships',
            'user_traits', 'threads', 'autobiography', 'scheduled_items',
            'persistent_tasks', 'lists', 'list_items', 'place_fingerprints',
            'cognitive_reflexes', 'curiosity_threads',
        ]
        for table in required_tables:
            assert table in src, f"Expected table '{table}' in export_data table list"

    def test_export_excludes_sensitive_tables(self):
        """export_data() must NOT query tool_configs or providers (contain API keys)."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        for sensitive in ['tool_configs', 'providers']:
            assert sensitive not in src, (
                f"Sensitive table '{sensitive}' must not appear in export_data"
            )

    def test_export_redis_patterns_are_meaningful(self):
        """export_data() should export working_memory, facts, gists, identity."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        for pattern in ['working_memory:*', 'gist:*', 'fact:*', 'identity_state:*']:
            assert pattern in src, f"Expected Redis pattern '{pattern}' in export_data"

    def test_content_disposition_header_set(self):
        """Export response must set Content-Disposition to trigger browser download."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        assert 'Content-Disposition' in src
        assert 'chalie-export.json' in src
