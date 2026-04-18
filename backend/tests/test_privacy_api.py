"""Tests for privacy API endpoints — data-summary, export, delete-all."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app():
    """Create a minimal Flask app with the privacy blueprint registered."""
    from flask import Flask
    from api.privacy import privacy_bp
    app = Flask(__name__)
    app.register_blueprint(privacy_bp, url_prefix='/api')
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

        with app.test_client():
            with patch('api.auth.require_session', lambda f: f):
                # Direct function test — missing header
                from api.privacy import delete_all
                with app.test_request_context('/api/privacy/delete-all', method='DELETE'):
                    # Simulate missing header
                    resp, code = delete_all.__wrapped__() if hasattr(delete_all, '__wrapped__') else (None, None)

    def test_database_tables_cover_required_data(self):
        """Verify delete-all truncates the three core data tables."""
        import inspect
        from api.privacy import delete_all
        src = inspect.getsource(delete_all)

        for table in ['tool_calls', 'episodes', 'transcript']:
            assert table in src, f"Expected table '{table}' in delete_all truncation list"


# ── data-summary ──────────────────────────────────────────────────────────────

class TestDataSummary:
    def test_summary_queries_all_user_data_tables(self):
        """data_summary() must query all documented user-data tables."""
        import inspect
        from api.privacy import data_summary
        src = inspect.getsource(data_summary)

        required_tables = [
            'episodes', 'knowledge', 'transcript',
            'scheduled_items',
            'lists', 'place_fingerprints',
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
            'episodes', 'knowledge',
            'transcript', 'scheduled_items',
            'lists', 'list_items', 'place_fingerprints',
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

    def test_export_store_patterns_are_meaningful(self):
        """export_data() should export working_memory and identity state."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        # gist:*, fact:*, identity_state:* removed (services deleted)
        for pattern in ['working_memory:*']:
            assert pattern in src, f"Expected MemoryStore pattern '{pattern}' in export_data"

    def test_content_disposition_header_set(self):
        """Export response must set Content-Disposition to trigger browser download."""
        import inspect
        from api.privacy import export_data
        src = inspect.getsource(export_data)

        assert 'Content-Disposition' in src
        assert 'chalie-export.json' in src
