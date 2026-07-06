# Tests for backend/api/privacy.py -- privacy blueprint.

import sqlite3
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from api.privacy import privacy_ns
from services.memory_store import MemoryStore
from tests.restx_test_app import mount_namespace


@pytest.mark.unit
class TestPrivacyAPI:
    @pytest.fixture
    def client(self, db: sqlite3.Connection) -> FlaskClient:
        app = mount_namespace(privacy_ns)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self) -> Iterator[None]:
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /privacy/data-summary
    # ------------------------------------------------------------------

    def test_data_summary_returns_counts(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            response = client.get('/api/privacy/data-summary')

            assert response.status_code == 200
            data = response.get_json()
            # Table counts should be present (all 0 from empty test DB)
            assert "episodes" in data
            assert "transcript" in data
            assert "scheduled_items" in data

            # Verify counts are 0 for an empty database
            assert data["episodes"] == 0

    # ------------------------------------------------------------------
    # DELETE /privacy/delete-all
    # ------------------------------------------------------------------

    def test_delete_all_without_confirm_header_returns_422(self, client: FlaskClient) -> None:
        response = client.delete('/api/privacy/delete-all')

        assert response.status_code == 422
        data = response.get_json()
        assert "error" in data
        assert "X-Confirm-Delete" in data["error"]

    def test_delete_all_with_header_clears_data(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        # Seed data
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel) "
            "VALUES (?, ?, ?, ?)",
            ("ep1", 'test gist', 5, 'test'),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            ("user", "user", "hello"),
        )
        db.commit()

        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 1

        response = client.delete(
            '/api/privacy/delete-all',
            headers={"X-Confirm-Delete": "yes"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["deleted"] is True
        assert "timestamp" in data

        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 0

    def test_delete_all_clears_extended_tables(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        # ── Seed data_graph ───────────────────────────────────────────────────
        db.execute(
            "INSERT INTO data_graph (kind, key, value) VALUES (?, ?, ?)",
            ("fact", "favourite_colour", "blue"),
        )

        # ── Seed lists + list_items (list_items FK → lists) ───────────────────
        db.execute(
            "INSERT INTO lists (id, name, list_type) VALUES (?, ?, ?)",
            ("list-1", "Groceries", "checklist"),
        )
        db.execute(
            "INSERT INTO list_items (id, list_id, content) VALUES (?, ?, ?)",
            ("item-1", "list-1", "Milk"),
        )

        # ── Seed scheduled_items (TKT-1434 prompt-only dumb-cron schema) ──────
        db.execute(
            "INSERT INTO scheduled_items (message, start_at, created_at) VALUES (?, ?, ?)",
            ("Buy groceries", "2026-05-01T10:00:00+00:00", "2026-05-01T10:00:00+00:00"),
        )

        # ── Seed documents ────────────────────────────────────────────────────
        db.execute(
            "INSERT INTO documents (id, original_name, mime_type, file_path) "
            "VALUES (?, ?, ?, ?)",
            ("doc-1", "notes.txt", "text/plain", "data/uploads/notes.txt"),
        )

        db.commit()

        # Pre-conditions: every seeded table has >= 1 row
        assert db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0] >= 1
        assert db.execute("SELECT COUNT(*) FROM lists").fetchone()[0] >= 1
        assert db.execute("SELECT COUNT(*) FROM list_items").fetchone()[0] >= 1
        assert db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0] >= 1
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] >= 1

        response = client.delete(
            '/api/privacy/delete-all',
            headers={"X-Confirm-Delete": "yes"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["deleted"] is True

        # Post-conditions: every seeded table must be empty
        assert db.execute("SELECT COUNT(*) FROM data_graph").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM lists").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM list_items").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0

    def test_delete_all_tables_all_exist_in_schema(self) -> None:
        import re

        from api.privacy import _DELETE_ALL_TABLES
        from services.file_mapper_service import FileMapperService

        schema_path = FileMapperService.get_schema_path()
        schema_sql = schema_path.read_text()

        # Match both plain and virtual CREATE TABLE declarations
        schema_tables = set(
            re.findall(
                r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)",
                schema_sql,
                re.IGNORECASE,
            )
        )

        dead = [t for t in _DELETE_ALL_TABLES if t not in schema_tables]
        assert dead == [], (
            f"_DELETE_ALL_TABLES references tables not in schema.sql: {dead}"
        )
