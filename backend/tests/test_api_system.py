import sqlite3
from collections.abc import Iterator
from unittest.mock import patch, MagicMock

import pytest
from flask.testing import FlaskClient

from api.system import health_ns, system_ns
from configs.enums.channels import Channel
from services import compaction_persistence
from services.memory_store import MemoryStore
from tests.restx_test_app import mount_namespace


@pytest.mark.unit
class TestSystemAPI:

    @pytest.fixture
    def client(self) -> FlaskClient:
        app = mount_namespace(health_ns, system_ns)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self) -> Iterator[None]:
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # GET /health

    def test_get_health_returns_ok_and_version(self, client: FlaskClient) -> None:
        """GET /health returns status 'ok' and the current APP_VERSION."""
        with patch('consumer.APP_VERSION', '2.5.0'):
            resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['version'] == '2.5.0'

    # GET /system/status

    def test_system_status_returns_expected_keys(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """GET /system/status returns status, memory, storage top-level keys."""
        store = MemoryStore()
        # Seed 2 keys for each memory namespace so counts == 2.
        for ns in ('working_memory', 'gist_index', 'fact_index'):
            store.set(f'{ns}:a', 'x')
            store.set(f'{ns}:b', 'x')
        # Seed DMN delivery ZSET with a recent entry.
        store.zadd('dmn:deliveries', {'test-delivery': 1711500000.0})

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            resp = client.get('/api/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'memory' in data
        assert 'storage' in data
        # Memory keys should reflect store.keys() calls (3 calls: working_memory, gist, fact)
        assert data['memory']['working_memory_keys'] == 2
        assert data['memory']['gist_keys'] == 2
        assert data['memory']['fact_keys'] == 2

    def test_system_status_degraded_when_store_fails(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """GET /system/status reports 'degraded' when MemoryStore ping raises."""
        # Category C (error-path): keep MagicMock to simulate a broken store.
        broken_store = MagicMock()
        broken_store.ping.side_effect = ConnectionError('store unreachable')
        broken_store.llen.return_value = 0
        broken_store.get.return_value = None

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=broken_store):
            resp = client.get('/api/system/status')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'degraded'
        assert 'memory_store_error' in data

    # GET /system/observability/records

    def test_records_episodes_source_returns_gist_and_location(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """Episodes source returns value=gist + location, ordered by COALESCE(last_relevant_at, created_at) DESC.

        : last_accessed_at is write-dead; the endpoint now projects and orders by
        COALESCE(last_relevant_at, created_at). A row with NULL last_relevant_at sorts by
        its created_at — there is no NULLs-last behaviour anymore.

        Seed:
          ep-a: created=Jan-01, last_relevant_at=Jan-03  → sort key Jan-03 (third)
          ep-b: created=Jan-02, last_relevant_at=Jan-04  → sort key Jan-04 (first)
          ep-c: created=Jan-05, last_relevant_at=NULL     → sort key Jan-05 via COALESCE (second)
        """
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_relevant_at, location_name) "
            "VALUES ('ep-a', 'first gist', 5, 'user', '2026-01-01T00:00:00', '2026-01-03T00:00:00', 'Valletta, Malta')"
        )
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at, last_relevant_at, location_name) "
            "VALUES ('ep-b', 'second gist', 5, 'user', '2026-01-02T00:00:00', '2026-01-04T00:00:00', NULL)"
        )
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel, created_at) "
            "VALUES ('ep-c', 'null relevant', 5, 'user', '2026-01-05T00:00:00')"
        )
        db.commit()

        resp = client.get('/api/system/observability/records?source=episodes')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['source'] == 'episodes'
        assert data['returned'] == 3
        values = [r['value'] for r in data['rows']]
        # ep-c (sort key Jan-05 via COALESCE) first; ep-b (sort key Jan-04) second; ep-a (Jan-03) third
        assert values[0] == 'null relevant'
        assert values[1] == 'second gist'
        assert values[2] == 'first gist'
        # NULL last_relevant_at row sorts by its created_at (Jan-05) — ends up FIRST, not last
        assert values.index('null relevant') == 0
        # location passthrough
        row = next(r for r in data['rows'] if r['value'] == 'first gist')
        assert row['location'] == 'Valletta, Malta'
        assert 'key' not in row
        row_null_loc = next(r for r in data['rows'] if r['value'] == 'second gist')
        assert row_null_loc['location'] == ''

    def test_records_search_filters_by_like(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """Search param filters gist (episodes) and key/value (data_graph)."""
        now_iso = '2026-01-01T00:00:00+00:00'
        db.execute(
            "INSERT INTO data_graph (kind, key, value, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'favorite_color', 'blue', ?, ?)",
            (now_iso, now_iso),
        )
        db.execute(
            "INSERT INTO data_graph (kind, key, value, first_seen_at, last_confirmed_at) "
            "VALUES ('user_specific', 'pet_name', 'Rex', ?, ?)",
            (now_iso, now_iso),
        )
        db.commit()

        resp = client.get('/api/system/observability/records?source=user&q=blue')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['returned'] == 1
        assert data['rows'][0]['key'] == 'favorite_color'

    def test_records_pagination_offset_works(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """First page returns 250 rows with has_more=true; offset=250 returns remainder."""
        now_iso = '2026-01-01T00:00:00+00:00'
        for i in range(260):
            db.execute(
                "INSERT INTO episodes (id, gist, salience, channel, created_at) "
                "VALUES (?, ?, 5, 'user', ?)",
                (f'ep-{i:04d}', f'gist {i}', now_iso),
            )
        db.commit()

        resp1 = client.get('/api/system/observability/records?source=episodes')
        assert resp1.status_code == 200
        data1 = resp1.get_json()
        assert data1['returned'] == 250
        assert data1['has_more'] is True

        resp2 = client.get('/api/system/observability/records?source=episodes&offset=250')
        assert resp2.status_code == 200
        data2 = resp2.get_json()
        assert data2['returned'] == 10
        assert data2['has_more'] is False

    def test_records_invalid_source_400(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """Unknown source returns 400 with error payload."""
        resp = client.get('/api/system/observability/records?source=bogus')
        assert resp.status_code == 400
        assert resp.get_json()['error'] == 'invalid source'

    # GET /system/observability/tools

    def test_observability_tools_returns_stats(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """GET /system/observability/tools returns per-tool usage counts from tool_calls.

        Verifies: aggregation (COUNT, MAX), ORDER BY last_used_at DESC, and
        exclusion of pseudo-tool audit rows (compaction/thinking).
        """
        # Seed a transcript row so FK constraint is satisfied
        db.execute(
            "INSERT INTO transcript (role, content, channel) VALUES ('user', 'hi', 'test')"
        )
        transcript_id = db.execute(
            "SELECT id FROM transcript ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        # weather: 2 calls
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'weather', 'sunny', '2025-01-01T00:00:00'),
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'weather', 'cloudy', '2025-01-03T00:00:00'),
        )
        # code_exec: 1 call
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'code_exec', 'ok', '2025-01-02T00:00:00'),
        )
        # Pseudo-tool rows must NOT surface in the panel
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'compaction', '{}', '2025-01-04T00:00:00'),
        )
        db.execute(
            "INSERT INTO tool_calls (transcript_id, tool_name, result, created_at) "
            "VALUES (?, ?, ?, ?)",
            (transcript_id, 'thinking', '{}', '2025-01-05T00:00:00'),
        )
        db.commit()

        resp = client.get('/api/system/observability/tools')

        assert resp.status_code == 200
        data = resp.get_json()
        assert 'generated_at' in data
        tools = data['tools']

        # Pseudo-tool rows excluded
        names = [t['tool_name'] for t in tools]
        assert 'compaction' not in names
        assert 'thinking' not in names

        # Two real tools surfaced
        assert {'weather', 'code_exec'} == set(names)

        weather = next(t for t in tools if t['tool_name'] == 'weather')
        code_exec = next(t for t in tools if t['tool_name'] == 'code_exec')

        # COUNT aggregation
        assert weather['count'] == 2
        assert code_exec['count'] == 1

        # MAX(last_used_at) — weather's latest is 2025-01-03
        assert weather['last_used_at'] == '2025-01-03T00:00:00'
        assert code_exec['last_used_at'] == '2025-01-02T00:00:00'

        # ORDER BY last_used_at DESC — weather (Jan 3) before code_exec (Jan 2)
        assert names.index('weather') < names.index('code_exec')

        # Response shape: only tool_name, count, last_used_at
        for tool in tools:
            assert 'tool_name' in tool
            assert 'count' in tool
            assert 'last_used_at' in tool


    # GET /system/observability/compaction

    @staticmethod
    def _seed_compaction(db: sqlite3.Connection, *, channel: str, summary: str,
                         created_at: str, compacted_up_to: int = 42) -> int:
        """Seed a compaction the way production does: the real
        ``write_compaction`` writer appends a row to the dedicated ``compactions``
        table on the MAIN axis (``for_turn_id`` NULL). The table default fills
        ``created_at``; we then pin the known timestamp under test so the API's
        date-formatting asserts against a fixed value. Returns ``compacted_up_to``
        — the turn_id watermark the endpoint surfaces as ``compacted_up_to_id``."""
        compaction_persistence.write_compaction(channel, None, compacted_up_to, summary)
        db.execute(
            "UPDATE compactions SET created_at = ? WHERE id = "
            "(SELECT MAX(id) FROM compactions WHERE channel = ? AND for_turn_id IS NULL)",
            (created_at, channel),
        )
        db.commit()
        return compacted_up_to

    def test_observability_compaction_returns_null_when_none(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """No compaction rows → 200 with {"compaction": null} (drives the empty-state card)."""
        resp = client.get('/api/system/observability/compaction')
        assert resp.status_code == 200
        assert resp.get_json() == {'compaction': None}

    def test_observability_compaction_returns_summary_and_formatted_timestamp(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """A success compaction on the 'user' channel surfaces its summary, watermark, and a
        backend-formatted timestamp (locale_service, for_ui — UTC fallback with no telemetry)."""
        watermark = self._seed_compaction(
            db, channel=Channel.USER.value,
            summary='Earlier turns condensed here.',
            created_at='2026-01-01 00:00:01',
        )
        resp = client.get('/api/system/observability/compaction')
        assert resp.status_code == 200
        comp = resp.get_json()['compaction']
        assert comp is not None
        assert comp['summary'] == 'Earlier turns condensed here.'
        # compacted_up_to_id surfaces the stored turn_id watermark.
        assert comp['compacted_up_to_id'] == watermark
        # Timestamp is pre-formatted server-side; tests have no telemetry → UTC.
        assert comp['compacted_at'] == '2026-01-01 00:00'

    def test_observability_compaction_formats_production_iso_timestamp(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """transcript.created_at is a free-form TEXT column that may hold an ISO-8601
        string with a 'T' separator and a '+00:00' offset. The tab must render it as a
        clean human-readable string with NO 'T', NO offset, and NO 'Z' (the exact
        regression the UI scenario guards)."""
        self._seed_compaction(
            db, channel=Channel.USER.value,
            summary='Condensed.',
            created_at='2026-05-30T22:45:01.123456+00:00',
        )
        resp = client.get('/api/system/observability/compaction')
        assert resp.status_code == 200
        compacted_at = resp.get_json()['compaction']['compacted_at']
        assert compacted_at == '2026-05-30 22:45'
        assert 'T' not in compacted_at
        assert '+' not in compacted_at and not compacted_at.endswith('Z')

    def test_observability_compaction_channel_isolation(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """A compaction on a non-user channel must never appear in the chat tab."""
        self._seed_compaction(
            db, channel='subagent',
            summary='subagent only',
            created_at='2026-01-01 00:00:01',
        )
        resp = client.get('/api/system/observability/compaction')
        assert resp.status_code == 200
        assert resp.get_json()['compaction'] is None

    def test_observability_compaction_latest_wins(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """When several compactions exist on a scope, the newest (highest-id) row
        in the compactions table wins (get_compaction orders by id DESC). The
        compactor only writes a row when summary extraction succeeds, so there is
        no failure-row state to filter — the latest row is always canonical."""
        self._seed_compaction(
            db, channel=Channel.USER.value,
            summary='old summary', created_at='2026-01-01 00:00:01', compacted_up_to=10,
        )
        new_watermark = self._seed_compaction(
            db, channel=Channel.USER.value,
            summary='new summary', created_at='2026-01-02 00:00:01', compacted_up_to=20,
        )
        resp = client.get('/api/system/observability/compaction')
        assert resp.status_code == 200
        comp = resp.get_json()['compaction']
        assert comp['summary'] == 'new summary'
        assert comp['compacted_up_to_id'] == new_watermark


    # GET /ready

    def _ready_patches(self, db_ok: bool = True, store_ok: bool = True) -> dict[str, object]:
        """Build patch context for /ready — database and store can be individually broken.

        Args:
            db_ok (bool): When False, patches Database.conn() to raise so the /ready DB check fails.
            store_ok (bool): When True, uses a real MemoryStore (Category A). When False,
                uses a broken_store MagicMock whose ping() raises (Category C).

        Returns:
            dict: Mapping of patch target strings to mock/real values for use with
                ``unittest.mock.patch``.
        """
        if store_ok:
            # Category A: real MemoryStore — ping() succeeds, no pre-population needed.
            store = MemoryStore()
        else:
            # Category C: error-path only — keep a MagicMock so ping() can raise.
            broken_store = MagicMock()
            broken_store.ping.side_effect = Exception('store down')
            store = broken_store

        # Load the real ONNX embedding model so the /ready endpoint sees _session
        # as non-None and reports 'ok'. No mock — we verify the real model works.
        from services.embedding_service import _get_session_and_tokenizer
        _get_session_and_tokenizer()

        # Patch ONNX service to report ready
        mock_onnx_svc = MagicMock()
        mock_onnx_svc.ready = True

        patches: dict[str, object] = {
            'services.memory_client.MemoryClientService.create_connection': MagicMock(return_value=store),
            'services.onnx_inference_service.get_onnx_inference_service': MagicMock(return_value=mock_onnx_svc),
        }

        if db_ok:
            # When db_ok, the real db fixture is active — it points the Database
            # gateway at this test's SQLite file — so the preflight DB check
            # (Database.conn().execute('SELECT 1')) passes with no patching.
            pass
        else:
            # Force the database check to fail: run_preflight opens the DB via
            # Database.conn(), so make that raise. Patched on the class attribute
            # the preflight module references.
            patches['services.preflight_service.Database.conn'] = MagicMock(side_effect=Exception('db down'))

        return patches

    def test_ready_all_ok_returns_200_with_component_status(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """/ready with all components healthy returns 200 and structured component objects."""
        patches = self._ready_patches()
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['ready'] is True
        assert data['database'] == {'status': 'ok', 'connected': True}
        assert data['memory_store'] == {'status': 'ok'}

    def test_ready_db_failure_returns_503(self, client: FlaskClient, db: sqlite3.Connection) -> None:
        """/ready with database down returns 503 and error status in database component."""
        patches = self._ready_patches(db_ok=False)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for target, mock_val in patches.items():
                stack.enter_context(patch(target, mock_val))
            resp = client.get('/ready')

        assert resp.status_code == 503
        data = resp.get_json()
        assert data['ready'] is False
        assert data['database']['status'] == 'error'
        assert data['database']['connected'] is False
        assert 'message' in data['database']
        assert data['memory_store']['status'] == 'ok'


