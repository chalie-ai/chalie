"""
Feature tests for GET /system/observability/errors.

Real production code path end-to-end:
  - Real Flask app (system_bp registered on a thin Flask instance)
  - Real file I/O (tmp_path-based log file written per test)
  - Zero mocks of production logic — only auth session validation and the
    module-level path constant _LOG_FILE_PATH are substituted (both are config
    boundaries, not production logic per the test discipline rules)

Behaviors asserted (each covering ≥1 real scenario):
  1. Filter + order + malformed-skip: ERROR and CRITICAL lines are returned;
     INFO/WARNING/DEBUG are excluded; newest-first ordering; malformed JSON
     lines are silently skipped.
  2. Missing log file: FileNotFoundError → empty errors list, not a 500.
  3. Empty log file: no crash, returns empty list.
  4. Cap at 200: writing 250 ERROR lines returns exactly 200 (the newest 200).
"""

import json

import pytest
from flask import Flask

import api.system as system_module
from api.system import system_bp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Thin Flask test client wired to system_bp with auth bypassed.

    Monkeypatches:
      - api.system._LOG_FILE_PATH → a tmp_path file (config boundary, not
        production logic — the path is a string constant, not behaviour)
      - services.auth_session_service.validate_session → always True
        (auth session validation is an infrastructure boundary; the endpoint
        behaviour under test is independent of whether a real session exists)

    Returns:
        tuple[FlaskClient, Path]: test client + the (possibly not-yet-created)
        log file path so each test can write its own fixture lines.
    """
    log_file = tmp_path / "chalie.log"

    monkeypatch.setattr(system_module, "_LOG_FILE_PATH", str(log_file))

    # Bypass session auth — the cookie/session mechanism is not the subject
    # of these tests and requires a full app context to work correctly.
    monkeypatch.setattr(
        "services.auth_session_service.validate_session",
        lambda *_args, **_kwargs: True,
    )

    app = Flask(__name__)
    app.register_blueprint(system_bp)
    app.config["TESTING"] = True  # Disables exception propagation only; project has no Flask-WTF/CSRF middleware

    with app.test_client() as tc:
        yield tc, log_file


# ---------------------------------------------------------------------------
# Helper: build one JSON log line as the formatter emits it
# ---------------------------------------------------------------------------


def _log_line(level: str, message: str, timestamp: str = "2026-05-03T10:00:00.000000Z") -> str:
    return json.dumps({
        "timestamp": timestamp,
        "level": level,
        "logger": "services.something",
        "service": "services",
        "message": message,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestObservabilityErrorsEndpoint:

    def test_filter_order_and_malformed_skip(self, client):
        """ERROR and CRITICAL lines are returned newest-first; INFO/WARNING filtered out;
        malformed JSON lines are silently skipped and never appear in output.

        Scenario: log file contains INFO, WARNING, malformed, ERROR, CRITICAL lines written
        in chronological order (oldest first in file). The endpoint must return the two
        error-class lines in reversed order (newest first) with no crash on the bad JSON.
        """
        tc, log_file = client

        lines = [
            _log_line("INFO",     "system started",          "2026-05-03T09:00:00.000000Z"),
            _log_line("WARNING",  "low memory",              "2026-05-03T09:01:00.000000Z"),
            "NOT VALID JSON AT ALL",
            _log_line("ERROR",    "database timeout",        "2026-05-03T09:02:00.000000Z"),
            _log_line("CRITICAL", "disk full",               "2026-05-03T09:03:00.000000Z"),
            '{"broken": true',   # partial JSON
            _log_line("DEBUG",    "debug noise",             "2026-05-03T09:04:00.000000Z"),
        ]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        resp = tc.get("/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()

        errors = data["errors"]

        # Only ERROR and CRITICAL appear
        messages = [e["message"] for e in errors]
        assert "database timeout" in messages
        assert "disk full" in messages
        assert "system started" not in messages
        assert "low memory" not in messages
        assert "debug noise" not in messages

        # Malformed lines did not crash the endpoint and do not appear
        assert len(errors) == 2

        # Newest-first: CRITICAL (09:03) before ERROR (09:02)
        assert errors[0]["message"] == "disk full"
        assert errors[1]["message"] == "database timeout"

        # Each entry has timestamp and message fields from the log
        assert errors[0]["timestamp"] == "2026-05-03T09:03:00.000000Z"
        assert errors[1]["timestamp"] == "2026-05-03T09:02:00.000000Z"

    def test_missing_log_file_returns_empty_list(self, client):
        """When the log file does not exist the endpoint returns errors:[] and status 200.

        Guards against the regression where a missing /tmp/chalie.log (e.g. first boot
        before Logger.start() runs) would produce a 500.
        """
        tc, log_file = client
        # Deliberately do not create log_file — it must not exist
        assert not log_file.exists()

        resp = tc.get("/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["errors"] == []
        assert "generated_at" in data

    def test_empty_log_file_returns_empty_list(self, client):
        """An empty log file (Chalie just started, no lines yet) returns errors:[] cleanly."""
        tc, log_file = client
        log_file.write_text("", encoding="utf-8")

        resp = tc.get("/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["errors"] == []

    def test_cap_at_200_returns_newest_200(self, client):
        """Writing 250 ERROR lines returns exactly 200 (the 200 newest, not the oldest).

        Validates the _ERROR_CAP guard and that 'newest-first' + cap = the tail of the
        file, not the head. Lines are written oldest-first (ascending timestamps); the
        endpoint must return entries whose messages are 'error-050' through 'error-249'
        (the last 200 written) in descending order.
        """
        tc, log_file = client

        lines = []
        for i in range(250):
            ts = f"2026-05-03T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}.000000Z"
            lines.append(_log_line("ERROR", f"error-{i:03d}", ts))

        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        resp = tc.get("/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        errors = data["errors"]

        # Hard cap at 200
        assert len(errors) == 200

        # The 200 newest are error-050 through error-249 (lines index 50..249)
        # After reversal the first element is error-249, last is error-050
        assert errors[0]["message"] == "error-249"
        assert errors[-1]["message"] == "error-050"

    def test_large_file_triggers_tail_seek(self, client):
        """A log file larger than _LOG_TAIL_BYTES (256 KB) must trigger the seek branch
        without crashing — regression guard for the 'can't do nonzero end-relative seeks'
        bug where the helper used end-relative seek (whence=2) on a text-mode file.

        Pads the file with INFO lines until it exceeds the tail-window threshold, then
        appends a single ERROR line at the very end. The endpoint must seek into the tail,
        skip the partial first line, parse forward, and return only the trailing ERROR.
        """
        tc, log_file = client
        from api.system import _LOG_TAIL_BYTES

        # One filler line is ~120 bytes; pad past the threshold with margin
        pad_line = _log_line("INFO", "x" * 100, "2026-05-03T08:00:00.000000Z")
        n_pad = (_LOG_TAIL_BYTES // len(pad_line)) + 200
        lines = [pad_line] * n_pad
        lines.append(_log_line("ERROR", "tail-error", "2026-05-03T09:00:00.000000Z"))

        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert log_file.stat().st_size > _LOG_TAIL_BYTES   # confirm we cross the threshold

        resp = tc.get("/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        messages = [e["message"] for e in data["errors"]]
        assert "tail-error" in messages
