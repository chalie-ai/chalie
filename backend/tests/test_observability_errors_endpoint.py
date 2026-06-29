
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask.testing import FlaskClient

import api.system as system_module
from api.system import health_ns, system_ns
from tests.restx_test_app import mount_namespace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FlaskClient, Path]]:

    log_file = tmp_path / "chalie.log"

    monkeypatch.setattr(system_module, "_LOG_FILE_PATH", str(log_file))

    # Bypass session auth — the cookie/session mechanism is not the subject
    # of these tests and requires a full app context to work correctly.
    monkeypatch.setattr(
        "services.auth_session_service.validate_session",
        lambda *_args, **_kwargs: True,
    )

    app = mount_namespace(health_ns, system_ns)
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

    def test_filter_order_and_malformed_skip(self, client: tuple[FlaskClient, Path]) -> None:
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

        resp = tc.get("/api/system/observability/errors")

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

    def test_missing_log_file_returns_empty_list(self, client: tuple[FlaskClient, Path]) -> None:
        tc, log_file = client
        # Deliberately do not create log_file — it must not exist
        assert not log_file.exists()

        resp = tc.get("/api/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["errors"] == []
        assert "generated_at" in data

    def test_empty_log_file_returns_empty_list(self, client: tuple[FlaskClient, Path]) -> None:
        tc, log_file = client
        log_file.write_text("", encoding="utf-8")

        resp = tc.get("/api/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["errors"] == []

    def test_cap_at_200_returns_newest_200(self, client: tuple[FlaskClient, Path]) -> None:
        tc, log_file = client

        lines = []
        for i in range(250):
            ts = f"2026-05-03T{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}.000000Z"
            lines.append(_log_line("ERROR", f"error-{i:03d}", ts))

        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        resp = tc.get("/api/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        errors = data["errors"]

        # Hard cap at 200
        assert len(errors) == 200

        # The 200 newest are error-050 through error-249 (lines index 50..249)
        # After reversal the first element is error-249, last is error-050
        assert errors[0]["message"] == "error-249"
        assert errors[-1]["message"] == "error-050"

    def test_large_file_triggers_tail_seek(self, client: tuple[FlaskClient, Path]) -> None:
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

        resp = tc.get("/api/system/observability/errors")

        assert resp.status_code == 200
        data = resp.get_json()
        messages = [e["message"] for e in data["errors"]]
        assert "tail-error" in messages
