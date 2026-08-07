"""Feature tests for count-based rotation of the application log file.

These drive the real production hot path: the real ``_CountCappedFileHandler``
with the real ``_ChalieJsonFormatter`` attached to a real ``logging.Logger``,
emitting through the real ``logging`` dispatch and asserting the bytes on disk.
The observability test drives the real /system/observability/errors endpoint
against the file the handler actually wrote. Zero mocks, zero alternative paths.
"""

import json
import logging
from pathlib import Path

import pytest

import api.system as system_module
from api.system import health_ns, system_ns
from tests.restx_test_app import mount_namespace
from utils.logger import (
    LOG_FILE_MAX_ENTRIES,
    _ChalieJsonFormatter,
    _CountCappedFileHandler,
    _LOG_TRIM_BATCH,
)

# Real production constants — these tests run the actual 20k cap / 2k batch.
_MAX = LOG_FILE_MAX_ENTRIES
_BATCH = _LOG_TRIM_BATCH
_TRIM_AT = _MAX + _BATCH


def _attach_handler(log_file: Path) -> logging.Logger:
    """Wire the real handler + formatter onto an isolated logger (no root pollution)."""
    handler = _CountCappedFileHandler(str(log_file))
    handler.setFormatter(_ChalieJsonFormatter())
    lg = logging.getLogger(f"test.rotation.{id(log_file)}")
    lg.handlers.clear()
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    lg.propagate = False
    return lg


def _messages(log_file: Path) -> list[str]:
    return [json.loads(line)["message"] for line in log_file.read_text(encoding="utf-8").splitlines()]


@pytest.mark.unit
class TestCountCappedRotation:

    def test_keeps_most_recent_after_trim(self, tmp_path: Path) -> None:
        log_file = tmp_path / "chalie.log"
        lg = _attach_handler(log_file)

        # Write past the trim threshold so a rewrite fires.
        total = _TRIM_AT + 1
        for i in range(total):
            lg.info(f"msg-{i:06d}")

        kept = _messages(log_file)

        # Exactly the cap remains, the oldest are gone, the newest are present and in order.
        assert len(kept) == _MAX
        assert kept[0] == f"msg-{total - _MAX:06d}"
        assert kept[-1] == f"msg-{total - 1:06d}"
        assert kept == [f"msg-{i:06d}" for i in range(total - _MAX, total)]
        assert "msg-000000" not in kept   # entry 1 dropped

    def test_boundary_overshoot_then_drop(self, tmp_path: Path) -> None:
        log_file = tmp_path / "chalie.log"
        lg = _attach_handler(log_file)

        # At the cap: nothing dropped.
        for i in range(_MAX):
            lg.info(f"msg-{i:06d}")
        assert len(_messages(log_file)) == _MAX
        assert _messages(log_file)[0] == "msg-000000"

        # Up to cap + batch: overshoot is allowed, still nothing dropped (amortised trim).
        for i in range(_MAX, _TRIM_AT):
            lg.info(f"msg-{i:06d}")
        assert len(_messages(log_file)) == _TRIM_AT
        assert _messages(log_file)[0] == "msg-000000"

        # One more write crosses the threshold → the trim drops the oldest block.
        lg.info(f"msg-{_TRIM_AT:06d}")
        kept = _messages(log_file)
        assert len(kept) == _MAX
        assert "msg-000000" not in kept
        assert kept[-1] == f"msg-{_TRIM_AT:06d}"

    def test_observability_reader_correct_after_trim(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_file = tmp_path / "chalie.log"
        lg = _attach_handler(log_file)

        # An early error that the trim must discard.
        lg.error("early-error")
        # Fill so the next-to-last INFO crosses the threshold and triggers a rewrite.
        for i in range(_TRIM_AT):
            lg.info(f"filler-{i:06d}")
        # Two errors that must survive and surface newest-first.
        lg.error("late-error-1")
        lg.error("late-error-2")

        monkeypatch.setattr(system_module, "_LOG_FILE_PATH", str(log_file))
        monkeypatch.setattr("services.auth_session_service.AuthSessionService.validate_session", lambda *_a, **_k: True)
        app = mount_namespace(health_ns, system_ns)
        app.config["TESTING"] = True

        with app.test_client() as tc:
            resp = tc.get("/api/system/observability/errors")

        assert resp.status_code == 200
        messages = [e["message"] for e in resp.get_json()["errors"]]
        assert messages[0] == "late-error-2"   # newest first
        assert messages[1] == "late-error-1"
        assert "early-error" not in messages    # dropped by the trim
