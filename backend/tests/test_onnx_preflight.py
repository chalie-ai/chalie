"""Feature tests for the onnxruntime boot self-heal + /ready pre-flight.

Drives the real production surfaces with zero mocks:
  - the real /ready Flask route → services.preflight_service.run_preflight,
  - the real ERROR-level logging path → /tmp/chalie.log → the real
    GET /system/observability/errors reader (Cognition → Errors).

The point the user cares about: a runtime that can't load must surface an
actionable hint *in the Cognition → Errors panel*, which only shows
ERROR/CRITICAL — so the hint must be logged at ERROR, and /ready must report
``embeddings: error`` (with the hint) rather than an eternal ``loading``.
"""

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Flask
from flask.testing import FlaskClient

import api.system as system_module
from api.system import system_bp
from services import embedding_service
from services.runtime_deps_service import RuntimeDepsService
from utils.logger import _ChalieJsonFormatter

_LIBCUDART_ERR = ImportError("libcudart.so.13: cannot open shared object file: No such file or directory")


class _BrokenWheelFinder:
    """Reproduce the production failure: the first ``import onnxruntime`` raises the
    real libcudart ImportError (the broken GPU wheel), then steps aside so the import
    after the CPU reinstall succeeds. A one-shot meta-path finder models exactly that
    boot transition — it induces the real OS-level failure, it does not mock onnxruntime."""

    def __init__(self) -> None:
        self.fired = False

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "onnxruntime" and not self.fired:
            self.fired = True
            raise _LIBCUDART_ERR
        return None


@pytest.fixture(autouse=True)
def _reset_runtime_state() -> Iterator[None]:
    """Save/restore the process-global self-heal state and embedding session so
    tests can establish a clean precondition without leaking into one another."""
    saved = (
        RuntimeDepsService._onnxruntime_status,
        RuntimeDepsService._onnxruntime_hint,
        embedding_service._session,
    )
    yield
    (
        RuntimeDepsService._onnxruntime_status,
        RuntimeDepsService._onnxruntime_hint,
        embedding_service._session,
    ) = saved


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FlaskClient, Path]]:
    log_file = tmp_path / "chalie.log"
    monkeypatch.setattr(system_module, "_LOG_FILE_PATH", str(log_file))
    monkeypatch.setattr(
        "services.auth_session_service.validate_session",
        lambda *_a, **_k: True,
    )
    app = Flask(__name__)
    app.register_blueprint(system_bp)
    app.config["TESTING"] = True
    with app.test_client() as tc:
        yield tc, log_file


@pytest.mark.unit
class TestOnnxRuntimeHint:
    """Unit D — the hint catalogue maps each native-load failure to remediation text."""

    def test_libcudart_failure_yields_cuda_hint(self) -> None:
        hint = RuntimeDepsService._onnxruntime_hint_for(_LIBCUDART_ERR)
        # The parsed CUDA major (spec §D) pinpoints the wheel↔host mismatch for the operator.
        assert "libcudart.so.13" in hint
        assert "CUDA 13" in hint
        assert "CPU" in hint  # tells the operator the automatic fallback runs on CPU

    def test_generic_cuda_failure_without_version_still_yields_cuda_hint(self) -> None:
        hint = RuntimeDepsService._onnxruntime_hint_for(ImportError("CUDA driver initialisation failed"))
        assert "CUDA toolkit" in hint

    def test_rocm_failure_yields_rocm_hint(self) -> None:
        hint = RuntimeDepsService._onnxruntime_hint_for(ImportError("libamdhip64.so: cannot open shared object file"))
        assert "ROCm" in hint

    def test_unknown_failure_yields_generic_hint(self) -> None:
        hint = RuntimeDepsService._onnxruntime_hint_for(ImportError("totally unrelated failure"))
        assert "reinstall onnxruntime" in hint


@pytest.mark.unit
class TestEnsureOnnxRuntimeHappyPath:
    """Unit A — on a host where onnxruntime imports, ensure_onnxruntime is a no-op."""

    def test_import_ok_is_noop_and_idempotent(self) -> None:
        RuntimeDepsService._onnxruntime_status = "unknown"
        RuntimeDepsService._onnxruntime_hint = None

        RuntimeDepsService.ensure_onnxruntime()
        # "ok" is reachable ONLY via the import-success branch; the swap branch
        # sets healed_to_cpu/failed. So this proves no wheel swap was attempted.
        assert RuntimeDepsService.onnxruntime_status() == "ok"
        assert RuntimeDepsService.onnxruntime_hint() is None

        # Second call returns immediately on the guard — state unchanged.
        RuntimeDepsService.ensure_onnxruntime()
        assert RuntimeDepsService.onnxruntime_status() == "ok"


@pytest.mark.unit
class TestReadyPreflightSurfacesHint:
    """Units B+C — when the self-heal failed, /ready returns 503 with the hint."""

    def test_ready_reports_embeddings_error_with_hint(self, client: tuple[FlaskClient, Path]) -> None:
        tc, _ = client
        hint = RuntimeDepsService._onnxruntime_hint_for(_LIBCUDART_ERR)
        # Establish the real post-heal-failure precondition (no session built,
        # runtime unusable) — exactly what ensure_onnxruntime() leaves behind.
        RuntimeDepsService._onnxruntime_status = "failed"
        RuntimeDepsService._onnxruntime_hint = hint
        embedding_service._session = None

        resp = tc.get("/ready")

        assert resp.status_code == 503
        body = resp.get_json()
        assert body["ready"] is False
        assert body["embeddings"] == {"status": "error", "message": hint}

    def test_ready_still_loading_when_session_absent_but_runtime_ok(self, client: tuple[FlaskClient, Path]) -> None:
        tc, _ = client
        # Runtime is fine, warmup just hasn't finished — must read 'loading', not 'error'.
        RuntimeDepsService._onnxruntime_status = "ok"
        RuntimeDepsService._onnxruntime_hint = None
        embedding_service._session = None

        resp = tc.get("/ready")

        assert resp.get_json()["embeddings"] == {"status": "loading"}


@pytest.mark.unit
class TestEnsureOnnxRuntimeHealsAndLogs:
    """The user's requirement, driven through the REAL producer — the boot self-heal
    detects a broken GPU wheel, swaps to CPU, and the diagnosis lands in Cognition →
    Errors at ERROR level. No fabricated log lines: ensure_onnxruntime() emits them."""

    def test_real_heal_path_surfaces_error_hint_in_panel(
        self, client: tuple[FlaskClient, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tc, log_file = client

        # Non-destructive guard: never let the real swap uninstall an actual GPU/ROCm wheel.
        import importlib.metadata as md
        for dist in ("onnxruntime-gpu", "onnxruntime-rocm"):
            try:
                md.version(dist)
                pytest.skip(f"{dist} is installed — the real swap would uninstall it")
            except md.PackageNotFoundError:
                pass
        # The CPU 'onnxruntime' is already present, so the reinstall is an offline no-op.
        monkeypatch.setenv("PIP_NO_INDEX", "1")
        monkeypatch.setenv("UV_OFFLINE", "1")

        # Wire the REAL production formatter + FileHandler to the file the endpoint
        # reads — exactly utils.logger.Logger.start()'s wiring.
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(_ChalieJsonFormatter())
        root = logging.getLogger()
        saved_level = root.level
        root.setLevel(logging.INFO)
        root.addHandler(handler)

        # Reproduce the broken-wheel world: the first onnxruntime import raises libcudart.
        saved_mods = {n: m for n, m in sys.modules.items() if n == "onnxruntime" or n.startswith("onnxruntime.")}
        for n in saved_mods:
            del sys.modules[n]
        finder = _BrokenWheelFinder()
        sys.meta_path.insert(0, finder)
        RuntimeDepsService._onnxruntime_status = "unknown"
        RuntimeDepsService._onnxruntime_hint = None
        try:
            RuntimeDepsService.ensure_onnxruntime()
        finally:
            sys.meta_path.remove(finder)
            handler.flush()
            root.removeHandler(handler)
            root.setLevel(saved_level)
            sys.modules.update(saved_mods)

        # The real heal ran end to end: broken import caught, CPU wheel confirmed usable.
        assert finder.fired is True
        assert RuntimeDepsService.onnxruntime_status() == "healed_to_cpu"
        assert "libcudart" in (RuntimeDepsService.onnxruntime_hint() or "")

        # Every line ensure_onnxruntime emitted is ERROR — guards the must-be-ERROR
        # decision: a downgrade to warning would silently drop the hint from the panel.
        emitted = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        runtime_lines = [r for r in emitted if r["logger"] == "services.runtime_deps_service"]
        assert runtime_lines, "ensure_onnxruntime emitted nothing"
        assert all(r["level"] == "ERROR" for r in runtime_lines)

        # …and the hint actually surfaces in the real Cognition → Errors endpoint.
        panel = [e["message"] for e in tc.get("/system/observability/errors").get_json()["errors"]]
        assert any("libcudart" in m and "CPU" in m for m in panel)
