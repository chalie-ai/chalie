"""Feature tests for the ``/ready`` pre-flight probe's embeddings component
(:func:`services.preflight_service.run_preflight`).

onnxruntime is now a plain install-time base dependency (v2/v3 of the ONNX
runtime rebuild deleted the boot-time self-heal, the CUDA/ROCm hint
catalogue, and the "healed_to_cpu" retry machinery entirely — nothing
manages, swaps, or repairs the runtime at boot anymore). The embeddings
check in ``run_preflight()`` is now a bare ``import onnxruntime`` try/except:
a broken runtime surfaces as ``{"status": "error", "message": <exception>}``
instead of the old fabricated ``RuntimeDepsService`` status string.

These tests exercise what is genuinely real and observable through the live
``/ready`` route in THIS environment, where onnxruntime is a real, working
base dependency:

* the try/except must never misreport a genuinely-working runtime as broken
  ("shouldn't-fire" path for the error branch);
* the embeddings component must distinguish "still warming up" (``loading``)
  from a real failure (``error``) — a regression here previously wedged
  ``/ready`` at an eternal ``loading`` (see the pre-rebuild self-heal design).

NOT covered here, and deliberately so: forcing ``import onnxruntime`` to
genuinely fail. Doing that without mocking Python's own import machinery
(``sys.modules``/``sys.meta_path`` trickery) is not possible — and faking the
import machinery is exactly what this rebuild's test philosophy rules out.
See the TESTER REPORT in the build doc for this gap, reported rather than
papered over with a test double.
"""

from collections.abc import Iterator
import sqlite3

import pytest
from flask import Flask
from flask.testing import FlaskClient
from flask_restx import Api

from api.system import health_ns
from services import embedding_service

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_embedding_session() -> Iterator[None]:
    """The embeddings component reads the real module-level session singleton
    directly. Save/restore it so a test that forces it to ``None`` can never
    leak that precondition into a test that runs after it."""
    saved = embedding_service._session
    yield
    embedding_service._session = saved


@pytest.fixture
def client(db: sqlite3.Connection) -> Iterator[FlaskClient]:
    """A real ``/ready`` route on a real (per-test, isolated) SQLite file —
    the ``db`` fixture keeps the database component off the real chalie.db."""
    app = Flask(__name__)
    api = Api(doc=False)
    api.add_namespace(health_ns)
    api.init_app(app)
    app.config["TESTING"] = True
    with app.test_client() as tc:
        yield tc


class TestReadyEmbeddingsComponent:
    def test_ready_never_reports_embeddings_error_when_onnxruntime_is_really_installed(
        self, client: FlaskClient
    ) -> None:
        # onnxruntime is a real base dependency in this environment — the same
        # wheel every consumer (embeddings, classifier heads, rapidocr, the
        # Silero VAD wrapper) imports through. The try/except must not
        # spuriously trip against a runtime that genuinely works.
        import onnxruntime  # noqa: F401 — proves the precondition this test relies on

        resp = client.get("/ready")

        assert resp.get_json()["embeddings"]["status"] != "error"

    def test_ready_reports_loading_not_error_when_session_not_yet_built(
        self, client: FlaskClient
    ) -> None:
        # A real, reachable production state: onnxruntime imports fine but
        # the encoder session hasn't finished warming up yet (boot in
        # progress). Must read 'loading', never 'error' — that distinction is
        # the entire point of the try/except wrapping just the import, not
        # the session-presence check.
        embedding_service._session = None

        resp = client.get("/ready")

        assert resp.status_code == 503
        body = resp.get_json()
        assert body["ready"] is False
        assert body["embeddings"] == {"status": "loading"}
