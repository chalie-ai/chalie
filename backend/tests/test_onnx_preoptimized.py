"""Unit tests for pre-optimized ONNX model loading.

Covers the fast-path (model.optimized.onnx exists → ORT_DISABLE_ALL) and the
slow-path (only model.onnx exists → ORT_ENABLE_ALL + cache to .optimized.onnx)
in both embedding_service._get_session_and_tokenizer() and the three entry
points in onnx_inference_service:

  - OnnxInferenceService._get_base_session()
  - OnnxInferenceService._load_legacy_model()
  - OnnxInferenceService._ensure_base_model()

All tests mock onnxruntime — no real ONNX files are loaded.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


# ── onnxruntime stub ──────────────────────────────────────────────────────────
# Build a minimal in-process stub so the services can import it without the
# real package being present.  Tests that need specific behaviour override
# individual attributes via patch().

def _make_ort_stub():
    ort = types.ModuleType("onnxruntime")

    # GraphOptimizationLevel enum values
    opt_level = types.SimpleNamespace(
        ORT_DISABLE_ALL=0,
        ORT_ENABLE_BASIC=1,
        ORT_ENABLE_EXTENDED=2,
        ORT_ENABLE_ALL=99,
    )
    ort.GraphOptimizationLevel = opt_level

    # SessionOptions captures what was set so assertions can inspect it
    class SessionOptions:
        def __init__(self):
            self.graph_optimization_level = None
            self.optimized_model_filepath = None
            self.intra_op_num_threads = None
            self.inter_op_num_threads = None
            self.enable_mem_pattern = None
            self.enable_cpu_mem_arena = None

    ort.SessionOptions = SessionOptions

    # InferenceSession — returns a minimal mock by default
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = []
    mock_session.get_outputs.return_value = []
    ort.InferenceSession = MagicMock(return_value=mock_session)

    return ort


# ── embedding_service helpers ─────────────────────────────────────────────────

def _reset_embedding_globals():
    """Clear module-level singletons so each test starts fresh."""
    import services.embedding_service as em
    em._session = None
    em._tokenizer = None
    em._output_names = []
    em._input_names = []


@pytest.fixture(autouse=False)
def clean_embedding(monkeypatch):
    """Ensure embedding_service globals are reset before and after each test."""
    _reset_embedding_globals()
    yield
    _reset_embedding_globals()


# ── onnx_inference_service helpers ────────────────────────────────────────────

@pytest.fixture
def models_dir(tmp_path):
    """Return a temporary directory suitable for OnnxInferenceService."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def ort_stub(monkeypatch):
    """Inject the onnxruntime stub into sys.modules for the duration of the test."""
    stub = _make_ort_stub()
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    return stub


# ═══════════════════════════════════════════════════════════════════════════════
# 1. embedding_service — _get_session_and_tokenizer()
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingServicePreOptimized:
    """_get_session_and_tokenizer() loads the right file with the right opts."""

    def _call_get_session(self, model_dir: Path, ort_stub, monkeypatch):
        """Drive _get_session_and_tokenizer() with a fake model dir."""
        import services.embedding_service as em

        # Point the service at our temp dir
        monkeypatch.setattr(em, "_model_dir", lambda: model_dir)

        # Stub out tokenizer loading — not under test here
        fake_tokenizer = MagicMock()
        fake_tokenizer_cls = MagicMock(return_value=fake_tokenizer)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = fake_tokenizer_cls
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "onnxruntime", ort_stub)

        return em._get_session_and_tokenizer()

    # ── pre-optimized path ────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_optimized_file_when_present(self, tmp_path, monkeypatch, clean_embedding):
        """When model.optimized.onnx exists it is passed to InferenceSession."""
        ort_stub = _make_ort_stub()
        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        raw_path = onnx_dir / "model.onnx"
        opt_path = onnx_dir / "model.optimized.onnx"
        raw_path.write_bytes(b"raw")
        opt_path.write_bytes(b"opt")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        ort_stub.InferenceSession.assert_called_once()
        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(opt_path)

    @pytest.mark.unit
    def test_uses_ort_disable_all_when_optimized_present(self, tmp_path, monkeypatch, clean_embedding):
        """When model.optimized.onnx exists, graph_optimization_level = ORT_DISABLE_ALL."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        original_session_cls = ort_stub.InferenceSession

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return original_session_cls.return_value

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")
        (onnx_dir / "model.optimized.onnx").write_bytes(b"opt")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert opts.graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_DISABLE_ALL

    @pytest.mark.unit
    def test_does_not_set_optimized_filepath_when_optimized_present(self, tmp_path, monkeypatch, clean_embedding):
        """When pre-optimized model exists, optimized_model_filepath must not be set."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return MagicMock()

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")
        (onnx_dir / "model.optimized.onnx").write_bytes(b"opt")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        opts = captured_opts[0]
        # Must be None — setting it would trigger a second optimization pass
        assert opts.optimized_model_filepath is None

    # ── raw-only path ─────────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_raw_file_when_no_optimized(self, tmp_path, monkeypatch, clean_embedding):
        """When only model.onnx exists, InferenceSession receives the raw path."""
        ort_stub = _make_ort_stub()
        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        raw_path = onnx_dir / "model.onnx"
        raw_path.write_bytes(b"raw")
        # No .optimized.onnx

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(raw_path)

    @pytest.mark.unit
    def test_uses_ort_enable_all_when_only_raw_exists(self, tmp_path, monkeypatch, clean_embedding):
        """When only model.onnx exists, graph_optimization_level = ORT_ENABLE_ALL."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return MagicMock()

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        opts = captured_opts[0]
        assert opts.graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_ENABLE_ALL

    @pytest.mark.unit
    def test_sets_optimized_model_filepath_when_only_raw_exists(self, tmp_path, monkeypatch, clean_embedding):
        """When only model.onnx exists, optimized_model_filepath points to .optimized.onnx."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return MagicMock()

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")
        expected_cache = str(onnx_dir / "model.optimized.onnx")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        opts = captured_opts[0]
        assert opts.optimized_model_filepath == expected_cache

    # ── mem options are always set ────────────────────────────────────────────

    @pytest.mark.unit
    def test_mem_pattern_and_arena_set_on_optimized_path(self, tmp_path, monkeypatch, clean_embedding):
        """enable_mem_pattern and enable_cpu_mem_arena are True when loading pre-optimized."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return MagicMock()

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")
        (onnx_dir / "model.optimized.onnx").write_bytes(b"opt")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        opts = captured_opts[0]
        assert opts.enable_mem_pattern is True
        assert opts.enable_cpu_mem_arena is True

    @pytest.mark.unit
    def test_mem_pattern_and_arena_set_on_raw_path(self, tmp_path, monkeypatch, clean_embedding):
        """enable_mem_pattern and enable_cpu_mem_arena are True when loading raw model."""
        ort_stub = _make_ort_stub()
        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            return MagicMock()

        ort_stub.InferenceSession = capturing_session

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")

        self._call_get_session(model_dir, ort_stub, monkeypatch)

        opts = captured_opts[0]
        assert opts.enable_mem_pattern is True
        assert opts.enable_cpu_mem_arena is True

    # ── download not triggered when model already exists ─────────────────────

    @pytest.mark.unit
    def test_no_download_when_optimized_exists(self, tmp_path, monkeypatch, clean_embedding):
        """hf_hub_download is NOT called when model.optimized.onnx already exists."""
        ort_stub = _make_ort_stub()
        monkeypatch.setitem(sys.modules, "onnxruntime", ort_stub)

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")
        (onnx_dir / "model.optimized.onnx").write_bytes(b"opt")

        import services.embedding_service as em
        monkeypatch.setattr(em, "_model_dir", lambda: model_dir)

        fake_tokenizer_cls = MagicMock(return_value=MagicMock())
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = fake_tokenizer_cls
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        fake_download = MagicMock()
        fake_hf_hub = types.ModuleType("huggingface_hub")
        fake_hf_hub.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        em._get_session_and_tokenizer()

        fake_download.assert_not_called()

    @pytest.mark.unit
    def test_no_download_when_raw_exists(self, tmp_path, monkeypatch, clean_embedding):
        """hf_hub_download is NOT called when model.onnx already exists."""
        ort_stub = _make_ort_stub()
        monkeypatch.setitem(sys.modules, "onnxruntime", ort_stub)

        model_dir = tmp_path / "gte"
        onnx_dir = model_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_bytes(b"raw")

        import services.embedding_service as em
        monkeypatch.setattr(em, "_model_dir", lambda: model_dir)

        fake_tokenizer_cls = MagicMock(return_value=MagicMock())
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = fake_tokenizer_cls
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        fake_download = MagicMock()
        fake_hf_hub = types.ModuleType("huggingface_hub")
        fake_hf_hub.hf_hub_download = fake_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        em._get_session_and_tokenizer()

        fake_download.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OnnxInferenceService._get_base_session()
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetBaseSession:
    """_get_base_session() selects the correct model and optimization settings."""

    def _make_service(self, models_dir):
        from services.onnx_inference_service import OnnxInferenceService
        svc = OnnxInferenceService(str(models_dir))
        return svc

    def _base_dir(self, models_dir):
        from services.onnx_inference_service import BASE_MODEL_NAME
        return models_dir / BASE_MODEL_NAME

    # ── pre-optimized path ────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_optimized_file_when_present(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() passes model.optimized.onnx to InferenceSession."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")
        opt_path = base / "model.optimized.onnx"
        opt_path.write_bytes(b"opt")

        fake_tokenizer = MagicMock()
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: fake_tokenizer,
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(opt_path)

    @pytest.mark.unit
    def test_uses_ort_disable_all_when_optimized_present(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() uses ORT_DISABLE_ALL when model.optimized.onnx exists."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")
        (base / "model.optimized.onnx").write_bytes(b"opt")

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            m.get_outputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session

        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        assert captured_opts[0].graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_DISABLE_ALL

    @pytest.mark.unit
    def test_does_not_set_optimized_filepath_when_optimized_present(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() must not set optimized_model_filepath when loading pre-optimized."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")
        (base / "model.optimized.onnx").write_bytes(b"opt")

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            m.get_outputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        assert captured_opts[0].optimized_model_filepath is None

    # ── raw-only path ─────────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_raw_when_no_optimized(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() passes model.onnx to InferenceSession when no optimized file."""
        base = self._base_dir(models_dir)
        base.mkdir()
        raw_path = base / "model.onnx"
        raw_path.write_bytes(b"raw")

        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(raw_path)

    @pytest.mark.unit
    def test_uses_ort_enable_all_when_only_raw(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() uses ORT_ENABLE_ALL when only model.onnx exists."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            m.get_outputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        assert captured_opts[0].graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_ENABLE_ALL

    @pytest.mark.unit
    def test_sets_optimized_filepath_when_only_raw(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() points optimized_model_filepath at .optimized.onnx for caching."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")
        expected = str(base / "model.optimized.onnx")

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            m.get_outputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        assert captured_opts[0].optimized_model_filepath == expected

    @pytest.mark.unit
    def test_returns_none_when_no_model_files(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() returns None gracefully when neither model file is present."""
        svc = self._make_service(models_dir)
        result = svc._get_base_session()
        assert result is None
        ort_stub.InferenceSession.assert_not_called()

    @pytest.mark.unit
    def test_optimized_only_without_raw_is_accepted(self, models_dir, ort_stub, monkeypatch):
        """_get_base_session() succeeds when only model.optimized.onnx exists (no raw)."""
        base = self._base_dir(models_dir)
        base.mkdir()
        opt_path = base / "model.optimized.onnx"
        opt_path.write_bytes(b"opt")
        # Deliberately no model.onnx

        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        result = svc._get_base_session()

        # Session should be loaded
        ort_stub.InferenceSession.assert_called_once()
        assert result is not None

    # ── mem options ───────────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_mem_options_always_set(self, models_dir, ort_stub, monkeypatch):
        """enable_mem_pattern and enable_cpu_mem_arena are True regardless of which path is used."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            m.get_outputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._get_base_session()

        opts = captured_opts[0]
        assert opts.enable_mem_pattern is True
        assert opts.enable_cpu_mem_arena is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OnnxInferenceService._load_legacy_model()
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadLegacyModel:
    """_load_legacy_model() applies the same pre-optimized logic as _get_base_session()."""

    def _make_service(self, models_dir):
        from services.onnx_inference_service import OnnxInferenceService
        return OnnxInferenceService(str(models_dir))

    def _write_meta(self, model_dir: Path, *, base_model="Qwen/Qwen2.5-0.5B", pruned=True):
        meta = {
            "labels": ["RESPOND", "ACT"],
            "version": "v1.0",
            "model_type": "single_label",
            "base_model": base_model,
            "pruned": pruned,
        }
        (model_dir / "classifier_meta.json").write_text(json.dumps(meta))
        return meta

    # ── pre-optimized path ────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_optimized_onnx_when_present(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() passes .optimized.onnx to InferenceSession."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        raw_path = model_dir / "model.onnx"
        opt_path = model_dir / "model.optimized.onnx"
        raw_path.write_bytes(b"raw")
        opt_path.write_bytes(b"opt")
        meta = self._write_meta(model_dir)

        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(opt_path)

    @pytest.mark.unit
    def test_uses_ort_disable_all_when_optimized_present(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() uses ORT_DISABLE_ALL when .optimized.onnx is on disk."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"raw")
        (model_dir / "model.optimized.onnx").write_bytes(b"opt")
        meta = self._write_meta(model_dir)

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        assert captured_opts[0].graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_DISABLE_ALL

    @pytest.mark.unit
    def test_does_not_set_optimized_filepath_when_optimized_present(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() must not set optimized_model_filepath when .optimized.onnx exists."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"raw")
        (model_dir / "model.optimized.onnx").write_bytes(b"opt")
        meta = self._write_meta(model_dir)

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        assert captured_opts[0].optimized_model_filepath is None

    # ── raw-only path ─────────────────────────────────────────────────────────

    @pytest.mark.unit
    def test_loads_raw_when_no_optimized(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() passes model.onnx to InferenceSession when .optimized.onnx absent."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        raw_path = model_dir / "model.onnx"
        raw_path.write_bytes(b"raw")
        meta = self._write_meta(model_dir)

        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        loaded_path = ort_stub.InferenceSession.call_args[0][0]
        assert loaded_path == str(raw_path)

    @pytest.mark.unit
    def test_uses_ort_enable_all_when_only_raw(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() uses ORT_ENABLE_ALL when only model.onnx present."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"raw")
        meta = self._write_meta(model_dir)

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        assert captured_opts[0].graph_optimization_level == ort_stub.GraphOptimizationLevel.ORT_ENABLE_ALL

    @pytest.mark.unit
    def test_sets_optimized_filepath_when_only_raw(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() sets optimized_model_filepath so ORT caches the graph."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        raw_path = model_dir / "model.onnx"
        raw_path.write_bytes(b"raw")
        meta = self._write_meta(model_dir)
        expected = str(raw_path.with_suffix(".optimized.onnx"))

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        assert captured_opts[0].optimized_model_filepath == expected

    @pytest.mark.unit
    def test_mem_options_set_on_raw_path(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() sets enable_mem_pattern and enable_cpu_mem_arena on raw path."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"raw")
        meta = self._write_meta(model_dir)

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        opts = captured_opts[0]
        assert opts.enable_mem_pattern is True
        assert opts.enable_cpu_mem_arena is True

    @pytest.mark.unit
    def test_mem_options_set_on_optimized_path(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() sets enable_mem_pattern and enable_cpu_mem_arena on optimized path."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"raw")
        (model_dir / "model.optimized.onnx").write_bytes(b"opt")
        meta = self._write_meta(model_dir)

        captured_opts = []

        def capturing_session(path, sess_options=None, providers=None):
            captured_opts.append(sess_options)
            m = MagicMock()
            m.get_inputs.return_value = []
            return m

        ort_stub.InferenceSession = capturing_session
        monkeypatch.setattr(
            "services.onnx_inference_service._get_shared_tokenizer",
            lambda *a, **kw: MagicMock(),
        )

        svc = self._make_service(models_dir)
        svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        opts = captured_opts[0]
        assert opts.enable_mem_pattern is True
        assert opts.enable_cpu_mem_arena is True

    @pytest.mark.unit
    def test_returns_none_when_no_onnx_file(self, models_dir, ort_stub, monkeypatch):
        """_load_legacy_model() returns None gracefully when the .onnx file is missing."""
        model_dir = models_dir / "mode-tiebreaker"
        model_dir.mkdir()
        meta = self._write_meta(model_dir)
        # Deliberately no .onnx file

        svc = self._make_service(models_dir)
        result = svc._load_legacy_model(
            "mode-tiebreaker", model_dir, meta,
            meta["labels"], meta["version"], meta["model_type"], {},
        )

        assert result is None
        ort_stub.InferenceSession.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OnnxInferenceService._ensure_base_model() — download skip logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsureBaseModel:
    """_ensure_base_model() skips the network entirely when any model file exists."""

    def _make_service(self, models_dir):
        from services.onnx_inference_service import OnnxInferenceService
        return OnnxInferenceService(str(models_dir))

    def _base_dir(self, models_dir):
        from services.onnx_inference_service import BASE_MODEL_NAME
        return models_dir / BASE_MODEL_NAME

    @pytest.mark.unit
    def test_skips_download_when_raw_exists(self, models_dir, monkeypatch):
        """_ensure_base_model() does not call urlopen when model.onnx is present."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")

        mock_urlopen = MagicMock()
        monkeypatch.setattr("services.onnx_inference_service.urlopen", mock_urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        mock_urlopen.assert_not_called()

    @pytest.mark.unit
    def test_skips_download_when_optimized_exists(self, models_dir, monkeypatch):
        """_ensure_base_model() does not call urlopen when model.optimized.onnx exists alone."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.optimized.onnx").write_bytes(b"opt")
        # Deliberately no model.onnx

        mock_urlopen = MagicMock()
        monkeypatch.setattr("services.onnx_inference_service.urlopen", mock_urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        mock_urlopen.assert_not_called()

    @pytest.mark.unit
    def test_skips_download_when_both_exist(self, models_dir, monkeypatch):
        """_ensure_base_model() does not call urlopen when both model files are present."""
        base = self._base_dir(models_dir)
        base.mkdir()
        (base / "model.onnx").write_bytes(b"raw")
        (base / "model.optimized.onnx").write_bytes(b"opt")

        mock_urlopen = MagicMock()
        monkeypatch.setattr("services.onnx_inference_service.urlopen", mock_urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        mock_urlopen.assert_not_called()

    @pytest.mark.unit
    def test_attempts_download_when_no_model_exists(self, models_dir, monkeypatch):
        """_ensure_base_model() contacts GitHub when neither model file exists."""
        from urllib.error import URLError

        mock_urlopen = MagicMock(side_effect=URLError("no network"))
        monkeypatch.setattr("services.onnx_inference_service.urlopen", mock_urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()  # Should not raise — network failure is swallowed

        mock_urlopen.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Asset selection priority in _ensure_base_model()
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsureBaseModelAssetSelection:
    """_ensure_base_model() picks assets in order: optimized > quantized > raw."""

    def _make_service(self, models_dir):
        from services.onnx_inference_service import OnnxInferenceService
        return OnnxInferenceService(str(models_dir))

    def _release_response(self, assets):
        """Build a minimal GitHub releases API response."""
        return json.dumps({
            "tag_name": "v1.0",
            "assets": assets,
        }).encode()

    def _asset(self, name, url):
        return {"name": name, "browser_download_url": url}

    def _mock_urlopen(self, responses, monkeypatch):
        """
        Mock urlopen to return successive responses from a list.
        Each entry in `responses` is either bytes (success) or an exception.
        """
        call_count = [0]

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            idx = call_count[0]
            call_count[0] += 1
            r = responses[idx]
            if isinstance(r, Exception):
                raise r
            return _FakeResp(r)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)
        return call_count

    @pytest.mark.unit
    def test_prefers_optimized_over_quantized_and_raw(self, models_dir, monkeypatch):
        """When release has optimized + quantized + raw, _ensure_base_model downloads the optimized."""
        assets = [
            self._asset("qwen2.5-0.5b_base.onnx", "http://example.com/raw.onnx"),
            self._asset("qwen2.5-0.5b_base_quantized.onnx", "http://example.com/quant.onnx"),
            self._asset("qwen2.5-0.5b_base_optimized.onnx", "http://example.com/opt.onnx"),
        ]
        # Response 1: GitHub API listing; Response 2: actual file bytes
        release_payload = self._release_response(assets)
        file_bytes = b"fake-onnx-bytes"

        downloaded_urls = []


        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            downloaded_urls.append(url)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(file_bytes)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        # The second download URL should be the optimized asset
        assert len(downloaded_urls) == 2
        assert downloaded_urls[1] == "http://example.com/opt.onnx"

    @pytest.mark.unit
    def test_prefers_quantized_over_raw(self, models_dir, monkeypatch):
        """When release has quantized + raw (no optimized), _ensure_base_model downloads quantized."""
        assets = [
            self._asset("qwen2.5-0.5b_base.onnx", "http://example.com/raw.onnx"),
            self._asset("qwen2.5-0.5b_base_quantized.onnx", "http://example.com/quant.onnx"),
        ]
        release_payload = self._release_response(assets)
        file_bytes = b"fake-onnx-bytes"
        downloaded_urls = []

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            downloaded_urls.append(url)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(file_bytes)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        assert len(downloaded_urls) == 2
        assert downloaded_urls[1] == "http://example.com/quant.onnx"

    @pytest.mark.unit
    def test_falls_back_to_raw_when_only_raw_available(self, models_dir, monkeypatch):
        """When only raw asset is available, _ensure_base_model downloads it."""
        assets = [
            self._asset("qwen2.5-0.5b_base.onnx", "http://example.com/raw.onnx"),
        ]
        release_payload = self._release_response(assets)
        file_bytes = b"fake-onnx-bytes"
        downloaded_urls = []

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            downloaded_urls.append(url)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(file_bytes)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        assert len(downloaded_urls) == 2
        assert downloaded_urls[1] == "http://example.com/raw.onnx"

    @pytest.mark.unit
    def test_optimized_asset_saved_as_model_optimized_onnx(self, models_dir, monkeypatch):
        """When optimized asset is downloaded it is saved as model.optimized.onnx."""
        from services.onnx_inference_service import BASE_MODEL_NAME

        assets = [
            self._asset("qwen2.5-0.5b_base_optimized.onnx", "http://example.com/opt.onnx"),
        ]
        release_payload = self._release_response(assets)
        file_bytes = b"optimized-model-bytes"

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(file_bytes)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        dest = models_dir / BASE_MODEL_NAME / "model.optimized.onnx"
        assert dest.exists()
        assert dest.read_bytes() == file_bytes

    @pytest.mark.unit
    def test_raw_asset_saved_as_model_onnx(self, models_dir, monkeypatch):
        """When raw asset is downloaded it is saved as model.onnx."""
        from services.onnx_inference_service import BASE_MODEL_NAME

        assets = [
            self._asset("qwen2.5-0.5b_base.onnx", "http://example.com/raw.onnx"),
        ]
        release_payload = self._release_response(assets)
        file_bytes = b"raw-model-bytes"

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(file_bytes)

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        dest = models_dir / BASE_MODEL_NAME / "model.onnx"
        assert dest.exists()
        assert dest.read_bytes() == file_bytes

    @pytest.mark.unit
    def test_no_matching_asset_does_not_download(self, models_dir, monkeypatch):
        """When no base-model asset is in the release, nothing is downloaded."""
        assets = [
            # Unrelated asset — does not match qwen2.5-0.5b_base prefix
            self._asset("mode-tiebreaker_head.npz", "http://example.com/head.npz"),
        ]
        release_payload = self._release_response(assets)
        downloaded_urls = []

        class _FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def _urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            downloaded_urls.append(url)
            if "api.github.com" in url:
                return _FakeResp(release_payload)
            return _FakeResp(b"")

        monkeypatch.setattr("services.onnx_inference_service.urlopen", _urlopen)

        svc = self._make_service(models_dir)
        svc._ensure_base_model()

        # Only the API call, no file download
        assert len(downloaded_urls) == 1
        assert "api.github.com" in downloaded_urls[0]

    @pytest.mark.unit
    def test_network_error_is_swallowed(self, models_dir, monkeypatch):
        """_ensure_base_model() does not raise when the GitHub API is unreachable."""
        from urllib.error import URLError

        monkeypatch.setattr(
            "services.onnx_inference_service.urlopen",
            MagicMock(side_effect=URLError("timeout")),
        )

        svc = self._make_service(models_dir)
        svc._ensure_base_model()  # Must not raise
