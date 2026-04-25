"""Feature tests for the pre-trained asset layout.

Asserts that:
  1. OnnxInferenceService boots cleanly with both shipped heads from pretrained_dir.
  2. Pre-shipped assets are NOT re-downloaded when already present (mtime invariant).
  3. Moving the deliberation_score meta aside causes a clean failure-to-register,
     not a silent fallback. File is restored at the end so the suite stays hermetic.

Skips when the shared encoder ONNX is absent from disk (CI machines that have not
yet downloaded the encoder). No mocks, no fakes.

pytestmark applies pytest.mark.integration to every test in this file.
"""

import os
import shutil

import pytest

from services.onnx_inference_service import OnnxInferenceService

pytestmark = pytest.mark.integration

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BACKEND_ROOT, "data", "models")
_PRETRAINED_DIR = os.path.join(_BACKEND_ROOT, "data", "pre-trained")
_ENCODER_PATH = os.path.join(_MODELS_DIR, "gte-modernbert-base", "onnx", "model.onnx")

_DELIB_META = os.path.join(
    _PRETRAINED_DIR, "deliberation_score",
    "deliberation-score-classifier_meta.json"
)
_DELIB_NPZ = os.path.join(
    _PRETRAINED_DIR, "deliberation_score",
    "deliberation-score_head.npz"
)
_MODE_META = os.path.join(
    _PRETRAINED_DIR, "mode_detector",
    "mode-detector-classifier_meta.json"
)


def _require_encoder():
    if not os.path.exists(_ENCODER_PATH):
        pytest.skip("gte-modernbert-base encoder not on disk — skipping encoder-dependent test")


def _require_shipped_assets():
    for path in (_DELIB_META, _DELIB_NPZ, _MODE_META):
        if not os.path.exists(path):
            pytest.skip(f"Shipped pre-trained asset missing: {path}")


class TestPretrainedAssetLayoutBoot:
    """OnnxInferenceService boots cleanly with the real shipped layout."""

    def test_both_heads_register_without_exception(self):
        """Constructing OnnxInferenceService and explicitly registering both heads
        must not raise. Both tasks must load cleanly from the shipped pretrained_dir.
        """
        _require_encoder()
        _require_shipped_assets()

        from services.embedding_service import _get_session_and_tokenizer
        _get_session_and_tokenizer()

        svc = OnnxInferenceService(_MODELS_DIR, _PRETRAINED_DIR)

        delib_head = svc._register_task("deliberation_score")
        assert delib_head is not None, (
            "deliberation_score head failed to register from pre-trained dir"
        )
        assert delib_head.task_type == "regression"
        assert delib_head.num_outputs == 1

        mode_head = svc._register_task("mode_detector")
        assert mode_head is not None, (
            "mode_detector head failed to register from pre-trained dir"
        )

    def test_pretrained_assets_not_redownloaded_when_present(self):
        """ensure_models() (or any direct file access) must not touch / overwrite
        the shipped .npz when it is already present. Verified by stat-checking mtime
        before and after constructing the service and calling _register_task.
        """
        _require_encoder()
        _require_shipped_assets()

        from services.embedding_service import _get_session_and_tokenizer
        _get_session_and_tokenizer()

        mtime_before = os.path.getmtime(_DELIB_NPZ)

        svc = OnnxInferenceService(_MODELS_DIR, _PRETRAINED_DIR)
        svc._register_task("deliberation_score")

        mtime_after = os.path.getmtime(_DELIB_NPZ)

        assert mtime_before == mtime_after, (
            "deliberation-score_head.npz mtime changed — shipped asset was overwritten "
            f"(before={mtime_before}, after={mtime_after})"
        )


class TestPretrainedAssetMissingMeta:
    """Moving the deliberation_score meta aside causes a clean failure, not silent fallback."""

    def test_missing_meta_causes_register_failure_not_silent_success(self):
        """If deliberation-score-classifier_meta.json is absent, _register_task must
        return None (not raise, not silently succeed with a wrong head).

        The meta file is moved to a tmp location and restored in the finally block
        so the test suite remains hermetic for subsequent tests.
        """
        _require_encoder()
        _require_shipped_assets()

        from services.embedding_service import _get_session_and_tokenizer
        _get_session_and_tokenizer()

        backup_path = _DELIB_META + ".bak_test"
        try:
            shutil.move(_DELIB_META, backup_path)

            svc = OnnxInferenceService(_MODELS_DIR, _PRETRAINED_DIR)
            result = svc._register_task("deliberation_score")
            assert result is None, (
                "_register_task should return None when meta is absent, "
                f"but got: {result!r}"
            )
            # predict_scalar must also gracefully return None, not raise.
            scalar = svc.predict_scalar("deliberation_score", "test input")
            assert scalar is None, (
                "predict_scalar should return None when the head cannot register"
            )
        finally:
            if os.path.exists(backup_path):
                shutil.move(backup_path, _DELIB_META)
