# Tests assert: both heads register, shipped assets aren't overwritten, and missing meta causes clean failure (no silent fallback). Skips when shared encoder is absent from disk.

import os
import shutil

import pytest

from services.file_mapper_service import FileMapperService
from services.onnx_inference_service import OnnxInferenceService

pytestmark = pytest.mark.integration

_MODELS_DIR = str(FileMapperService.get_models_path())
_PRETRAINED_DIR = str(FileMapperService.get_pretrained_path())
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


def _require_encoder() -> None:
    if not os.path.exists(_ENCODER_PATH):
        pytest.skip("gte-modernbert-base encoder not on disk — skipping encoder-dependent test")


def _require_shipped_assets() -> None:
    for path in (_DELIB_META, _DELIB_NPZ, _MODE_META):
        if not os.path.exists(path):
            pytest.skip(f"Shipped pre-trained asset missing: {path}")


class TestPretrainedAssetLayoutBoot:

    def test_both_heads_register_without_exception(self) -> None:
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

    def test_pretrained_assets_not_redownloaded_when_present(self) -> None:
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

    def test_missing_meta_causes_register_failure_not_silent_success(self) -> None:
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
