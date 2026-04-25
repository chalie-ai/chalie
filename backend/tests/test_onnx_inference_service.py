"""Tests for services/onnx_inference_service.py — real encoder, real heads, no mocks.

These tests exercise the full classifier pipeline end-to-end:

1. Real ``gte-modernbert-base`` encoder ONNX (cached at backend/data/models/).
2. Real shipped deliberation_score head (backend/data/pre-trained/deliberation_score/).
3. Tmpdir-injected heads to simulate corruption, contract violations, and missing
   assets — written as real ``.npz`` + ``meta.json`` files on disk, loaded by the
   real ``_register_task`` codepath. **Zero mocks, zero stubs.**

Risks covered:

* **NaN propagation** — head with NaN weights returns ``None`` (not a NaN scalar).
* **Out-of-range scalar** — guard rejects scalars outside [0, 1].
* **Encoder/head drift** — sha256 pin mismatch raises ``RuntimeError`` at registration.
* **Boot-time contract** — regression heads with the wrong shape/activation raise.
* **Graceful degradation** — missing or corrupt files return ``None``, never crash.
* **Consumer ``None`` semantics** — ``DeliberationScoreService.classify`` returns
  ``None`` for empty input and any downstream failure.
"""

import json
import math
import os

import numpy as np
import pytest

from services.deliberation_score_service import DeliberationScoreService
from services.embedding_service import _get_session_and_tokenizer
from services.onnx_inference_service import (
    OnnxInferenceService,
    _compute_encoder_sha256,
    _get_encoder_sha256,
)


# Default repo paths for real assets — resolved relative to backend/.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MODELS_DIR = os.path.join(_BACKEND_ROOT, "data", "models")
_DEFAULT_PRETRAINED_DIR = os.path.join(_BACKEND_ROOT, "data", "pre-trained")


def _real_encoder_sha() -> str:
    """Return the sha256 of the actual shipped encoder ONNX (computed once)."""
    from pathlib import Path
    return _get_encoder_sha256(
        Path(_DEFAULT_MODELS_DIR) / "gte-modernbert-base" / "onnx" / "model.onnx"
    )


def _write_regression_head(
    pretrained_dir,
    *,
    task_name: str = "deliberation_score",
    prefix: str = "deliberation-score",
    sha256: str | None = None,
    num_outputs: int = 1,
    extra_feature_dim: int = 0,
    output_activation: str = "sigmoid",
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    head_asset_name: str | None = None,
    skip_meta: bool = False,
    skip_npz: bool = False,
    corrupt_meta: bool = False,
    omit_head_asset: bool = False,
) -> None:
    """Write a regression head + meta JSON pair into ``pretrained_dir/<task>/``.

    Defaults produce a valid scalar regression head with the real encoder sha256.
    Override knobs flip individual contract/format properties for negative tests.
    """
    task_dir = os.path.join(str(pretrained_dir), task_name)
    os.makedirs(task_dir, exist_ok=True)

    if sha256 is None:
        sha256 = _real_encoder_sha()
    if w1 is None:
        w1 = np.zeros((256, 768), dtype=np.float32)
    if b1 is None:
        b1 = np.zeros(256, dtype=np.float32)
    if w2 is None:
        w2 = np.zeros((num_outputs, 256), dtype=np.float32)
    if b2 is None:
        b2 = np.zeros(num_outputs, dtype=np.float32)

    head_filename = head_asset_name or f"{prefix}_head.npz"

    if not skip_npz:
        np.savez(
            os.path.join(task_dir, head_filename),
            W1=w1, b1=b1, W2=w2, b2=b2,
        )

    if skip_meta:
        return

    meta = {
        "labels": ["score"],
        "model_name": task_name,
        "task_type": "regression",
        "base_encoder_sha256": sha256,
        "input_dim": int(w1.shape[1]),
        "hidden_dim": int(w1.shape[0]),
        "num_outputs": num_outputs,
        "num_classes": num_outputs,
        "extra_feature_dim": extra_feature_dim,
        "activation": "gelu",
        "output_activation": output_activation,
    }
    if not omit_head_asset:
        meta["head_asset"] = head_filename

    meta_path = os.path.join(task_dir, f"{prefix}-classifier_meta.json")
    if corrupt_meta:
        with open(meta_path, "w") as f:
            f.write("{this is not valid json")
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f)


# ─────────────────────────────────────────────────────────────────────────────
# Real shipped deliberation_score head + real encoder
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictScalarReal:
    """Real shipped head + real encoder — verify output contract holds end-to-end."""

    @classmethod
    def setup_class(cls):
        # Load the real encoder once for the whole class. Subsequent tests reuse it.
        _get_session_and_tokenizer()

    def _svc(self):
        return OnnxInferenceService(_DEFAULT_MODELS_DIR, _DEFAULT_PRETRAINED_DIR)

    def test_predict_scalar_returns_finite_float_in_range(self):
        """Real text → finite float strictly inside [0, 1]."""
        svc = self._svc()
        result = svc.predict_scalar(
            "deliberation_score",
            "Explain the philosophical implications of compatibilist free will.",
        )
        assert result is not None
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_predict_scalar_handles_short_text(self):
        """Short text should still produce finite scalar in range."""
        svc = self._svc()
        result = svc.predict_scalar("deliberation_score", "ok")
        assert result is not None
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_predict_scalar_handles_long_text(self):
        """Text longer than the 256 token cap is truncated and still scored."""
        svc = self._svc()
        long_text = "philosophy " * 500  # ~500 tokens, well over the 256 cap
        result = svc.predict_scalar("deliberation_score", long_text)
        assert result is not None
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_predict_scalar_handles_unicode(self):
        """Unicode/emoji input doesn't crash and produces a finite scalar."""
        svc = self._svc()
        result = svc.predict_scalar(
            "deliberation_score",
            "Should I trust the universe? 🌌 ¿Por qué es el cielo azul?",
        )
        assert result is not None
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_predict_scalar_unknown_task_returns_none(self):
        """An unregistered task name returns None, not raises."""
        svc = self._svc()
        result = svc.predict_scalar("not_a_real_task", "anything")
        assert result is None

    def test_predict_scalar_repeatable_for_same_input(self):
        """Same text → same scalar (within float tolerance) on repeated calls."""
        svc = self._svc()
        text = "What's the right way to think about uncertainty?"
        a = svc.predict_scalar("deliberation_score", text)
        b = svc.predict_scalar("deliberation_score", text)
        assert a is not None and b is not None
        assert a == pytest.approx(b, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# NaN / Inf injection — tmp dir with real .npz files containing bad weights
# ─────────────────────────────────────────────────────────────────────────────


class TestPredictScalarNaNGuard:
    """Inject heads with NaN/Inf weights and verify the guard returns None."""

    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    def test_nan_weights_in_w1_returns_none(self, tmp_path):
        w1 = np.zeros((256, 768), dtype=np.float32)
        w1[0, 0] = np.nan  # Single NaN poisons the whole forward pass.
        _write_regression_head(tmp_path, w1=w1)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN W1 must be caught by the guard"

    def test_nan_in_b2_returns_none(self, tmp_path):
        b2 = np.zeros(1, dtype=np.float32)
        b2[0] = np.nan
        _write_regression_head(tmp_path, b2=b2)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN b2 must be caught by the guard"

    def test_nan_in_w2_returns_none(self, tmp_path):
        w2 = np.zeros((1, 256), dtype=np.float32)
        w2[0, 5] = np.nan
        _write_regression_head(tmp_path, w2=w2)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN W2 must be caught by the guard"

    def test_zero_weights_produce_legal_midpoint(self, tmp_path):
        """Sanity check: all-zero weights → sigmoid(0) = 0.5, NOT None.

        Confirms the guard rejects ONLY non-finite or out-of-range values, not
        legitimate edge cases. Without this control, a too-aggressive guard
        could mask a real bug by returning None for valid corner inputs.
        """
        _write_regression_head(tmp_path)  # all zeros by default

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is not None
        assert result == pytest.approx(0.5, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Boot-time regression contract enforcement
# ─────────────────────────────────────────────────────────────────────────────


class TestRegisterTaskContractViolations:
    """A regression head that violates the scalar-output contract must raise."""

    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    def test_num_outputs_not_one_raises(self, tmp_path):
        _write_regression_head(
            tmp_path,
            num_outputs=3,
            w2=np.zeros((3, 256), dtype=np.float32),
            b2=np.zeros(3, dtype=np.float32),
        )
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="num_outputs"):
            svc._register_task("deliberation_score")

    def test_extra_feature_dim_nonzero_raises(self, tmp_path):
        _write_regression_head(tmp_path, extra_feature_dim=5)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="extra_feature_dim"):
            svc._register_task("deliberation_score")

    def test_output_activation_not_sigmoid_raises(self, tmp_path):
        _write_regression_head(tmp_path, output_activation="softmax")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="sigmoid"):
            svc._register_task("deliberation_score")

    def test_w2_row_count_mismatch_raises(self, tmp_path):
        # num_outputs=1 in meta but W2 has 2 rows — boot validation must catch this.
        _write_regression_head(
            tmp_path,
            w2=np.zeros((2, 256), dtype=np.float32),
            b2=np.zeros(2, dtype=np.float32),
        )
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="W2"):
            svc._register_task("deliberation_score")


# ─────────────────────────────────────────────────────────────────────────────
# Encoder/head drift — sha256 pin mismatch
# ─────────────────────────────────────────────────────────────────────────────


class TestSha256PinCheck:
    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    def test_wrong_sha256_in_meta_raises(self, tmp_path):
        """A head trained against a different encoder must NOT register."""
        _write_regression_head(
            tmp_path,
            sha256="0" * 64,  # Definitely not the real encoder sha.
        )
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            svc._register_task("deliberation_score")

    def test_matching_sha256_succeeds(self, tmp_path):
        _write_regression_head(tmp_path)  # uses real encoder sha by default
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        head = svc._register_task("deliberation_score")
        assert head is not None
        assert head.task_type == "regression"


# ─────────────────────────────────────────────────────────────────────────────
# Graceful degradation — missing/corrupt files return None, no crash
# ─────────────────────────────────────────────────────────────────────────────


class TestRegisterTaskGracefulDegradation:
    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    def test_missing_meta_returns_none(self, tmp_path):
        # Only a .npz, no meta — register should bail with None.
        _write_regression_head(tmp_path, skip_meta=True)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        # Public surface stays None, no exception.
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_corrupt_meta_returns_none(self, tmp_path):
        _write_regression_head(tmp_path, corrupt_meta=True)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_missing_npz_returns_none(self, tmp_path):
        _write_regression_head(tmp_path, skip_npz=True)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_meta_omits_head_asset_returns_none(self, tmp_path):
        _write_regression_head(tmp_path, omit_head_asset=True)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_meta_references_nonexistent_npz_returns_none(self, tmp_path):
        _write_regression_head(
            tmp_path,
            head_asset_name="totally-not-shipped.npz",
            skip_npz=True,
        )
        # Now write a meta whose head_asset points at the missing file.
        os.makedirs(os.path.join(str(tmp_path), "deliberation_score"), exist_ok=True)
        meta_path = os.path.join(
            str(tmp_path), "deliberation_score", "deliberation-score-classifier_meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump({
                "task_type": "regression",
                "base_encoder_sha256": _real_encoder_sha(),
                "input_dim": 768,
                "hidden_dim": 256,
                "num_outputs": 1,
                "num_classes": 1,
                "extra_feature_dim": 0,
                "output_activation": "sigmoid",
                "head_asset": "totally-not-shipped.npz",
                "labels": ["score"],
            }, f)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None


# ─────────────────────────────────────────────────────────────────────────────
# DeliberationScoreService — end-to-end consumer wrapper, no mocks
# ─────────────────────────────────────────────────────────────────────────────


class TestDeliberationScoreServiceReal:
    """Verify the wrapper's None semantics against the real classifier stack."""

    @classmethod
    def setup_class(cls):
        _get_session_and_tokenizer()

    def _real_svc(self):
        return DeliberationScoreService(
            inference_service=OnnxInferenceService(_DEFAULT_MODELS_DIR, _DEFAULT_PRETRAINED_DIR)
        )

    def test_classify_real_text_returns_finite_scalar(self):
        result = self._real_svc().classify(
            "Walk me through how you'd debug a memory leak in a long-running service."
        )
        assert result is not None
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_classify_empty_string_returns_none(self):
        assert self._real_svc().classify("") is None

    def test_classify_whitespace_only_returns_none(self):
        assert self._real_svc().classify("   \t\n  ") is None

    def test_classify_with_nan_head_returns_none(self, tmp_path):
        """End-to-end NaN propagation guard: bad head → wrapper returns None."""
        w1 = np.zeros((256, 768), dtype=np.float32)
        w1[0, 0] = np.nan
        _write_regression_head(tmp_path, w1=w1)

        svc = DeliberationScoreService(
            inference_service=OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        )
        assert svc.classify("anything goes") is None

    def test_classify_with_missing_head_returns_none(self, tmp_path):
        """Missing assets in the configured pretrained dir → wrapper returns None."""
        # Empty tmp dir — no meta, no npz.
        svc = DeliberationScoreService(
            inference_service=OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        )
        assert svc.classify("anything") is None
