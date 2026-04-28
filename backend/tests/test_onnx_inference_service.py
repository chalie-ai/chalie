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


_TASK_NAME = "deliberation_score"
_PREFIX = "deliberation-score"


def _default_arrays(num_outputs: int) -> dict:
    """Build the default all-zeros W1/b1/W2/b2 array set for ``num_outputs`` head."""
    return {
        "W1": np.zeros((256, 768), dtype=np.float32),
        "b1": np.zeros(256, dtype=np.float32),
        "W2": np.zeros((num_outputs, 256), dtype=np.float32),
        "b2": np.zeros(num_outputs, dtype=np.float32),
    }


def _build_meta(sha256: str, arrays: dict, *, num_outputs: int, extra_feature_dim: int, output_activation: str, head_asset: str | None) -> dict:
    """Assemble a regression-head meta dict from concrete pieces."""
    meta = {
        "labels": ["score"],
        "model_name": _TASK_NAME,
        "task_type": "regression",
        "base_encoder_sha256": sha256,
        "input_dim": int(arrays["W1"].shape[1]),
        "hidden_dim": int(arrays["W1"].shape[0]),
        "num_outputs": num_outputs,
        "num_classes": num_outputs,
        "extra_feature_dim": extra_feature_dim,
        "activation": "gelu",
        "output_activation": output_activation,
    }
    if head_asset is not None:
        meta["head_asset"] = head_asset
    return meta


def _ensure_task_dir(pretrained_dir) -> str:
    task_dir = os.path.join(str(pretrained_dir), _TASK_NAME)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def _write_npz(task_dir: str, head_filename: str, arrays: dict) -> None:
    np.savez(os.path.join(task_dir, head_filename), **arrays)


def _write_meta(task_dir: str, meta: dict) -> None:
    with open(os.path.join(task_dir, f"{_PREFIX}-classifier_meta.json"), "w") as f:
        json.dump(meta, f)


# ── Public factories — each focused on one negative-test axis ────────────────


def _write_valid_head(
    pretrained_dir,
    *,
    sha256: str | None = None,
    num_outputs: int = 1,
    extra_feature_dim: int = 0,
    output_activation: str = "sigmoid",
) -> None:
    """Write a valid head with default zero-weights. Override meta knobs to provoke contract failures."""
    task_dir = _ensure_task_dir(pretrained_dir)
    arrays = _default_arrays(num_outputs)
    head_filename = f"{_PREFIX}_head.npz"
    _write_npz(task_dir, head_filename, arrays)
    meta = _build_meta(
        sha256 or _real_encoder_sha(),
        arrays,
        num_outputs=num_outputs,
        extra_feature_dim=extra_feature_dim,
        output_activation=output_activation,
        head_asset=head_filename,
    )
    _write_meta(task_dir, meta)


def _write_head_with_arrays(
    pretrained_dir,
    *,
    num_outputs: int = 1,
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
) -> None:
    """Write a head with caller-supplied weight arrays (poison weights, shape mismatches)."""
    task_dir = _ensure_task_dir(pretrained_dir)
    defaults = _default_arrays(num_outputs)
    arrays = {
        "W1": w1 if w1 is not None else defaults["W1"],
        "b1": b1 if b1 is not None else defaults["b1"],
        "W2": w2 if w2 is not None else defaults["W2"],
        "b2": b2 if b2 is not None else defaults["b2"],
    }
    head_filename = f"{_PREFIX}_head.npz"
    _write_npz(task_dir, head_filename, arrays)
    meta = _build_meta(
        _real_encoder_sha(),
        arrays,
        num_outputs=num_outputs,
        extra_feature_dim=0,
        output_activation="sigmoid",
        head_asset=head_filename,
    )
    _write_meta(task_dir, meta)


def _write_corrupted_head(pretrained_dir, *, mode: str) -> None:
    """Write a head deliberately broken in one specific way.

    ``mode`` must be one of:
      - ``"skip_meta"``        — npz exists, no meta JSON
      - ``"skip_npz"``         — meta exists, no npz
      - ``"corrupt_meta"``     — meta JSON is malformed
      - ``"omit_head_asset"``  — meta JSON missing the ``head_asset`` key
    """
    task_dir = _ensure_task_dir(pretrained_dir)
    arrays = _default_arrays(1)
    head_filename = f"{_PREFIX}_head.npz"

    if mode != "skip_npz":
        _write_npz(task_dir, head_filename, arrays)
    if mode == "skip_meta":
        return

    meta_path = os.path.join(task_dir, f"{_PREFIX}-classifier_meta.json")
    if mode == "corrupt_meta":
        with open(meta_path, "w") as f:
            f.write("{this is not valid json")
        return
    head_asset = None if mode == "omit_head_asset" else head_filename
    meta = _build_meta(
        _real_encoder_sha(),
        arrays,
        num_outputs=1,
        extra_feature_dim=0,
        output_activation="sigmoid",
        head_asset=head_asset,
    )
    _write_meta(task_dir, meta)


def _write_head_with_custom_asset(pretrained_dir, *, head_asset_name: str) -> None:
    """Write only a meta whose ``head_asset`` points at a non-existent file (no npz written)."""
    task_dir = _ensure_task_dir(pretrained_dir)
    arrays = _default_arrays(1)
    meta = _build_meta(
        _real_encoder_sha(),
        arrays,
        num_outputs=1,
        extra_feature_dim=0,
        output_activation="sigmoid",
        head_asset=head_asset_name,
    )
    _write_meta(task_dir, meta)


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
        _write_head_with_arrays(tmp_path, w1=w1)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN W1 must be caught by the guard"

    def test_nan_in_b2_returns_none(self, tmp_path):
        b2 = np.zeros(1, dtype=np.float32)
        b2[0] = np.nan
        _write_head_with_arrays(tmp_path, b2=b2)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN b2 must be caught by the guard"

    def test_nan_in_w2_returns_none(self, tmp_path):
        w2 = np.zeros((1, 256), dtype=np.float32)
        w2[0, 5] = np.nan
        _write_head_with_arrays(tmp_path, w2=w2)

        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        result = svc.predict_scalar("deliberation_score", "anything")
        assert result is None, "NaN W2 must be caught by the guard"

    def test_zero_weights_produce_legal_midpoint(self, tmp_path):
        """Sanity check: all-zero weights → sigmoid(0) = 0.5, NOT None.

        Confirms the guard rejects ONLY non-finite or out-of-range values, not
        legitimate edge cases. Without this control, a too-aggressive guard
        could mask a real bug by returning None for valid corner inputs.
        """
        _write_valid_head(tmp_path)  # all zeros by default

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
        _write_head_with_arrays(
            tmp_path,
            num_outputs=3,
            w2=np.zeros((3, 256), dtype=np.float32),
            b2=np.zeros(3, dtype=np.float32),
        )
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="num_outputs"):
            svc._register_task("deliberation_score")

    def test_extra_feature_dim_nonzero_raises(self, tmp_path):
        _write_valid_head(tmp_path, extra_feature_dim=5)
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="extra_feature_dim"):
            svc._register_task("deliberation_score")

    def test_output_activation_not_sigmoid_raises(self, tmp_path):
        _write_valid_head(tmp_path, output_activation="softmax")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(ValueError, match="sigmoid"):
            svc._register_task("deliberation_score")

    def test_w2_row_count_mismatch_raises(self, tmp_path):
        # num_outputs=1 in meta but W2 has 2 rows — boot validation must catch this.
        _write_head_with_arrays(
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
        _write_valid_head(tmp_path, sha256="0" * 64)  # Definitely not the real encoder sha.
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            svc._register_task("deliberation_score")

    def test_matching_sha256_succeeds(self, tmp_path):
        _write_valid_head(tmp_path)  # uses real encoder sha by default
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
        _write_corrupted_head(tmp_path, mode="skip_meta")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        # Public surface stays None, no exception.
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_corrupt_meta_returns_none(self, tmp_path):
        _write_corrupted_head(tmp_path, mode="corrupt_meta")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_missing_npz_returns_none(self, tmp_path):
        _write_corrupted_head(tmp_path, mode="skip_npz")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_meta_omits_head_asset_returns_none(self, tmp_path):
        _write_corrupted_head(tmp_path, mode="omit_head_asset")
        svc = OnnxInferenceService(_DEFAULT_MODELS_DIR, str(tmp_path))
        assert svc._register_task("deliberation_score") is None
        assert svc.predict_scalar("deliberation_score", "x") is None

    def test_meta_references_nonexistent_npz_returns_none(self, tmp_path):
        _write_head_with_custom_asset(tmp_path, head_asset_name="totally-not-shipped.npz")
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
        _write_head_with_arrays(tmp_path, w1=w1)

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
