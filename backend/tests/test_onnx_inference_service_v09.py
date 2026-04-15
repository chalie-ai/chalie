"""
Unit tests for OnnxInferenceService v0.9.0 — shared encoder + MLP head architecture.

Tests the boot-time sha256 pin, MLP head forward pass, and softmax stability.
All tests are pure-unit (no ONNX session, no filesystem access beyond tmp).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Suppress noisy logs during tests
logging.disable(logging.CRITICAL)


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_npz(tmp_path: Path, W1, b1, W2, b2) -> Path:
    """Write a synthetic head .npz to tmp_path and return its path."""
    npz_path = tmp_path / "head.npz"
    np.savez(str(npz_path), W1=W1, b1=b1, W2=W2, b2=b2)
    return npz_path


def _make_meta(tmp_path: Path, task_name: str, head_asset: str, sha256: str,
               input_dim: int, hidden_dim: int, num_classes: int,
               labels: list, activation: str = "gelu") -> Path:
    """Write a synthetic classifier_meta.json and return its path."""
    meta = {
        "labels": labels,
        "model_name": task_name,
        "model_type": "sequence_classifier",
        "base_encoder": "Alibaba-NLP/gte-modernbert-base",
        "base_encoder_sha256": sha256,
        "embedding_dim": 768,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "activation": activation,
        "head_asset": head_asset,
    }
    meta_path = tmp_path / "classifier_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta_path


def _make_fake_onnx(tmp_path: Path, content: bytes = b"fake") -> Path:
    """Write a fake ONNX file for sha256 testing."""
    p = tmp_path / "model.onnx"
    p.write_bytes(content)
    return p


# ── Import the module under test ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_encoder_sha256_cache():
    """Reset the module-level sha256 cache between tests."""
    import services.onnx_inference_service as svc_mod
    original = svc_mod._encoder_sha256_cache
    svc_mod._encoder_sha256_cache = None
    yield
    svc_mod._encoder_sha256_cache = original


@pytest.fixture()
def service_factory(tmp_path):
    """Return a factory that creates an OnnxInferenceService with a tmp models_dir."""
    def _make():
        import importlib
        import services.onnx_inference_service as svc_mod
        importlib.reload(svc_mod)  # fresh singleton state
        svc_mod._encoder_sha256_cache = None
        from services.onnx_inference_service import OnnxInferenceService
        return OnnxInferenceService(str(tmp_path))
    return _make


# ── Test 1: sha256 mismatch raises RuntimeError and refuses registration ──────

@pytest.mark.unit
def test_boot_sha256_mismatch_raises_runtime_error(tmp_path):
    """Feed a meta with wrong sha; assert RuntimeError with 'sha256 mismatch' substring."""
    import services.onnx_inference_service as svc_mod
    svc_mod._encoder_sha256_cache = None

    from services.onnx_inference_service import OnnxInferenceService

    # Create fake encoder ONNX
    encoder_dir = tmp_path / "gte-modernbert-base" / "onnx"
    encoder_dir.mkdir(parents=True)
    encoder_onnx = encoder_dir / "model.onnx"
    encoder_onnx.write_bytes(b"not_the_real_model")
    actual_sha = svc_mod._compute_encoder_sha256(encoder_onnx)

    # Create task dir with a WRONG sha in meta
    task_dir = tmp_path / "thinking_level"
    task_dir.mkdir()
    wrong_sha = "0" * 64  # guaranteed wrong
    assert wrong_sha != actual_sha

    # Synthetic head (shape doesn't matter for this test — sha check fires first)
    W1 = np.random.randn(256, 772).astype(np.float32)
    b1 = np.zeros(256, dtype=np.float32)
    W2 = np.random.randn(3, 256).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)
    _make_npz(task_dir, W1, b1, W2, b2)
    _make_meta(task_dir, "thinking_level", "head.npz", wrong_sha,
               input_dim=772, hidden_dim=256, num_classes=3,
               labels=["low", "medium", "high"])

    svc = OnnxInferenceService(str(tmp_path))

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        svc._register_task("thinking_level")

    # Task must NOT be registered
    assert "thinking_level" not in svc._heads


# ── Test 2: sha256 match registers and emits boot marker ─────────────────────

@pytest.mark.unit
def test_boot_sha256_match_registers_and_logs_marker(tmp_path, caplog):
    """Matching sha256 → head registered, boot marker logged."""
    import services.onnx_inference_service as svc_mod
    svc_mod._encoder_sha256_cache = None

    from services.onnx_inference_service import OnnxInferenceService

    # Create fake encoder ONNX and compute its real sha256
    encoder_dir = tmp_path / "gte-modernbert-base" / "onnx"
    encoder_dir.mkdir(parents=True)
    encoder_onnx = encoder_dir / "model.onnx"
    encoder_onnx.write_bytes(b"synthetic_encoder_bytes")
    correct_sha = svc_mod._compute_encoder_sha256(encoder_onnx)

    # Create task dir with correct sha
    task_dir = tmp_path / "contradiction"
    task_dir.mkdir()

    W1 = np.random.randn(256, 768).astype(np.float32)
    b1 = np.zeros(256, dtype=np.float32)
    W2 = np.random.randn(5, 256).astype(np.float32)
    b2 = np.zeros(5, dtype=np.float32)
    _make_npz(task_dir, W1, b1, W2, b2)
    _make_meta(task_dir, "contradiction", "head.npz", correct_sha,
               input_dim=768, hidden_dim=256, num_classes=5,
               labels=["temporal_change", "true_contradiction",
                       "context_dependent", "figurative", "compatible"])

    svc = OnnxInferenceService(str(tmp_path))

    with caplog.at_level(logging.INFO, logger="services.onnx_inference_service"):
        logging.disable(logging.NOTSET)
        head = svc._register_task("contradiction")

    assert head is not None
    assert "contradiction" not in svc._heads  # _register_task doesn't cache; _get_head does

    # Check log marker
    marker_lines = [r.getMessage() for r in caplog.records
                    if "[CLASSIFIER BOOT]" in r.getMessage()]
    assert any("contradiction" in line for line in marker_lines), (
        f"Expected [CLASSIFIER BOOT] contradiction in logs. Got: {caplog.text}"
    )


# ── Test 3: MLP head forward shape ────────────────────────────────────────────

@pytest.mark.unit
def test_mlp_head_forward_shape(tmp_path):
    """Synthetic W1/b1/W2/b2; verify predict() returns (label, confidence) with label in meta.labels."""
    import services.onnx_inference_service as svc_mod
    svc_mod._encoder_sha256_cache = None

    from services.onnx_inference_service import OnnxInferenceService, _ClassifierHead

    labels = ["low", "medium", "high"]
    input_dim, hidden_dim, num_classes = 772, 256, 3

    W1 = np.random.randn(hidden_dim, input_dim).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(num_classes, hidden_dim).astype(np.float32)
    b2 = np.zeros(num_classes, dtype=np.float32)

    head = _ClassifierHead(W1=W1, b1=b1, W2=W2, b2=b2, labels=labels, activation="gelu")

    # Verify forward pass shape
    features = np.random.randn(1, input_dim).astype(np.float32)
    logits = head.forward(features)
    assert logits.shape == (1, num_classes), f"Expected (1, 3), got {logits.shape}"

    # Verify softmax sums to 1
    from services.onnx_inference_service import _softmax
    probs = _softmax(logits)[0]
    assert abs(probs.sum() - 1.0) < 1e-5, f"Softmax sum: {probs.sum()}"

    winner_idx = int(np.argmax(probs))
    label = labels[winner_idx]
    confidence = float(probs[winner_idx])

    assert label in labels
    assert 0.0 <= confidence <= 1.0

    # Verify the full predict() path by mocking _embed and _get_head
    encoder_dir = tmp_path / "gte-modernbert-base" / "onnx"
    encoder_dir.mkdir(parents=True)
    encoder_onnx = encoder_dir / "model.onnx"
    encoder_onnx.write_bytes(b"fake_encoder")
    correct_sha = svc_mod._compute_encoder_sha256(encoder_onnx)

    task_dir = tmp_path / "thinking_level"
    task_dir.mkdir()
    npz_path = _make_npz(task_dir, W1, b1, W2, b2)
    npz_path.rename(task_dir / "thinking_level_head.npz")
    _make_meta(task_dir, "thinking_level", "thinking_level_head.npz", correct_sha,
               input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,
               labels=labels)

    svc = OnnxInferenceService(str(tmp_path))

    # Mock _embed to return a known (1, 768) embedding
    fake_embedding = np.random.randn(1, 768).astype(np.float32)
    extra = np.zeros((1, 4), dtype=np.float32)
    extra[0, 0] = 1.0  # none → index 0

    with patch.object(svc, "_embed", return_value=fake_embedding):
        result_label, result_conf = svc.predict("thinking_level", "test prompt",
                                                 extra_features=extra)

    assert result_label in labels
    assert 0.0 <= result_conf <= 1.0


# ── Test 4: softmax numerical stability ───────────────────────────────────────

@pytest.mark.unit
def test_softmax_numerical_stability():
    """Extreme-logit inputs must not overflow or produce NaN/Inf."""
    from services.onnx_inference_service import _softmax

    # Large positive values
    logits_large = np.array([[1e30, 1e30, 1e30]], dtype=np.float32)
    probs_large = _softmax(logits_large)
    assert not np.any(np.isnan(probs_large)), "NaN in large logits"
    assert not np.any(np.isinf(probs_large)), "Inf in large logits"
    assert abs(probs_large.sum() - 1.0) < 1e-5

    # Large negative values
    logits_neg = np.array([[-1e30, -1e30, -1e30]], dtype=np.float32)
    probs_neg = _softmax(logits_neg)
    assert not np.any(np.isnan(probs_neg)), "NaN in negative logits"
    assert abs(probs_neg.sum() - 1.0) < 1e-5

    # Mixed extreme
    logits_mixed = np.array([[1e30, -1e30, 0.0]], dtype=np.float32)
    probs_mixed = _softmax(logits_mixed)
    assert not np.any(np.isnan(probs_mixed)), "NaN in mixed logits"
    assert abs(probs_mixed.sum() - 1.0) < 1e-5
    # First class should dominate
    assert probs_mixed[0, 0] > 0.99, f"Expected first class to dominate: {probs_mixed}"
