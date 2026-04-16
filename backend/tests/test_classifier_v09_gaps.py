"""
Gap-fill tests for the v0.9.0 ONNX classifier integration.

Covers behaviors that were asserted in nightly scenarios 094/095/096 but had
no unit-level coverage:

  1. _get_head caches the head on first load (shared encoder lifecycle)
  2. MLP forward uses torch nn.Linear weight convention (W1.T, W2.T)
  3. predict() returns (None, 0.0) when feature dim mismatches meta.input_dim
  4. _is_established() frozen thresholds for all four memory types
  5. _build_onnx_input() maps unknown source types to 'episode' guard
  6. _classify_pair_onnx() falls through to None when confidence < 0.65
  7. sha256 mismatch stops _get_head from caching the task
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ── Helpers shared across tests ───────────────────────────────────────────────

def _write_head_and_meta(task_dir: Path, W1, b1, W2, b2, sha: str,
                         input_dim: int, hidden_dim: int, num_classes: int,
                         labels: list, head_name: str = "head.npz"):
    """Write a .npz head + classifier_meta.json under task_dir."""
    np.savez(str(task_dir / head_name), W1=W1, b1=b1, W2=W2, b2=b2)
    meta = {
        "labels": labels,
        "model_name": task_dir.name,
        "model_type": "sequence_classifier",
        "base_encoder": "Alibaba-NLP/gte-modernbert-base",
        "base_encoder_sha256": sha,
        "embedding_dim": 768,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "activation": "gelu",
        "head_asset": head_name,
    }
    (task_dir / "classifier_meta.json").write_text(json.dumps(meta))


def _make_encoder_onnx(models_dir: Path, content: bytes = b"fake") -> Path:
    """Write a fake encoder ONNX and return its path."""
    p = models_dir / "gte-modernbert-base" / "onnx"
    p.mkdir(parents=True, exist_ok=True)
    onnx = p / "model.onnx"
    onnx.write_bytes(content)
    return onnx


def _reset_sha_cache():
    import services.onnx_inference_service as svc_mod
    svc_mod._encoder_sha256_cache = None


# ── Test 1: _get_head caches the head — shared encoder lifecycle ──────────────

@pytest.mark.unit
def test_get_head_caches_on_first_call(tmp_path):
    """_get_head must store the loaded head so the second call hits the cache,
    not _register_task again. This proves the encoder is not re-loaded per call."""
    _reset_sha_cache()
    import services.onnx_inference_service as svc_mod
    from services.onnx_inference_service import OnnxInferenceService

    onnx_path = _make_encoder_onnx(tmp_path)
    sha = svc_mod._compute_encoder_sha256(onnx_path)

    task_dir = tmp_path / "thinking_level"
    task_dir.mkdir()
    W1 = np.random.randn(256, 772).astype(np.float32)
    W2 = np.random.randn(3, 256).astype(np.float32)
    _write_head_and_meta(task_dir, W1, np.zeros(256, np.float32),
                         W2, np.zeros(3, np.float32), sha,
                         input_dim=772, hidden_dim=256, num_classes=3,
                         labels=["low", "medium", "high"])

    svc = OnnxInferenceService(str(tmp_path))

    # Count how many times _register_task is called
    register_calls = []
    original_register = svc._register_task

    def counting_register(task_name):
        register_calls.append(task_name)
        return original_register(task_name)

    svc._register_task = counting_register

    head_a = svc._get_head("thinking_level")
    head_b = svc._get_head("thinking_level")

    assert head_a is head_b, "_get_head must return the exact same object on repeat calls"
    assert len(register_calls) == 1, (
        f"_register_task called {len(register_calls)} times; expected exactly 1 "
        "(encoder must not be re-loaded per inference call)"
    )
    assert "thinking_level" in svc._heads


# ── Test 2: MLP weight convention — W1.T and W2.T are applied correctly ───────

@pytest.mark.unit
def test_mlp_forward_uses_transposed_weights():
    """
    nn.Linear stores (out, in). A (1,3) input through a head with W1=(5,3), W2=(2,5)
    must produce (1,2) logits — not (1,5) from a wrong matmul shape.

    This catches a regression where W1 was applied without .T (i.e. features @ W1
    instead of features @ W1.T), which would silently error or produce wrong shape.
    """
    from services.onnx_inference_service import _ClassifierHead

    # Deterministic: input [1, 0, 0] with identity-like W1 pinned to a known result
    W1 = np.eye(5, 3, dtype=np.float32)   # (hidden=5, input=3)  — nn.Linear convention
    b1 = np.zeros(5, dtype=np.float32)
    W2 = np.eye(2, 5, dtype=np.float32)   # (classes=2, hidden=5)
    b2 = np.zeros(2, dtype=np.float32)

    head = _ClassifierHead(W1=W1, b1=b1, W2=W2, b2=b2, labels=["a", "b"], activation="relu")

    features = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # (1, 3)
    logits = head.forward(features)

    assert logits.shape == (1, 2), f"Expected (1,2), got {logits.shape}"
    # With relu: h = relu([1,0,0,0,0]) = [1,0,0,0,0]; out = [1,0]
    assert abs(logits[0, 0] - 1.0) < 1e-5, f"Expected logits[0,0]=1.0, got {logits[0,0]}"
    assert abs(logits[0, 1] - 0.0) < 1e-5, f"Expected logits[0,1]=0.0, got {logits[0,1]}"


# ── Test 3: predict() returns (None, 0.0) on feature dim mismatch ────────────

@pytest.mark.unit
def test_predict_returns_none_on_dim_mismatch(tmp_path):
    """If extra_features makes the feature vector longer than meta.input_dim,
    predict() must return (None, 0.0) rather than crash or silently misclassify."""
    _reset_sha_cache()
    import services.onnx_inference_service as svc_mod
    from services.onnx_inference_service import OnnxInferenceService
    from unittest.mock import patch

    onnx_path = _make_encoder_onnx(tmp_path)
    sha = svc_mod._compute_encoder_sha256(onnx_path)

    task_dir = tmp_path / "contradiction"
    task_dir.mkdir()
    W1 = np.random.randn(256, 768).astype(np.float32)
    W2 = np.random.randn(5, 256).astype(np.float32)
    _write_head_and_meta(task_dir, W1, np.zeros(256, np.float32),
                         W2, np.zeros(5, np.float32), sha,
                         input_dim=768, hidden_dim=256, num_classes=5,
                         labels=["temporal_change", "true_contradiction",
                                 "context_dependent", "figurative", "compatible"])

    svc = OnnxInferenceService(str(tmp_path))
    fake_embedding = np.random.randn(1, 768).astype(np.float32)
    # extra_features with 4 dims → concatenated total is 772, but head expects 768
    wrong_extra = np.zeros((1, 4), dtype=np.float32)

    with patch.object(svc, "_embed", return_value=fake_embedding):
        label, confidence = svc.predict("contradiction", "test",
                                        extra_features=wrong_extra)

    assert label is None, f"Expected None for dim mismatch, got {label!r}"
    assert confidence == 0.0, f"Expected 0.0 confidence for dim mismatch, got {confidence}"


# ── Test 4: _is_established() frozen thresholds ───────────────────────────────

@pytest.mark.unit
class TestIsEstablished:
    """
    These thresholds are burned into the trained contradiction head.
    A regression here would silently degrade MLP accuracy.
    """

    def setup_method(self):
        from services.contradiction_classifier_service import _is_established
        self.fn = _is_established

    def test_incoming_always_false(self):
        assert self.fn("incoming", {"reinforcement_count": 99}) is False

    def test_episode_always_false(self):
        assert self.fn("episode", {"confidence": 1.0, "access_count": 100}) is False

    def test_trait_below_threshold_false(self):
        assert self.fn("trait", {"reinforcement_count": 2}) is False

    def test_trait_at_threshold_true(self):
        # threshold is exactly 3
        assert self.fn("trait", {"reinforcement_count": 3}) is True

    def test_trait_above_threshold_true(self):
        assert self.fn("trait", {"reinforcement_count": 10}) is True

    def test_trait_missing_key_defaults_to_one_false(self):
        # default reinforcement_count is 1, below 3
        assert self.fn("trait", {}) is False

    def test_concept_high_confidence_true(self):
        assert self.fn("concept", {"confidence": 0.8, "access_count": 0}) is True

    def test_concept_just_below_confidence_false(self):
        assert self.fn("concept", {"confidence": 0.79, "access_count": 0}) is False

    def test_concept_high_access_count_true(self):
        assert self.fn("concept", {"confidence": 0.0, "access_count": 5}) is True

    def test_concept_access_count_below_threshold_false(self):
        assert self.fn("concept", {"confidence": 0.0, "access_count": 4}) is False

    def test_unknown_type_returns_false(self):
        assert self.fn("xyzzy", {"confidence": 1.0, "reinforcement_count": 100}) is False


# ── Test 5: _build_onnx_input() maps unknown source type to 'episode' ─────────

@pytest.mark.unit
def test_build_onnx_input_unknown_type_mapped_to_episode():
    """
    Source types not in _ONNX_KNOWN_TYPES must be mapped to 'episode' as the
    final guard. The model has never seen anything else.
    """
    from services.contradiction_classifier_service import ContradictionClassifierService
    svc = ContradictionClassifierService()

    output = svc._build_onnx_input(
        text_a="a",
        text_b="b",
        meta_a={"source": "some_future_source_type"},
        meta_b={"source": "another_unknown_type"},
    )

    parsed = json.loads(output)
    assert parsed["type_a"] == "episode", (
        f"Unknown source type must map to 'episode', got {parsed['type_a']!r}"
    )
    assert parsed["type_b"] == "episode", (
        f"Unknown source type must map to 'episode', got {parsed['type_b']!r}"
    )


@pytest.mark.unit
def test_build_onnx_input_consolidation_type_mapped_to_concept():
    """consolidation_new and consolidation_existing map to 'concept' via _SOURCE_TYPE_MAP."""
    from services.contradiction_classifier_service import ContradictionClassifierService
    svc = ContradictionClassifierService()

    output = svc._build_onnx_input(
        text_a="a",
        text_b="b",
        meta_a={"source": "consolidation_new"},
        meta_b={"source": "consolidation_existing"},
    )

    parsed = json.loads(output)
    assert parsed["type_a"] == "concept", f"Expected 'concept', got {parsed['type_a']!r}"
    assert parsed["type_b"] == "concept", f"Expected 'concept', got {parsed['type_b']!r}"


# ── Test 6: _classify_pair_onnx falls through when confidence < 0.65 ──────────

@pytest.mark.unit
def test_classify_pair_onnx_fallthrough_below_threshold():
    """
    When the MLP returns confidence < 0.65, _classify_pair_onnx must return
    None so the LLM fallback path is taken. This is the 096 scenario path.
    """
    from services.contradiction_classifier_service import ContradictionClassifierService
    from unittest.mock import MagicMock, patch

    svc = ContradictionClassifierService()
    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = ("temporal_change", 0.64)  # just below 0.65

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc._classify_pair_onnx(
            text_a="I live in Swieqi",
            text_b="I live in Valletta",
            meta_a={"source": "incoming"},
            meta_b={"source": "trait"},
        )

    assert result is None, (
        f"Expected None (fallthrough to LLM) when confidence=0.64 < 0.65, got {result!r}"
    )


@pytest.mark.unit
def test_classify_pair_onnx_passes_at_threshold():
    """Confidence exactly at 0.65 must NOT fall through."""
    from services.contradiction_classifier_service import ContradictionClassifierService
    from unittest.mock import MagicMock, patch

    svc = ContradictionClassifierService()
    mock_onnx = MagicMock()
    mock_onnx.predict.return_value = ("temporal_change", 0.65)

    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mock_onnx):
        result = svc._classify_pair_onnx(
            text_a="I live in Swieqi",
            text_b="I live in Valletta",
            meta_a={"source": "incoming"},
            meta_b={"source": "trait"},
        )

    assert result is not None, "Confidence=0.65 is at threshold and must not fall through"
    assert result["classification"] == "temporal_change"


# ── Test 7: sha256 mismatch — _register_task raises and task is not cached ────

@pytest.mark.unit
def test_sha256_mismatch_raises_and_does_not_cache(tmp_path):
    """
    _register_task raises RuntimeError on sha256 mismatch (verified by
    test_boot_sha256_mismatch_raises_runtime_error in test_onnx_inference_service_v09.py).

    BUG (documented): _get_head does not catch this RuntimeError — the exception
    propagates through predict() to the caller. ThinkingLevelClassifierService.classify()
    wraps the call in a broad except and swallows it (returning the sticky fallback),
    so end-user impact is contained. But predict() itself violates its own docstring
    ("label is None if model isn't available") for the sha256 case.

    This test asserts the ACTUAL behavior (raises) so any fix to make predict()
    return (None, 0.0) instead will be detected here and the test can be updated
    to verify the corrected contract.

    Nightly scenario 095 tests the positive path (boot marker present, no mismatch
    line) from the deployed container side. This test covers the negative path at
    the unit level.
    """
    _reset_sha_cache()
    import services.onnx_inference_service as svc_mod
    from services.onnx_inference_service import OnnxInferenceService

    onnx_path = _make_encoder_onnx(tmp_path)
    actual_sha = svc_mod._compute_encoder_sha256(onnx_path)
    wrong_sha = "0" * 64
    assert wrong_sha != actual_sha

    task_dir = tmp_path / "thinking_level"
    task_dir.mkdir()
    W1 = np.random.randn(256, 772).astype(np.float32)
    W2 = np.random.randn(3, 256).astype(np.float32)
    _write_head_and_meta(task_dir, W1, np.zeros(256, np.float32),
                         W2, np.zeros(3, np.float32), wrong_sha,
                         input_dim=772, hidden_dim=256, num_classes=3,
                         labels=["low", "medium", "high"])

    svc = OnnxInferenceService(str(tmp_path))

    # _register_task raises RuntimeError; _get_head does not catch it;
    # predict() propagates it. The RuntimeError must name the mismatch.
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        svc._get_head("thinking_level")

    # The failed task must not be cached
    assert "thinking_level" not in svc._heads, (
        "Mismatched task must not be stored in _heads"
    )


@pytest.mark.unit
def test_thinking_level_classify_survives_sha256_mismatch(tmp_path):
    """
    ThinkingLevelClassifierService.classify() wraps predict() in a broad except,
    so a sha256-mismatch RuntimeError from the ONNX layer must produce the
    sticky fallback response rather than propagating to the caller.

    This is the caller-side complement of test_sha256_mismatch_raises_and_does_not_cache.
    """
    _reset_sha_cache()
    import services.onnx_inference_service as svc_mod
    from services.onnx_inference_service import OnnxInferenceService
    from services.thinking_level_classifier_service import ThinkingLevelClassifierService

    onnx_path = _make_encoder_onnx(tmp_path)
    actual_sha = svc_mod._compute_encoder_sha256(onnx_path)
    wrong_sha = "0" * 64
    assert wrong_sha != actual_sha

    task_dir = tmp_path / "thinking_level"
    task_dir.mkdir()
    W1 = np.random.randn(256, 772).astype(np.float32)
    W2 = np.random.randn(3, 256).astype(np.float32)
    _write_head_and_meta(task_dir, W1, np.zeros(256, np.float32),
                         W2, np.zeros(3, np.float32), wrong_sha,
                         input_dim=772, hidden_dim=256, num_classes=3,
                         labels=["low", "medium", "high"])

    mismatched_svc = OnnxInferenceService(str(tmp_path))

    from unittest.mock import patch
    # Inject the mismatched ONNX service as the singleton
    with patch("services.onnx_inference_service.get_onnx_inference_service",
               return_value=mismatched_svc):
        result = ThinkingLevelClassifierService().classify("test", prev_level="high")

    # Must not raise; must return sticky fallback
    assert isinstance(result, dict)
    assert result["fallback"] is True
    assert result["level"] in ("low", "medium", "high")


# ── Degraded-state visibility (boot-gate health surface) ─────────────────────


@pytest.mark.unit
def test_ready_reports_false_when_degraded():
    """
    `ready` must return False whenever any task failed registration, even after
    the boot loop completed. Prevents the boot gate from being silently
    neutralized against the /api/system health endpoint.
    """
    from services.onnx_inference_service import OnnxInferenceService

    svc = OnnxInferenceService("/tmp/nonexistent-models-dir")
    svc._ready = True
    svc._failed_registrations = [("thinking_level", "sha256 mismatch")]

    assert svc.ready is False, "ready must be False when degraded"
    assert svc.degraded is True
    assert svc.failed_registrations == [("thinking_level", "sha256 mismatch")]


@pytest.mark.unit
def test_ready_reports_true_when_clean():
    """`ready` is True only after registration completed AND no failures."""
    from services.onnx_inference_service import OnnxInferenceService

    svc = OnnxInferenceService("/tmp/nonexistent-models-dir")
    svc._ready = True
    svc._failed_registrations = []

    assert svc.ready is True
    assert svc.degraded is False


@pytest.mark.unit
def test_degraded_false_during_loading():
    """`degraded` is False until the registration loop completes (loading state)."""
    from services.onnx_inference_service import OnnxInferenceService

    svc = OnnxInferenceService("/tmp/nonexistent-models-dir")
    # _ready not yet flipped — still loading
    assert svc.ready is False
    assert svc.degraded is False


# ── check_new_trait existing_meta forwarding ─────────────────────────────────


@pytest.mark.unit
def test_check_new_trait_forwards_existing_meta_to_onnx():
    """
    check_new_trait must pass existing_meta through to _build_onnx_input so the
    model sees correct age_b_days and established_b. Without this, a
    well-reinforced trait looks fresh and the contradiction head receives a
    training/inference mismatch.
    """
    from unittest.mock import patch, MagicMock
    from services.contradiction_classifier_service import ContradictionClassifierService

    svc = ContradictionClassifierService()

    # Capture the meta dicts that reach _build_onnx_input
    captured = {}
    real_build = svc._build_onnx_input

    def spy_build(text_a, text_b, meta_a, meta_b):
        captured['meta_a'] = dict(meta_a)
        captured['meta_b'] = dict(meta_b)
        return real_build(text_a, text_b, meta_a, meta_b)

    with patch.object(svc, '_build_onnx_input', side_effect=spy_build), \
         patch('services.onnx_inference_service.get_onnx_inference_service') as get_svc:
        mock_inf = MagicMock()
        # Force ONNX path to return None so we exit before LLM
        mock_inf.predict.return_value = (None, 0.0)
        get_svc.return_value = mock_inf

        existing_meta = {
            'created_at': '2025-01-01T00:00:00+00:00',
            'reinforcement_count': 17,
            'confidence': 0.95,
        }
        svc.check_new_trait(
            "I live in Valletta",
            "I live in Swieqi",
            source='chat',
            existing_meta=existing_meta,
        )

    assert captured['meta_a']['type'] == 'incoming'
    assert captured['meta_a']['established'] is False
    assert captured['meta_b']['type'] == 'trait'
    assert captured['meta_b']['created_at'] == '2025-01-01T00:00:00+00:00'
    assert captured['meta_b']['reinforcement_count'] == 17
    assert captured['meta_b']['confidence'] == 0.95


@pytest.mark.unit
def test_check_new_trait_without_existing_meta_still_works():
    """Backwards-compat: existing callers that don't pass existing_meta still get
    a sensible meta_b (type=trait) — just without age/reinforcement signals."""
    from unittest.mock import patch, MagicMock
    from services.contradiction_classifier_service import ContradictionClassifierService

    svc = ContradictionClassifierService()
    captured = {}
    real_build = svc._build_onnx_input

    def spy_build(text_a, text_b, meta_a, meta_b):
        captured['meta_b'] = dict(meta_b)
        return real_build(text_a, text_b, meta_a, meta_b)

    with patch.object(svc, '_build_onnx_input', side_effect=spy_build), \
         patch('services.onnx_inference_service.get_onnx_inference_service') as get_svc:
        mock_inf = MagicMock()
        mock_inf.predict.return_value = (None, 0.0)
        get_svc.return_value = mock_inf

        svc.check_new_trait("new", "old", source='chat')

    assert captured['meta_b']['type'] == 'trait'
    assert 'created_at' not in captured['meta_b']


@pytest.mark.unit
def test_build_onnx_input_prefers_explicit_type_over_source():
    """
    When meta has both `type` and `source`, `type` wins. This lets callers
    disambiguate when source='chat' but the memory itself is a trait.
    """
    from services.contradiction_classifier_service import ContradictionClassifierService

    svc = ContradictionClassifierService()
    payload = svc._build_onnx_input(
        "a", "b",
        meta_a={'type': 'trait', 'source': 'chat'},
        meta_b={'type': 'trait'},
    )
    parsed = json.loads(payload)
    assert parsed['type_a'] == 'trait', "explicit type must win over source"
    assert parsed['type_b'] == 'trait'
