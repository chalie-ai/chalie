# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ONNX Inference Service — shared gte-modernbert encoder + swappable MLP heads.

Architecture:
    1 shared ONNX encoder (gte-modernbert-base, downloaded into ``data/models/``)
    N 2-layer MLP heads loaded from per-task .npz files (shipped in
    ``backend/pre-trained/<task>/``).

Pre-shipped classifier files (meta + .npz) live under ``backend/pre-trained/``
and are tracked in git. Runtime-downloaded models (encoders, voice, doc2query)
stay under ``data/models/`` and are NOT tracked. All paths are hard-coded in
:mod:`paths`.

Boot-time sha256 pin:
    Computed once (streaming) against ``data/models/gte-modernbert-base/onnx/model.onnx``.
    Must match classifier_meta.json::base_encoder_sha256 for every registered task.
    On mismatch: RuntimeError raised, task not registered.

Thread-safe — multiple workers can call predict() / predict_scalar() concurrently.
"""

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ONNX]"

# Tasks to register on boot.
# Each entry: (task_name, asset_name_prefix)
# - meta file:  pre-trained/<task>/<prefix>-classifier_meta.json
# - head file:  pre-trained/<task>/<name from meta["head_asset"]>
MODEL_REGISTRY = [
    ("deliberation_score", "deliberation-score"),
    ("mode_detector", "mode-detector"),
]

_TASK_PREFIXES: Dict[str, str] = dict(MODEL_REGISTRY)

# sha256 of the shared encoder ONNX — computed once, cached here
_encoder_sha256_cache: Optional[str] = None
_encoder_sha256_lock = threading.Lock()


def _compute_encoder_sha256(onnx_path: Path) -> str:
    """Compute sha256 of the encoder ONNX file via streaming (no full-file load)."""
    h = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_encoder_sha256(onnx_path: Path) -> str:
    """Return cached sha256 of the shared encoder, computing it once per process."""
    global _encoder_sha256_cache
    if _encoder_sha256_cache is not None:
        return _encoder_sha256_cache

    with _encoder_sha256_lock:
        if _encoder_sha256_cache is not None:
            return _encoder_sha256_cache
        _encoder_sha256_cache = _compute_encoder_sha256(onnx_path)
        return _encoder_sha256_cache


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximation matching PyTorch's default)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum(axis=-1, keepdims=True)


class _ClassifierHead:
    """2-layer MLP head loaded from a .npz file.

    Weight convention: torch nn.Linear stores weights as (out_features, in_features),
    so W1.shape == (hidden_dim, input_dim) and W2.shape == (num_classes, hidden_dim).
    Forward uses the transpose: features @ W1.T.
    """

    __slots__ = ("W1", "b1", "W2", "b2", "labels", "activation",
                 "input_dim", "hidden_dim", "num_classes",
                 "task_type", "output_activation", "num_outputs",
                 "extra_feature_dim")

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        labels: List[str],
        activation: str = "gelu",
        task_type: str = "classification",
        output_activation: str = "softmax",
        num_outputs: int = 0,
        extra_feature_dim: int = 0,
    ):
        self.W1 = w1.astype(np.float32)    # (hidden_dim, input_dim)
        self.b1 = b1.astype(np.float32)    # (hidden_dim,)
        self.W2 = w2.astype(np.float32)    # (num_classes, hidden_dim)
        self.b2 = b2.astype(np.float32)    # (num_classes,)
        self.labels = labels
        self.activation = activation
        self.task_type = task_type
        self.output_activation = output_activation
        self.num_outputs = num_outputs
        self.extra_feature_dim = extra_feature_dim
        self.hidden_dim, self.input_dim = w1.shape
        self.num_classes = w2.shape[0]

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Apply the 2-layer MLP.

        Args:
            features: (batch, input_dim) float32

        Returns:
            logits: (batch, num_classes) float32
        """
        h = features @ self.W1.T + self.b1
        if self.activation == "gelu":
            h = _gelu(h)
        else:
            h = np.maximum(0.0, h)  # relu fallback
        return h @ self.W2.T + self.b2


def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Masked mean pool over token dimension."""
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_emb = (last_hidden_state * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1).clip(min=1e-9)
    return sum_emb / sum_mask


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
    return embeddings / norms


class OnnxInferenceService:
    """
    Shared gte-modernbert encoder + swappable 2-layer MLP classifier heads.

    Classifier heads (meta + .npz) are pre-shipped under
    ``backend/pre-trained/<task>/``. The shared encoder ONNX is downloaded
    into ``data/models/gte-modernbert-base/``. No network access for heads.

    Usage:
        svc = get_onnx_inference_service()
        scalar = svc.predict_scalar("deliberation_score", input_text)
        label, confidence = svc.predict("mode_detector", input_text)
    """

    def __init__(self, models_dir: str, pretrained_dir: str):
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._pretrained_dir = Path(pretrained_dir)

        # Registered classifier heads (task_name → _ClassifierHead)
        self._heads: Dict[str, _ClassifierHead] = {}
        self._heads_lock = threading.Lock()

        # Boot readiness — set to True after registration attempt completes
        self._ready = False
        # Tasks whose boot-time registration failed.
        # List of (task_name, error_message) tuples.
        self._failed_registrations: list[tuple[str, str]] = []

    @property
    def ready(self) -> bool:
        """True only after registration attempt completed AND every task registered."""
        return self._ready and not self._failed_registrations

    @property
    def degraded(self) -> bool:
        """True if registration finished but at least one task failed."""
        return self._ready and bool(self._failed_registrations)

    @property
    def failed_registrations(self) -> list[tuple[str, str]]:
        """Snapshot of (task_name, error) for tasks whose registration failed."""
        return list(self._failed_registrations)

    # ── Encoder access (borrowed from embedding_service) ──────────────────────

    def _get_encoder(self):
        """Return (session, tokenizer, output_names, input_names) from embedding_service."""
        from services import embedding_service as _emb_mod
        session, tokenizer = _emb_mod._get_session_and_tokenizer()
        return session, tokenizer, _emb_mod._output_names, _emb_mod._input_names

    def _encoder_onnx_path(self) -> Path:
        """Canonical path to the shared encoder ONNX."""
        return self._models_dir / "gte-modernbert-base" / "onnx" / "model.onnx"

    def _embed(self, text: str) -> np.ndarray:
        """Tokenize + encode one text → (1, 768) float32 L2-normalised.

        Uses max_length=256 to match training. Shares the embedding_service session
        so the model is loaded only once per process.
        """
        session, tokenizer, output_names, input_names = self._get_encoder()

        encoded = tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        outputs = session.run(None, feed)

        if "sentence_embedding" in output_names:
            pooled = outputs[output_names.index("sentence_embedding")]
        else:
            last_hidden = outputs[output_names.index("last_hidden_state")]
            pooled = _mean_pool(last_hidden, attention_mask)

        return _l2_normalize(pooled).astype(np.float32)  # (1, 768)

    # ── Task registration ─────────────────────────────────────────────────────

    def _load_task_meta(self, task_name: str):
        """Locate and parse the task's meta JSON. Returns (meta_dict, task_dir) or None."""
        task_dir = self._pretrained_dir / task_name
        prefix = _TASK_PREFIXES.get(task_name, task_name)
        meta_path = task_dir / f"{prefix}-classifier_meta.json"
        if not meta_path.exists():
            logger.warning(f"{LOG_PREFIX} Missing {meta_path.name} for task '{task_name}' at {task_dir}")
            return None
        try:
            with open(meta_path) as f:
                return json.load(f), task_dir
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"{LOG_PREFIX} Cannot read meta for '{task_name}': {e}")
            return None

    def _verify_encoder_sha(self, task_name: str, expected_sha: str) -> str | None:
        """Compute the encoder's sha256 and compare against `expected_sha`. Returns actual sha or None."""
        onnx_path = self._encoder_onnx_path()
        if not onnx_path.exists():
            logger.warning(f"{LOG_PREFIX} Encoder ONNX not found at {onnx_path}")
            return None
        actual_sha = _get_encoder_sha256(onnx_path)
        if actual_sha.lower() != expected_sha:
            raise RuntimeError(
                f"[CLASSIFIER BOOT] sha256 mismatch: task={task_name} "
                f"expected={expected_sha} got={actual_sha}"
            )
        return actual_sha

    def _load_head_arrays(self, task_name: str, task_dir, meta: dict):
        """Load W1/b1/W2/b2 from the head .npz; returns the four arrays or None on failure."""
        head_asset = meta.get("head_asset")
        if not head_asset:
            logger.warning(f"{LOG_PREFIX} No head_asset in meta for '{task_name}'")
            return None
        npz_path = task_dir / head_asset
        if not npz_path.exists():
            logger.warning(f"{LOG_PREFIX} Head file not found: {npz_path}")
            return None
        try:
            with np.load(str(npz_path)) as f:
                return (
                    f["W1"].astype(np.float32),
                    f["b1"].astype(np.float32),
                    f["W2"].astype(np.float32),
                    f["b2"].astype(np.float32),
                )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load head for '{task_name}': {e}")
            return None

    def _register_task(self, task_name: str) -> Optional[_ClassifierHead]:
        """Load meta + head for one task, perform sha256 pin check, emit boot marker.

        Returns the head on success. Raises RuntimeError on:
          - sha256 mismatch
          - regression task contract violations (num_outputs != 1, extra_feature_dim != 0,
            output_activation != sigmoid)
        Returns None if the task directory or meta is missing.
        """
        loaded = self._load_task_meta(task_name)
        if loaded is None:
            return None
        meta, task_dir = loaded

        actual_sha = self._verify_encoder_sha(task_name, meta.get("base_encoder_sha256", "").lower())
        if actual_sha is None:
            return None

        arrays = self._load_head_arrays(task_name, task_dir, meta)
        if arrays is None:
            return None
        w1, b1, w2, b2 = arrays

        labels = meta.get("labels", [])
        activation = meta.get("activation", meta.get("mlp_activation", "gelu"))
        task_type = meta.get("task_type", "classification")
        output_activation = meta.get("output_activation", "softmax")
        num_outputs = meta.get("num_outputs", meta.get("num_classes", w2.shape[0]))
        extra_feature_dim = meta.get("extra_feature_dim", 0)
        input_dim = meta.get("input_dim", w1.shape[1])
        hidden_dim = meta.get("hidden_dim", w1.shape[0])
        num_classes = meta.get("num_classes", w2.shape[0])

        # Boot-time assertions for regression tasks
        if task_type == "regression":
            if num_outputs != 1:
                raise ValueError(
                    f"deliberation-score head expected scalar regression (num_outputs=1); "
                    f"got num_outputs={num_outputs} in meta for task '{task_name}'"
                )
            if extra_feature_dim != 0:
                raise ValueError(
                    f"deliberation-score head expected extra_feature_dim=0; "
                    f"got extra_feature_dim={extra_feature_dim} for task '{task_name}'. "
                    f"One-hot concatenation is forbidden for regression heads."
                )
            if output_activation != "sigmoid":
                raise ValueError(
                    f"deliberation-score head expected output_activation='sigmoid'; "
                    f"got {output_activation!r} for task '{task_name}'"
                )
            if w2.shape[0] != 1:
                raise ValueError(
                    f"deliberation-score head W2 row count must be 1 for scalar regression; "
                    f"got W2.shape={w2.shape} for task '{task_name}'"
                )

        head = _ClassifierHead(
            w1=w1, b1=b1, w2=w2, b2=b2,
            labels=labels,
            activation=activation,
            task_type=task_type,
            output_activation=output_activation,
            num_outputs=num_outputs,
            extra_feature_dim=extra_feature_dim,
        )

        # Emit boot marker — format is contractual (scenarios grep exact prefix)
        if task_type == "regression":
            logger.info(
                "[CLASSIFIER BOOT] %s sha256=%s input_dim=%d num_outputs=%d task=%s",
                task_name, actual_sha, input_dim, num_outputs, task_type,
            )
        else:
            logger.info(
                "[CLASSIFIER BOOT] %s sha256=%s input_dim=%d hidden_dim=%d "
                "num_classes=%d activation=%s",
                task_name, actual_sha, input_dim, hidden_dim, num_classes, activation,
            )

        return head

    def _get_head(self, task_name: str) -> Optional[_ClassifierHead]:
        """Return the registered head for task_name, loading it on first call."""
        if task_name in self._heads:
            return self._heads[task_name]

        with self._heads_lock:
            if task_name in self._heads:
                return self._heads[task_name]

            head = self._register_task(task_name)
            if head is not None:
                self._heads[task_name] = head
            return head

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_scalar(self, task_name: str, text: str) -> 'float | None':
        """Regression inference: tokens → pooled → L2 → MLP → sigmoid → float.

        Asserts at registration time (via _register_task) that the task's
        classifier_meta.json has:
            task_type == "regression"
            num_outputs == 1
            extra_feature_dim == 0
            output_activation == "sigmoid"

        Returns float in [0, 1], or None if the head is unavailable.
        Never raises — any error is logged at WARNING and None is returned.

        Args:
            task_name: Registered task with task_type="regression"
                       (e.g. 'deliberation_score').
            text:      Raw text to encode and score.
        """
        head = self._get_head(task_name)
        if head is None:
            return None

        if head.task_type != "regression":
            raise ValueError(
                f"predict_scalar called on non-regression task '{task_name}' "
                f"(task_type={head.task_type!r}). Use predict() or predict_all_sigmoids() instead."
            )

        try:
            start = time.perf_counter()
            embedding = self._embed(text)  # (1, 768)

            if not np.isfinite(embedding).all():
                logger.warning(
                    f"{LOG_PREFIX} predict_scalar non-finite embedding for '{task_name}'"
                )
                return None

            logits = head.forward(embedding)  # (1, 1)
            logit = float(logits[0, 0])
            scalar = float(1.0 / (1.0 + np.exp(-logit)))

            # Guard the contract: regression heads must yield finite scalars in [0, 1].
            # NaN/Inf weights or NaN embeddings would otherwise propagate silently.
            if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
                logger.warning(
                    f"{LOG_PREFIX} predict_scalar out-of-contract output for '{task_name}': "
                    f"scalar={scalar!r} (must be finite in [0,1])"
                )
                return None

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"{LOG_PREFIX} {task_name}: scalar={scalar:.4f} in {elapsed_ms:.1f}ms"
            )
            return scalar

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} predict_scalar failed for '{task_name}': {e}")
            return None

    def predict(
        self,
        task_name: str,
        prompt: str,
        extra_features: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """Run single-label MLP classification.

        Args:
            task_name:      Registered task (e.g. 'mode_detector').
            prompt:         Raw text to encode.
            extra_features: Optional (1, extra_dim) float32 array to concatenate
                            after the embedding.

        Returns:
            (label, confidence) — label is None if the model isn't available
            or if the feature shape doesn't match meta.input_dim.

        Raises:
            ValueError if called on a regression task (use predict_scalar instead).
        """
        head = self._get_head(task_name)
        if head is None:
            return None, 0.0

        if head.task_type == "regression":
            raise ValueError(
                f"predict() called on regression task '{task_name}'. "
                f"Use predict_scalar() for regression heads."
            )

        try:
            start = time.perf_counter()

            embedding = self._embed(prompt)  # (1, 768)

            if extra_features is not None:
                features = np.concatenate([embedding, extra_features], axis=-1)
            else:
                features = embedding

            expected_input_dim = head.input_dim
            if features.shape[-1] != expected_input_dim:
                logger.warning(
                    f"{LOG_PREFIX} {task_name}: feature dim mismatch "
                    f"(got {features.shape[-1]}, expected {expected_input_dim})"
                )
                return None, 0.0

            if not np.isfinite(features).all():
                logger.warning(
                    f"{LOG_PREFIX} predict non-finite features for '{task_name}'"
                )
                return None, 0.0

            logits = head.forward(features)           # (1, num_classes)
            probs = _softmax(logits.astype(np.float32))[0]  # (num_classes,)

            if not np.isfinite(probs).all():
                logger.warning(
                    f"{LOG_PREFIX} predict non-finite probs for '{task_name}'"
                )
                return None, 0.0

            winner_idx = int(np.argmax(probs))
            confidence = float(probs[winner_idx])
            label = head.labels[winner_idx]

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"{LOG_PREFIX} {task_name}: {label} ({confidence:.3f}) in {elapsed_ms:.1f}ms"
            )
            return label, confidence

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Inference failed for '{task_name}': {e}")
            return None, 0.0

    def predict_all_sigmoids(self, task: str, text: str) -> Dict[str, float]:
        """Return sigmoid probabilities for every output head of a multi-label task.

        Returns a ``{label: probability}`` dict with NO threshold filtering.
        Intended for tasks whose ``classifier_meta.json`` sets
        ``task_type = "multi_label"`` and ``output_activation = "sigmoid"``.

        Args:
            task: Registered task name (e.g. ``'mode_detector'``).
            text: Raw user text to classify.

        Returns:
            Dict mapping each label to its sigmoid probability in [0, 1].
            Returns ``{}`` when the task is unavailable, the head cannot be
            loaded, feature dimensions do not match, or any other error occurs.
            Never raises.
        """
        try:
            head = self._get_head(task)
            if head is None:
                return {}

            embedding = self._embed(text)  # (1, 768)
            features = embedding

            if features.shape[-1] != head.input_dim:
                logger.warning(
                    f"{LOG_PREFIX} {task}: feature dim mismatch "
                    f"(got {features.shape[-1]}, expected {head.input_dim}) "
                    f"in predict_all_sigmoids"
                )
                return {}

            logits = head.forward(features)[0].astype(np.float64)  # (num_classes,)

            probs = 1.0 / (1.0 + np.exp(-logits))

            if not np.isfinite(probs).all():
                logger.warning(
                    f"{LOG_PREFIX} predict_all_sigmoids non-finite probs for '{task}'"
                )
                return {}

            return {label: float(probs[i]) for i, label in enumerate(head.labels) if i < len(probs)}

        except Exception as exc:
            logger.warning(f"{LOG_PREFIX} predict_all_sigmoids failed for '{task}': {exc}")
            return {}

    def is_available(self, model_name: str) -> bool:
        """Check if a model is loaded or loadable."""
        return self._get_head(model_name) is not None


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[OnnxInferenceService] = None
_instance_lock = threading.Lock()


def get_onnx_inference_service() -> OnnxInferenceService:
    """Get or create the singleton OnnxInferenceService."""
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance

        from services.file_mapper_service import FileMapperService

        models_dir = str(FileMapperService.get_models_path())
        pretrained_dir = str(FileMapperService.get_pretrained_path())

        _instance = OnnxInferenceService(models_dir, pretrained_dir)
        logger.info(
            f"{LOG_PREFIX} Initialized with models_dir={models_dir} "
            f"pretrained_dir={pretrained_dir}"
        )
        return _instance
