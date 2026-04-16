# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ONNX Inference Service — shared gte-modernbert encoder + swappable MLP heads.

Architecture (v0.9.0):
    1 shared ONNX encoder (gte-modernbert-base, already loaded by embedding_service)
    N 2-layer MLP heads loaded from per-task .npz files → class logits

The encoder session and tokenizer are borrowed from embedding_service's module-level
singletons so the 596 MB model is loaded exactly once per process. Each classifier
head is a tiny numpy MLP (~800 KB) loaded at task registration time.

Boot-time sha256 pin:
    Computed once (streaming) against backend/data/models/gte-modernbert-base/onnx/model.onnx.
    Must match classifier_meta.json::base_encoder_sha256 for every registered task.
    On mismatch: RuntimeError raised, task not registered.

Thread-safe — multiple workers can call predict() concurrently.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ONNX]"

# Default GitHub repo for model releases
DEFAULT_MODELS_REPO = "chalie-ai/models"

# v0.9.0 release tag — assets fetched from this specific tag
_RELEASE_TAG = "v0.9.0"

# Tasks to auto-download and register on boot.
# Each entry: (task_name, asset_name_prefix)
# - meta asset: <prefix>-classifier_meta.json → models/<task>/classifier_meta.json
# - head asset: <prefix>_head.npz             → models/<task>/<head_asset from meta>
# (note the mixed separator: hyphen before "classifier", underscore before "head")
MODEL_REGISTRY = [
    ("contradiction", "contradiction"),
    ("thinking_level", "thinking-level"),
]

# Asset filename constants
_CLASSIFIER_META_FILENAME = "classifier_meta.json"
_CLASSIFIER_META_STAGING_FILENAME = "classifier_meta_dl.json"

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
                 "input_dim", "hidden_dim", "num_classes")

    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        labels: List[str],
        activation: str = "gelu",
    ):
        self.W1 = W1.astype(np.float32)    # (hidden_dim, input_dim)
        self.b1 = b1.astype(np.float32)    # (hidden_dim,)
        self.W2 = W2.astype(np.float32)    # (num_classes, hidden_dim)
        self.b2 = b2.astype(np.float32)    # (num_classes,)
        self.labels = labels
        self.activation = activation
        self.hidden_dim, self.input_dim = W1.shape
        self.num_classes = W2.shape[0]

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Apply the 2-layer MLP.

        Args:
            features: (batch, input_dim) float32

        Returns:
            logits: (batch, num_classes) float32
        """
        # fc1: (batch, input_dim) @ (input_dim, hidden_dim) = (batch, hidden_dim)
        h = features @ self.W1.T + self.b1
        if self.activation == "gelu":
            h = _gelu(h)
        else:
            h = np.maximum(0.0, h)  # relu fallback
        # fc2: (batch, hidden_dim) @ (hidden_dim, num_classes) = (batch, num_classes)
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


def _download_with_curl(url: str, dest: Path) -> None:
    """Download a URL to dest, following redirects (GitHub 302s)."""
    result = subprocess.run(
        ["curl", "-sL", "--fail", "-o", str(dest), url],
        capture_output=True,
        timeout=600,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise OSError(f"curl failed (exit {result.returncode}): {stderr[:200]}")


class OnnxInferenceService:
    """
    Shared gte-modernbert encoder + swappable 2-layer MLP classifier heads.

    Usage:
        svc = OnnxInferenceService("/models")
        svc.ensure_models()
        label, confidence = svc.predict("contradiction", input_text)
    """

    def __init__(self, models_dir: str):
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Registered classifier heads (task_name → _ClassifierHead)
        self._heads: Dict[str, _ClassifierHead] = {}
        self._heads_lock = threading.Lock()

        # Boot readiness — set to True after registration attempt completes
        self._ready = False
        # Tasks whose boot-time registration failed (e.g. sha256 gate refused).
        # If non-empty, the service is in a degraded state — predict() returns
        # (None, 0.0) silently for those tasks. Health endpoint surfaces this.
        # List of (task_name, error_message) tuples.
        self._failed_registrations: list[tuple[str, str]] = []

    @property
    def ready(self) -> bool:
        """True only after registration attempt completed AND every task registered.

        A degraded service (one or more tasks failed sha256 gate) reports NOT
        ready so /api/system can fail-fast for monitoring. Use `degraded` to
        distinguish "still loading" from "loaded but some tasks refused".
        """
        return self._ready and not self._failed_registrations

    @property
    def degraded(self) -> bool:
        """True if registration finished but at least one task failed.

        Inspect `failed_registrations` for the per-task error messages.
        """
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

    # ── Download / update ─────────────────────────────────────────────────────

    def ensure_models(self):
        """Download missing or stale model assets from the v0.9.0 GitHub release."""
        for task_name, asset_prefix in MODEL_REGISTRY:
            try:
                self._ensure_task(task_name, asset_prefix)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to ensure {task_name}: {e}")

    def _ensure_task(self, task_name: str, asset_prefix: str):
        """Download classifier_meta.json and head .npz for one task if missing/stale."""
        task_dir = self._models_dir / task_name
        meta_path = task_dir / _CLASSIFIER_META_FILENAME

        # Check local version
        local_version = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    local_version = json.load(f).get("version")
            except (json.JSONDecodeError, OSError):
                pass

        if local_version == _RELEASE_TAG:
            logger.info(f"{LOG_PREFIX} {task_name}: up to date ({_RELEASE_TAG})")
            return

        action = "Updating" if local_version else "Downloading"
        logger.info(f"{LOG_PREFIX} {action} {task_name}: {local_version or '(none)'} → {_RELEASE_TAG}")

        base_url = f"https://github.com/{DEFAULT_MODELS_REPO}/releases/download/{_RELEASE_TAG}"
        # Asset naming in v0.9.0: `<prefix>-classifier_meta.json` and `<prefix>_head.npz`
        # (hyphen before "classifier", underscore before "head" — confirmed against
        # the release manifest; do NOT collapse these separators).
        meta_url = f"{base_url}/{asset_prefix}-classifier_meta.json"
        head_url = f"{base_url}/{asset_prefix}_head.npz"

        staging = self._models_dir / f".{task_name}_installing"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)

            # Download meta JSON
            _download_with_curl(meta_url, staging / _CLASSIFIER_META_STAGING_FILENAME)
            with open(staging / _CLASSIFIER_META_STAGING_FILENAME) as f:
                meta = json.load(f)
            meta["version"] = _RELEASE_TAG
            with open(staging / _CLASSIFIER_META_FILENAME, "w") as f:
                json.dump(meta, f, indent=2)
            (staging / _CLASSIFIER_META_STAGING_FILENAME).unlink()

            # Download head .npz
            head_asset = meta.get("head_asset", f"{asset_prefix}_head.npz")
            logger.info(f"{LOG_PREFIX} Downloading head for {task_name}...")
            _download_with_curl(head_url, staging / head_asset)
            size_kb = (staging / head_asset).stat().st_size / 1024
            logger.info(f"{LOG_PREFIX} Head downloaded ({size_kb:.1f}KB)")

            # Atomic swap
            if task_dir.exists():
                shutil.rmtree(task_dir)
            staging.rename(task_dir)

            # Evict cached head so next call re-registers from fresh files
            with self._heads_lock:
                self._heads.pop(task_name, None)

            logger.info(f"{LOG_PREFIX} Installed {task_name} ({_RELEASE_TAG})")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Download failed for {task_name}: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    # ── Task registration ─────────────────────────────────────────────────────

    def _register_task(self, task_name: str) -> Optional[_ClassifierHead]:
        """Load meta + head for one task, perform sha256 pin check, emit boot marker.

        Returns the head on success. Raises RuntimeError on sha256 mismatch.
        Returns None if the task directory or meta is missing.
        """
        task_dir = self._models_dir / task_name
        meta_path = task_dir / _CLASSIFIER_META_FILENAME

        if not meta_path.exists():
            logger.warning(f"{LOG_PREFIX} Missing {_CLASSIFIER_META_FILENAME} for task '{task_name}'")
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"{LOG_PREFIX} Cannot read meta for '{task_name}': {e}")
            return None

        # sha256 pin check — computed once per process, not per task
        expected_sha = meta.get("base_encoder_sha256", "").lower()
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

        # Load head .npz
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
                W1 = f["W1"].astype(np.float32)
                b1 = f["b1"].astype(np.float32)
                W2 = f["W2"].astype(np.float32)
                b2 = f["b2"].astype(np.float32)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load head for '{task_name}': {e}")
            return None

        labels = meta.get("labels", [])
        activation = meta.get("activation", meta.get("mlp_activation", "gelu"))
        input_dim = meta.get("input_dim", W1.shape[1])
        hidden_dim = meta.get("hidden_dim", W1.shape[0])
        num_classes = meta.get("num_classes", W2.shape[0])

        head = _ClassifierHead(W1=W1, b1=b1, W2=W2, b2=b2,
                               labels=labels, activation=activation)

        # Emit boot marker — format is contractual (scenarios grep exact prefix)
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

    def predict(
        self,
        task_name: str,
        prompt: str,
        extra_features: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """Run single-label MLP classification.

        Args:
            task_name:      Registered task ('thinking_level', 'contradiction', …).
            prompt:         Raw text to encode.
            extra_features: Optional (1, extra_dim) float32 array to concatenate
                            after the embedding (e.g. prev_level one-hot).

        Returns:
            (label, confidence) — label is None if the model isn't available
            or if the feature shape doesn't match meta.input_dim.
        """
        head = self._get_head(task_name)
        if head is None:
            return None, 0.0

        try:
            start = time.perf_counter()

            embedding = self._embed(prompt)  # (1, 768)

            if extra_features is not None:
                features = np.concatenate([embedding, extra_features], axis=-1)  # (1, input_dim)
            else:
                features = embedding  # (1, 768)

            expected_input_dim = head.input_dim
            if features.shape[-1] != expected_input_dim:
                logger.warning(
                    f"{LOG_PREFIX} {task_name}: feature dim mismatch "
                    f"(got {features.shape[-1]}, expected {expected_input_dim})"
                )
                return None, 0.0

            logits = head.forward(features)           # (1, num_classes)
            probs = _softmax(logits.astype(np.float32))[0]  # (num_classes,)

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

    def predict_multi_label(
        self,
        model_name: str,
        text: str,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Multi-label classification via sigmoid per output, threshold per label.

        Used by ConsequenceClassifierService. Returns list of (label, score) tuples
        above threshold, sorted descending by score.
        """
        head = self._get_head(model_name)
        if head is None:
            return []

        try:
            embedding = self._embed(text)  # (1, 768)
            features = embedding

            if features.shape[-1] != head.input_dim:
                logger.warning(
                    f"{LOG_PREFIX} {model_name}: feature dim mismatch for multi_label"
                )
                return []

            logits = head.forward(features)[0].astype(np.float64)  # (num_classes,)

            # Sigmoid per output
            probs = 1.0 / (1.0 + np.exp(-logits))

            default_threshold = 0.5
            results = []
            for i, label in enumerate(head.labels):
                if i >= len(probs):
                    break
                t = (threshold_overrides or {}).get(label, default_threshold)
                if probs[i] >= t:
                    results.append((label, float(probs[i])))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Multi-label inference failed for '{model_name}': {e}")
            return []

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

        import runtime_config

        _default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
        models_dir = runtime_config.get(
            "models_dir",
            os.environ.get("MODELS_DIR", _default),
        )

        _instance = OnnxInferenceService(models_dir)
        logger.info(f"{LOG_PREFIX} Initialized with models_dir={models_dir}")
        return _instance
