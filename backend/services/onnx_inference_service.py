# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ONNX Inference Service — shared-base + swappable-head classifier inference.

Architecture:
    1 shared ONNX base model (Qwen2.5-0.5B transformer) → last hidden state
    N tiny classifier heads (.npz numpy weights) → class logits

The base model (~473MB) is downloaded once and loaded into a single ONNX
session. Each classifier head is a linear projection (~7-50KB) loaded as
numpy weight matrices. Inference runs the shared base, then applies the
appropriate head via numpy matmul.

Supports two classification modes:
  - ``single_label``: N-class softmax classification
  - ``multi_label``: K independent sigmoid outputs with per-label thresholds

Falls back to legacy monolithic ONNX models if ``split: true`` is absent
from the classifier metadata (backward compatible with v0.3.0 releases).

Thread-safe — multiple workers can call predict() concurrently.
"""

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ONNX]"

# Default GitHub repo for model releases
DEFAULT_MODELS_REPO = "chalie-ai/models"

# Models that should be auto-downloaded on boot.
# Each entry: (subdirectory_name, github_repo_or_None_for_default, release_asset_prefix)
MODEL_REGISTRY = [
    ("mode-tiebreaker", None, "mode-tiebreaker"),
    ("contradiction", None, "contradiction"),
    ("skill-selector", None, "skill-selector"),
    ("trait-detector", None, "trait-detector"),
]

# Shared base model config
BASE_MODEL_NAME = "qwen2.5-0.5b_base"
BASE_MODEL_HF = "Qwen/Qwen2.5-0.5B"


class _ClassifierHead:
    """A tiny classifier head: numpy weight matrix + optional bias."""

    __slots__ = ("weight", "bias", "labels", "model_type", "thresholds",
                 "pruned", "version")

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray],
                 labels: List[str], model_type: str = "single_label",
                 thresholds: Optional[Dict[str, float]] = None,
                 pruned: bool = True, version: str = "unknown"):
        self.weight = weight          # (num_classes, hidden_dim)
        self.bias = bias              # (num_classes,) or None
        self.labels = labels
        self.model_type = model_type
        self.thresholds = thresholds or {}
        self.pruned = pruned
        self.version = version

    def forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Apply linear projection: (batch, hidden_dim) → (batch, num_classes)."""
        logits = hidden_state @ self.weight.T
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class _LegacyModel:
    """Holds a monolithic ONNX session for backward compatibility with v0.3.0."""

    __slots__ = ("session", "tokenizer", "labels", "label_token_ids", "version",
                 "pruned", "model_type", "thresholds", "_extra_inputs")

    def __init__(self, session, tokenizer, labels: List[str],
                 label_token_ids: List[int], version: str,
                 pruned: bool = False, model_type: str = "single_label",
                 thresholds: Optional[Dict[str, float]] = None):
        self.session = session
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = label_token_ids
        self.version = version
        self.pruned = pruned
        self.model_type = model_type
        self.thresholds = thresholds or {}
        known = {"input_ids", "attention_mask"}
        self._extra_inputs = [
            inp for inp in session.get_inputs() if inp.name not in known
        ]

    def build_feed(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> dict:
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        for inp in self._extra_inputs:
            shape = [s if isinstance(s, int) else input_ids.shape[0]
                     for s in inp.shape]
            dtype = np.float32 if "float" in inp.type else np.int64
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        return feed


# Shared tokenizer cache: base_model_name → tokenizer instance
_tokenizer_cache: Dict[str, object] = {}
_tokenizer_lock = threading.Lock()


def _get_shared_tokenizer(base_model: str, model_dir: Optional[Path] = None):
    """Load tokenizer from HuggingFace, sharing across models with same base."""
    if base_model in _tokenizer_cache:
        return _tokenizer_cache[base_model]

    with _tokenizer_lock:
        if base_model in _tokenizer_cache:
            return _tokenizer_cache[base_model]

        from transformers import AutoTokenizer

        if model_dir and (model_dir / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            logger.info(f"{LOG_PREFIX} Loaded tokenizer from {model_dir}")
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            logger.info(f"{LOG_PREFIX} Loaded tokenizer from HuggingFace: {base_model}")

        _tokenizer_cache[base_model] = tokenizer
        return tokenizer


class OnnxInferenceService:
    """
    Shared-base ONNX inference with swappable classifier heads.

    Usage:
        svc = OnnxInferenceService("/models")
        svc.ensure_models()
        label, confidence = svc.predict("mode-tiebreaker", input_text)
        skills = svc.predict_multi_label("skill-selector", input_text)
    """

    def __init__(self, models_dir: str):
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Shared base model (ONNX session + tokenizer)
        self._base_session = None
        self._base_extra_inputs = []
        self._base_tokenizer = None
        self._base_lock = threading.Lock()

        # Classifier heads (model_name → _ClassifierHead or _LegacyModel)
        self._heads: Dict[str, object] = {}
        self._heads_lock = threading.Lock()

        # Boot readiness — set to True after ensure_models() + warmup complete
        self._ready = False

    @property
    def ready(self) -> bool:
        """True after ensure_models() + warmup inference have completed."""
        return self._ready

    # ── Download & Version Check ──────────────────────────────

    def ensure_models(self):
        """Download missing models and update stale ones from GitHub releases."""
        # First ensure the shared base model
        try:
            self._ensure_base_model()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to ensure base model: {e}")

        # Then ensure each classifier head
        for model_name, repo, asset_prefix in MODEL_REGISTRY:
            try:
                self._ensure_head(model_name, repo or DEFAULT_MODELS_REPO, asset_prefix)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to ensure {model_name}: {e}")

    def _ensure_base_model(self):
        """Download the shared base ONNX model if missing or outdated."""
        base_dir = self._models_dir / BASE_MODEL_NAME
        onnx_path = base_dir / "model.onnx"

        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"{LOG_PREFIX} Base model present: {onnx_path} ({size_mb:.0f}MB)")
            return

        # Download from release — look for qwen2.5-0.5b_base*.onnx asset
        api_url = f"https://api.github.com/repos/{DEFAULT_MODELS_REPO}/releases/latest"
        req = Request(api_url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Chalie/1.0",
        })

        try:
            with urlopen(req, timeout=5) as resp:
                release = json.loads(resp.read())
        except (URLError, OSError) as e:
            logger.warning(f"{LOG_PREFIX} Cannot fetch release for base model: {e}")
            return

        assets = release.get("assets", [])
        base_url = None
        for asset in assets:
            name = asset.get("name", "")
            if name.startswith("qwen2.5-0.5b_base") and name.endswith(".onnx"):
                if "quantized" in name:
                    base_url = asset["browser_download_url"]
                    break
                elif base_url is None:
                    base_url = asset["browser_download_url"]

        if not base_url:
            logger.warning(f"{LOG_PREFIX} No base model asset found in release")
            return

        staging = self._models_dir / f".{BASE_MODEL_NAME}_installing"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)

            logger.info(f"{LOG_PREFIX} Downloading shared base model...")
            req = Request(base_url, headers={"User-Agent": "Chalie/1.0"})
            with urlopen(req, timeout=600) as resp:
                (staging / "model.onnx").write_bytes(resp.read())

            size_mb = (staging / "model.onnx").stat().st_size / (1024 * 1024)
            logger.info(f"{LOG_PREFIX} Base model downloaded ({size_mb:.0f}MB)")

            # Save version info
            tag = release.get("tag_name", "unknown")
            with open(staging / "version.json", "w") as f:
                json.dump({"version": tag, "base_model": BASE_MODEL_HF}, f)

            if base_dir.exists():
                shutil.rmtree(base_dir)
            staging.rename(base_dir)

            # Invalidate cached session
            with self._base_lock:
                self._base_session = None

            logger.info(f"{LOG_PREFIX} Installed base model ({tag})")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Base model download failed: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _ensure_head(self, model_name: str, repo: str, asset_prefix: str):
        """Download or update a classifier head from GitHub release assets."""
        model_dir = self._models_dir / model_name
        meta_path = model_dir / "classifier_meta.json"

        local_version = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    local_version = json.load(f).get("version")
            except (json.JSONDecodeError, OSError):
                pass

        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        req = Request(api_url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Chalie/1.0",
        })
        try:
            with urlopen(req, timeout=5) as resp:
                release = json.loads(resp.read())
        except (URLError, OSError) as e:
            if local_version:
                logger.info(f"{LOG_PREFIX} {model_name}: network unavailable, keeping {local_version}")
            else:
                logger.warning(f"{LOG_PREFIX} {model_name}: no local model, network unavailable ({e})")
            return

        remote_tag = release.get("tag_name")
        if not remote_tag:
            return

        if local_version and local_version == remote_tag:
            logger.info(f"{LOG_PREFIX} {model_name}: up to date ({local_version})")
            return

        assets = release.get("assets", [])
        norm_prefix = asset_prefix.replace("-", "_")

        meta_url = None
        head_url = None
        onnx_url = None
        onnx_full_url = None

        for asset in assets:
            name = asset.get("name", "")
            norm_name = name.replace("-", "_")
            url = asset.get("browser_download_url")

            if norm_name == f"{norm_prefix}.json":
                meta_url = url
            elif norm_name == f"{norm_prefix}_head.npz":
                head_url = url
            elif norm_name.startswith(norm_prefix) and name.endswith(".onnx"):
                if "quantized" in name:
                    onnx_url = url
                else:
                    onnx_full_url = url

        onnx_url = onnx_url or onnx_full_url

        if not meta_url:
            logger.warning(f"{LOG_PREFIX} {model_name}: no meta JSON in release {remote_tag}")
            return

        # Need either head .npz (split format) or full .onnx (legacy)
        if not head_url and not onnx_url:
            logger.warning(f"{LOG_PREFIX} {model_name}: no head or ONNX asset in release {remote_tag}")
            return

        action = "Updating" if local_version else "Downloading"
        logger.info(f"{LOG_PREFIX} {action} {model_name}: {local_version or '(none)'} → {remote_tag}")

        staging = self._models_dir / f".{model_name}_installing"
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)

            # Download meta JSON
            req = Request(meta_url, headers={"User-Agent": "Chalie/1.0"})
            with urlopen(req, timeout=10) as resp:
                raw_meta = resp.read()
            meta = json.loads(raw_meta)
            meta["version"] = remote_tag
            meta.setdefault("repo", repo)
            with open(staging / "classifier_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            is_split = meta.get("split", False)

            if is_split and head_url:
                # Download tiny head .npz
                logger.info(f"{LOG_PREFIX} Downloading head for {model_name}...")
                req = Request(head_url, headers={"User-Agent": "Chalie/1.0"})
                with urlopen(req, timeout=30) as resp:
                    (staging / "head.npz").write_bytes(resp.read())
                size_kb = (staging / "head.npz").stat().st_size / 1024
                logger.info(f"{LOG_PREFIX} Head downloaded ({size_kb:.1f}KB)")
            elif onnx_url:
                # Legacy: download full ONNX
                logger.info(f"{LOG_PREFIX} Downloading ONNX weights for {model_name}...")
                req = Request(onnx_url, headers={"User-Agent": "Chalie/1.0"})
                with urlopen(req, timeout=300) as resp:
                    (staging / "model.onnx").write_bytes(resp.read())
                size_mb = (staging / "model.onnx").stat().st_size / (1024 * 1024)
                logger.info(f"{LOG_PREFIX} ONNX downloaded ({size_mb:.0f}MB)")

            # Atomic swap
            if model_dir.exists():
                shutil.rmtree(model_dir)
            staging.rename(model_dir)

            with self._heads_lock:
                self._heads.pop(model_name, None)

            logger.info(f"{LOG_PREFIX} Installed {model_name} ({remote_tag})")

        except (URLError, OSError) as e:
            logger.warning(f"{LOG_PREFIX} Download failed for {model_name}: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Unexpected error installing {model_name}: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    # ── Base Model Loading ────────────────────────────────────

    def _get_base_session(self):
        """Lazy-load the shared ONNX base session."""
        if self._base_session is not None:
            return self._base_session

        with self._base_lock:
            if self._base_session is not None:
                return self._base_session

            base_dir = self._models_dir / BASE_MODEL_NAME
            onnx_path = base_dir / "model.onnx"

            if not onnx_path.exists():
                logger.warning(f"{LOG_PREFIX} Shared base model not found: {onnx_path}")
                return None

            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    str(onnx_path), sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )

                self._base_session = session

                # Cache extra inputs
                known = {"input_ids", "attention_mask"}
                self._base_extra_inputs = [
                    inp for inp in session.get_inputs() if inp.name not in known
                ]

                # Load tokenizer
                self._base_tokenizer = _get_shared_tokenizer(BASE_MODEL_HF)

                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                logger.info(f"{LOG_PREFIX} Loaded shared base model ({size_mb:.0f}MB)")
                return session

            except ImportError:
                logger.warning(f"{LOG_PREFIX} onnxruntime not installed")
                return None
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to load base model: {e}")
                return None

    def _run_base(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Optional[np.ndarray]:
        """Run the shared base model → last hidden state (batch, hidden_dim)."""
        session = self._get_base_session()
        if session is None:
            return None

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        for inp in self._base_extra_inputs:
            shape = [s if isinstance(s, int) else input_ids.shape[0]
                     for s in inp.shape]
            dtype = np.float32 if "float" in inp.type else np.int64
            feed[inp.name] = np.zeros(shape, dtype=dtype)

        outputs = session.run(None, feed)
        return outputs[0]  # (batch, hidden_dim)

    # ── Head Loading ──────────────────────────────────────────

    def _get_head(self, model_name: str):
        """Lazy-load a classifier head. Returns _ClassifierHead or _LegacyModel."""
        if model_name in self._heads:
            return self._heads[model_name]

        with self._heads_lock:
            if model_name in self._heads:
                return self._heads[model_name]

            head = self._load_head(model_name)
            self._heads[model_name] = head
            return head

    def _load_head(self, model_name: str):
        """Load a classifier head from disk."""
        model_dir = self._models_dir / model_name

        if not model_dir.is_dir():
            logger.warning(f"{LOG_PREFIX} Model directory not found: {model_dir}")
            return None

        meta_path = model_dir / "classifier_meta.json"
        if not meta_path.exists():
            logger.warning(f"{LOG_PREFIX} Missing classifier_meta.json in {model_dir}")
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            labels = meta["labels"]
            version = meta.get("version", "unknown")
            model_type = meta.get("model_type", "single_label")
            thresholds = meta.get("thresholds", {})
            is_split = meta.get("split", False)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"{LOG_PREFIX} Invalid classifier_meta.json: {e}")
            return None

        if is_split:
            return self._load_split_head(
                model_name, model_dir, labels, version, model_type, thresholds,
            )
        else:
            return self._load_legacy_model(
                model_name, model_dir, meta, labels, version, model_type, thresholds,
            )

    def _load_split_head(self, model_name, model_dir, labels, version,
                         model_type, thresholds):
        """Load a split-format head (.npz weights)."""
        npz_path = model_dir / "head.npz"
        if not npz_path.exists():
            logger.warning(f"{LOG_PREFIX} Missing head.npz in {model_dir}")
            return None

        try:
            data = np.load(str(npz_path))
            weight = data["weight"]
            bias = data.get("bias")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load head.npz: {e}")
            return None

        size_kb = npz_path.stat().st_size / 1024
        logger.info(
            f"{LOG_PREFIX} Loaded head {model_name} ({version}): "
            f"{weight.shape}, type={model_type}, {size_kb:.1f}KB"
        )

        return _ClassifierHead(
            weight=weight, bias=bias, labels=labels,
            model_type=model_type, thresholds=thresholds,
            version=version,
        )

    def _load_legacy_model(self, model_name, model_dir, meta, labels,
                           version, model_type, thresholds):
        """Load a monolithic ONNX model (backward compat with v0.3.0)."""
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            logger.warning(f"{LOG_PREFIX} No .onnx file in {model_dir}")
            return None

        onnx_path = onnx_files[0]
        pruned = meta.get("pruned", False)
        base_model = meta.get("base_model")

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(onnx_path), sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except ImportError:
            logger.warning(f"{LOG_PREFIX} onnxruntime not installed")
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load ONNX: {e}")
            return None

        try:
            if base_model:
                tokenizer = _get_shared_tokenizer(base_model, model_dir)
            elif (model_dir / "tokenizer.json").exists():
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            else:
                logger.warning(f"{LOG_PREFIX} No tokenizer for {model_name}")
                return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Tokenizer failed for {model_name}: {e}")
            return None

        label_token_ids = []
        if not pruned:
            for label in labels:
                token_ids = tokenizer.encode(label, add_special_tokens=False)
                if not token_ids:
                    return None
                label_token_ids.append(token_ids[0])

        logger.info(
            f"{LOG_PREFIX} Loaded legacy {model_name} ({version}): "
            f"{onnx_path.name}, type={model_type}, pruned={pruned}"
        )

        return _LegacyModel(
            session=session, tokenizer=tokenizer, labels=labels,
            label_token_ids=label_token_ids, version=version,
            pruned=pruned, model_type=model_type, thresholds=thresholds,
        )

    # ── Public API ────────────────────────────────────────────

    def predict(self, model_name: str, text: str) -> Tuple[Optional[str], float]:
        """
        Run single-label classification (softmax → argmax).

        Returns:
            (label, confidence) — label is None if the model isn't available.
        """
        head = self._get_head(model_name)
        if head is None:
            return None, 0.0

        if isinstance(head, _LegacyModel):
            return self._predict_legacy(head, model_name, text)

        return self._predict_split(head, model_name, text)

    def predict_multi_label(
        self, model_name: str, text: str,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Run multi-label classification (sigmoid per output, threshold per label).

        Returns:
            List of (label, confidence) tuples above threshold, sorted descending.
        """
        head = self._get_head(model_name)
        if head is None:
            return []

        if isinstance(head, _LegacyModel):
            return self._predict_multi_label_legacy(head, model_name, text, threshold_overrides)

        return self._predict_multi_label_split(head, model_name, text, threshold_overrides)

    def predict_batch(self, model_name: str, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        """Run single-label classification on a batch of inputs."""
        head = self._get_head(model_name)
        if head is None:
            return [(None, 0.0)] * len(texts)

        if isinstance(head, _LegacyModel):
            return self._predict_batch_legacy(head, model_name, texts)

        return self._predict_batch_split(head, model_name, texts)

    def is_available(self, model_name: str) -> bool:
        """Check if a model is loaded or loadable."""
        return self._get_head(model_name) is not None

    # ── Split-format inference ────────────────────────────────

    def _predict_split(self, head: _ClassifierHead, model_name: str,
                       text: str) -> Tuple[Optional[str], float]:
        try:
            start = time.perf_counter()

            tokenizer = self._base_tokenizer or _get_shared_tokenizer(BASE_MODEL_HF)
            encoded = tokenizer(
                text, return_tensors="np", padding=False,
                truncation=True, max_length=256,
            )

            hidden = self._run_base(encoded["input_ids"], encoded["attention_mask"])
            if hidden is None:
                return None, 0.0

            logits = head.forward(hidden)[0]  # (num_classes,)

            # Softmax
            shifted = logits - logits.max()
            exp_l = np.exp(shifted)
            probs = exp_l / exp_l.sum()

            winner_idx = int(np.argmax(probs))
            confidence = float(probs[winner_idx])
            label = head.labels[winner_idx]

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{LOG_PREFIX} {model_name}: {label} ({confidence:.3f}) in {elapsed_ms:.1f}ms")
            return label, confidence

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Inference failed for {model_name}: {e}")
            return None, 0.0

    def _predict_multi_label_split(
        self, head: _ClassifierHead, model_name: str, text: str,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        try:
            start = time.perf_counter()

            tokenizer = self._base_tokenizer or _get_shared_tokenizer(BASE_MODEL_HF)
            encoded = tokenizer(
                text, return_tensors="np", padding=False,
                truncation=True, max_length=256,
            )

            hidden = self._run_base(encoded["input_ids"], encoded["attention_mask"])
            if hidden is None:
                return []

            logits = head.forward(hidden)[0]  # (num_classes,)

            # Sigmoid per output
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))

            thresholds = threshold_overrides or head.thresholds
            results = []
            for i, label in enumerate(head.labels):
                if i >= len(probs):
                    break
                t = thresholds.get(label, 0.5)
                if probs[i] >= t:
                    results.append((label, float(probs[i])))

            results.sort(key=lambda x: x[1], reverse=True)

            elapsed_ms = (time.perf_counter() - start) * 1000
            active = [r[0] for r in results]
            logger.debug(f"{LOG_PREFIX} {model_name}: {active} in {elapsed_ms:.1f}ms")
            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Multi-label inference failed for {model_name}: {e}")
            return []

    def _predict_batch_split(self, head: _ClassifierHead, model_name: str,
                             texts: List[str]) -> List[Tuple[Optional[str], float]]:
        try:
            tokenizer = self._base_tokenizer or _get_shared_tokenizer(BASE_MODEL_HF)
            encoded = tokenizer(
                texts, return_tensors="np", padding=True,
                truncation=True, max_length=256,
            )

            hidden = self._run_base(encoded["input_ids"], encoded["attention_mask"])
            if hidden is None:
                return [(None, 0.0)] * len(texts)

            all_logits = head.forward(hidden)  # (batch, num_classes)

            results = []
            for i in range(len(texts)):
                logits = all_logits[i]
                shifted = logits - logits.max()
                exp_l = np.exp(shifted)
                probs = exp_l / exp_l.sum()
                winner_idx = int(np.argmax(probs))
                results.append((head.labels[winner_idx], float(probs[winner_idx])))

            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Batch inference failed for {model_name}: {e}")
            return [(None, 0.0)] * len(texts)

    # ── Legacy inference (backward compat) ────────────────────

    def _predict_legacy(self, model: _LegacyModel, model_name: str,
                        text: str) -> Tuple[Optional[str], float]:
        try:
            start = time.perf_counter()

            encoded = model.tokenizer(
                text, return_tensors="np", padding=False,
                truncation=True, max_length=256,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model.session.run(None, model.build_feed(input_ids, attention_mask))
            logits = outputs[0]

            if model.pruned:
                label_logits = logits[0]
            else:
                seq_len = int(attention_mask.sum()) - 1
                last_logits = logits[0, seq_len, :]
                vocab_size = len(last_logits)
                safe_ids = [tid for tid in model.label_token_ids if tid < vocab_size]
                label_logits = np.array([last_logits[tid] for tid in safe_ids])

            shifted = label_logits - label_logits.max()
            exp_l = np.exp(shifted)
            probs = exp_l / exp_l.sum()

            winner_idx = int(np.argmax(probs))
            confidence = float(probs[winner_idx])
            label = model.labels[winner_idx]

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{LOG_PREFIX} {model_name} (legacy): {label} ({confidence:.3f}) in {elapsed_ms:.1f}ms")
            return label, confidence

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Legacy inference failed for {model_name}: {e}")
            return None, 0.0

    def _predict_multi_label_legacy(
        self, model: _LegacyModel, model_name: str, text: str,
        threshold_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        try:
            start = time.perf_counter()

            encoded = model.tokenizer(
                text, return_tensors="np", padding=False,
                truncation=True, max_length=256,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model.session.run(None, model.build_feed(input_ids, attention_mask))
            logits = outputs[0]

            if model.pruned:
                raw_logits = logits[0]
            else:
                seq_len = int(attention_mask.sum()) - 1
                raw_logits = logits[0, seq_len, :]

            probs = 1.0 / (1.0 + np.exp(-raw_logits.astype(np.float64)))

            thresholds = threshold_overrides or model.thresholds
            results = []
            for i, label in enumerate(model.labels):
                if i >= len(probs):
                    break
                t = thresholds.get(label, 0.5)
                if probs[i] >= t:
                    results.append((label, float(probs[i])))

            results.sort(key=lambda x: x[1], reverse=True)

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"{LOG_PREFIX} {model_name} (legacy): {[r[0] for r in results]} in {elapsed_ms:.1f}ms")
            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Legacy multi-label failed for {model_name}: {e}")
            return []

    def _predict_batch_legacy(self, model: _LegacyModel, model_name: str,
                              texts: List[str]) -> List[Tuple[Optional[str], float]]:
        try:
            encoded = model.tokenizer(
                texts, return_tensors="np", padding=True,
                truncation=True, max_length=256,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model.session.run(None, model.build_feed(input_ids, attention_mask))
            logits = outputs[0]

            results = []
            for i in range(len(texts)):
                if model.pruned:
                    label_logits = logits[i]
                else:
                    seq_len = int(attention_mask[i].sum()) - 1
                    last_logits = logits[i, seq_len, :]
                    label_logits = np.array([last_logits[tid] for tid in model.label_token_ids])

                shifted = label_logits - label_logits.max()
                exp_l = np.exp(shifted)
                probs = exp_l / exp_l.sum()
                winner_idx = int(np.argmax(probs))
                results.append((model.labels[winner_idx], float(probs[winner_idx])))

            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Legacy batch failed for {model_name}: {e}")
            return [(None, 0.0)] * len(texts)


# ── Singleton ─────────────────────────────────────────────────

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