# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ONNX Inference Service — generic single-token classifier inference.

Loads ONNX models from a configurable directory and runs classification
inference on CPU.  Model-agnostic: each subdirectory holds a model's
ONNX weights and metadata.  Tokenizer is loaded from HuggingFace via
the ``base_model`` field in ``classifier_meta.json``.

On first boot (or version mismatch), models are downloaded from their
GitHub release assets (ONNX weights + meta JSON).

Directory layout (after download):
    <MODELS_DIR>/
        mode-tiebreaker/
            model.onnx
            classifier_meta.json   {"labels": ["A","B"], "base_model": "...", ...}
        contradiction/
            ...

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

# Default GitHub repo for model releases (override per-model in classifier_meta.json)
DEFAULT_MODELS_REPO = "chalie-ai/models"

# Models that should be auto-downloaded on boot.
# Each entry: (subdirectory_name, github_repo_or_None_for_default, release_asset_prefix)
# The asset is expected as a tarball: {prefix}.tar.gz in the latest release.
MODEL_REGISTRY = [
    ("mode-tiebreaker", None, "mode-tiebreaker"),
]


class _CachedModel:
    """Holds a loaded ONNX session, tokenizer, and label metadata."""

    __slots__ = ("session", "tokenizer", "labels", "label_token_ids", "version",
                 "_extra_inputs")

    def __init__(self, session, tokenizer, labels: List[str],
                 label_token_ids: List[int], version: str):
        self.session = session
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = label_token_ids
        self.version = version
        # Cache extra ONNX inputs (e.g. RoPE internals traced as graph inputs).
        # These need zero tensors at inference time.
        known = {"input_ids", "attention_mask"}
        self._extra_inputs = [
            inp for inp in session.get_inputs() if inp.name not in known
        ]

    def build_feed(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> dict:
        """Build complete ONNX input feed including extra traced inputs."""
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        for inp in self._extra_inputs:
            shape = [s if isinstance(s, int) else input_ids.shape[0]
                     for s in inp.shape]
            dtype = np.float32 if "float" in inp.type else np.int64
            feed[inp.name] = np.zeros(shape, dtype=dtype)
        return feed


class OnnxInferenceService:
    """
    Generic ONNX classifier inference with auto-download.

    Usage:
        svc = OnnxInferenceService("/models")
        svc.ensure_models()                       # download / version-check
        label, confidence = svc.predict("mode-tiebreaker", input_text)
    """

    def __init__(self, models_dir: str):
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Optional[_CachedModel]] = {}
        self._lock = threading.Lock()

    # ── Download & Version Check ──────────────────────────────

    def ensure_models(self):
        """Download missing models and update stale ones from GitHub releases.

        Safe to call from a background thread — uses stdlib only, short timeouts,
        atomic installs.  Skips gracefully on network failure.
        """
        for model_name, repo, asset_prefix in MODEL_REGISTRY:
            try:
                self._ensure_model(model_name, repo or DEFAULT_MODELS_REPO, asset_prefix)
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Failed to ensure {model_name}: {e}")

    def _ensure_model(self, model_name: str, repo: str, asset_prefix: str):
        """Download or update a single model from GitHub release assets.

        Release convention:
          - ``{prefix}.json``  — classifier_meta.json (labels, base_model, etc.)
          - ``{prefix}_quantized.onnx`` or ``{prefix}.onnx`` — ONNX weights

        Tokenizer is loaded from HuggingFace via the ``base_model`` field in the
        meta JSON, so no tokenizer files need to be shipped in the release.
        """
        model_dir = self._models_dir / model_name
        meta_path = model_dir / "classifier_meta.json"

        # Read local version (if installed)
        local_version = None
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    local_version = json.load(f).get("version")
            except (json.JSONDecodeError, OSError):
                pass

        # Fetch latest release tag from GitHub (5s timeout — fail fast)
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
                logger.info(
                    f"{LOG_PREFIX} {model_name}: network unavailable, "
                    f"keeping local {local_version}"
                )
            else:
                logger.warning(
                    f"{LOG_PREFIX} {model_name}: no local model and network "
                    f"unavailable ({e}) — classifier will use LLM fallback"
                )
            return

        remote_tag = release.get("tag_name")
        if not remote_tag:
            logger.warning(f"{LOG_PREFIX} {model_name}: no release tag found in {repo}")
            return

        # Skip if already up to date
        if local_version and local_version == remote_tag:
            logger.info(f"{LOG_PREFIX} {model_name}: up to date ({local_version})")
            return

        # Resolve asset URLs from the release
        assets = release.get("assets", [])
        norm_prefix = asset_prefix.replace("-", "_")

        meta_url = None
        onnx_url = None
        onnx_full_url = None
        for asset in assets:
            name = asset.get("name", "")
            norm_name = name.replace("-", "_")
            url = asset.get("browser_download_url")

            # Meta JSON: {prefix}.json
            if norm_name == f"{norm_prefix}.json":
                meta_url = url
            # ONNX: prefer quantized over full precision
            elif norm_name.startswith(norm_prefix) and name.endswith(".onnx"):
                if "quantized" in name:
                    onnx_url = url
                else:
                    onnx_full_url = url

        onnx_url = onnx_url or onnx_full_url

        if not onnx_url:
            logger.warning(
                f"{LOG_PREFIX} {model_name}: no ONNX asset matching "
                f"'{norm_prefix}*.onnx' in release {remote_tag}"
            )
            return
        if not meta_url:
            logger.warning(
                f"{LOG_PREFIX} {model_name}: no meta JSON asset "
                f"'{norm_prefix}.json' in release {remote_tag}"
            )
            return

        # Download and install
        action = "Updating" if local_version else "Downloading"
        logger.info(
            f"{LOG_PREFIX} {action} {model_name}: "
            f"{local_version or '(none)'} → {remote_tag}"
        )

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

            # Download ONNX weights (300s timeout — can be 150MB+)
            logger.info(f"{LOG_PREFIX} Downloading ONNX weights for {model_name}...")
            req = Request(onnx_url, headers={"User-Agent": "Chalie/1.0"})
            with urlopen(req, timeout=300) as resp:
                onnx_dest = staging / "model.onnx"
                onnx_dest.write_bytes(resp.read())
            logger.info(
                f"{LOG_PREFIX} ONNX weights downloaded "
                f"({onnx_dest.stat().st_size / 1048576:.0f}MB)"
            )

            # Atomic swap: remove old, rename staging to final
            if model_dir.exists():
                shutil.rmtree(model_dir)
            staging.rename(model_dir)

            # Invalidate cache so next predict() reloads
            with self._lock:
                self._cache.pop(model_name, None)

            logger.info(f"{LOG_PREFIX} Installed {model_name} ({remote_tag})")

        except (URLError, OSError) as e:
            logger.warning(f"{LOG_PREFIX} Download failed for {model_name}: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Unexpected error installing {model_name}: {e}")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    # ── Public API ────────────────────────────────────────────

    def predict(self, model_name: str, text: str) -> Tuple[Optional[str], float]:
        """
        Run single-token classification.

        Args:
            model_name: Subdirectory name under MODELS_DIR (e.g. "mode-tiebreaker")
            text: Full input text for the classifier

        Returns:
            (label, confidence) — label is None if the model isn't available.
            Confidence is the softmax probability of the winning label.
        """
        model = self._get_model(model_name)
        if model is None:
            return None, 0.0

        try:
            start = time.perf_counter()

            # Tokenize
            encoded = model.tokenizer(
                text,
                return_tensors="np",
                padding=False,
                truncation=True,
                max_length=256,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Run ONNX inference (build_feed handles extra traced inputs like RoPE)
            outputs = model.session.run(
                None,
                model.build_feed(input_ids, attention_mask),
            )

            # Extract logits: shape (batch, seq_len, vocab_size)
            logits = outputs[0]

            # Last real token position
            seq_len = int(attention_mask.sum()) - 1
            last_logits = logits[0, seq_len, :]

            # Compare logits for each label's token ID
            label_logits = np.array([last_logits[tid] for tid in model.label_token_ids])

            # Softmax over label logits only
            label_logits_shifted = label_logits - label_logits.max()
            exp_logits = np.exp(label_logits_shifted)
            probs = exp_logits / exp_logits.sum()

            winner_idx = int(np.argmax(probs))
            confidence = float(probs[winner_idx])
            label = model.labels[winner_idx]

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"{LOG_PREFIX} {model_name}: {label} ({confidence:.3f}) in {elapsed_ms:.1f}ms"
            )

            return label, confidence

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Inference failed for {model_name}: {e}")
            return None, 0.0

    def predict_batch(self, model_name: str, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        """
        Run classification on a batch of inputs.

        Returns list of (label, confidence) tuples, one per input.
        """
        model = self._get_model(model_name)
        if model is None:
            return [(None, 0.0)] * len(texts)

        try:
            encoded = model.tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=256,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = model.session.run(
                None,
                model.build_feed(input_ids, attention_mask),
            )

            logits = outputs[0]
            results = []

            for i in range(len(texts)):
                seq_len = int(attention_mask[i].sum()) - 1
                last_logits = logits[i, seq_len, :]

                label_logits = np.array([last_logits[tid] for tid in model.label_token_ids])
                label_logits_shifted = label_logits - label_logits.max()
                exp_logits = np.exp(label_logits_shifted)
                probs = exp_logits / exp_logits.sum()

                winner_idx = int(np.argmax(probs))
                confidence = float(probs[winner_idx])
                results.append((model.labels[winner_idx], confidence))

            return results

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Batch inference failed for {model_name}: {e}")
            return [(None, 0.0)] * len(texts)

    def is_available(self, model_name: str) -> bool:
        """Check if a model is loaded or loadable."""
        return self._get_model(model_name) is not None

    # ── Internal ──────────────────────────────────────────────

    def _get_model(self, model_name: str) -> Optional[_CachedModel]:
        """Lazy-load and cache a model. Returns None if unavailable."""
        # Fast path: already cached (including negative cache)
        if model_name in self._cache:
            return self._cache[model_name]

        with self._lock:
            # Double-check after acquiring lock
            if model_name in self._cache:
                return self._cache[model_name]

            model = self._load_model(model_name)
            self._cache[model_name] = model
            return model

    def _load_model(self, model_name: str) -> Optional[_CachedModel]:
        """Load ONNX session, tokenizer, and label metadata from disk."""
        model_dir = self._models_dir / model_name

        if not model_dir.is_dir():
            logger.warning(
                f"{LOG_PREFIX} Model directory not found: {model_dir} — "
                f"{model_name} classifier unavailable, will use fallback"
            )
            return None

        # Find the ONNX file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            logger.warning(f"{LOG_PREFIX} No .onnx file in {model_dir}")
            return None

        onnx_path = onnx_files[0]

        # Load classifier metadata
        meta_path = model_dir / "classifier_meta.json"
        if not meta_path.exists():
            logger.warning(f"{LOG_PREFIX} Missing classifier_meta.json in {model_dir}")
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            labels = meta["labels"]
            version = meta.get("version", "unknown")
            base_model = meta.get("base_model")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"{LOG_PREFIX} Invalid classifier_meta.json in {model_dir}: {e}")
            return None

        try:
            import onnxruntime as ort

            # CPU-only, single-thread for minimal latency on small models
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except ImportError:
            logger.warning(f"{LOG_PREFIX} onnxruntime not installed — ONNX classifiers unavailable")
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load ONNX session from {onnx_path}: {e}")
            return None

        # Load tokenizer — bundled in model directory (no HF download needed).
        # Falls back to HuggingFace base model name if local tokenizer missing.
        try:
            from transformers import AutoTokenizer

            if (model_dir / "tokenizer.json").exists():
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                logger.info(f"{LOG_PREFIX} Loaded tokenizer from {model_dir}")
            elif base_model:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                logger.info(f"{LOG_PREFIX} Loaded tokenizer from HuggingFace: {base_model}")
            else:
                logger.warning(f"{LOG_PREFIX} No tokenizer found for {model_name}")
                return None
        except ImportError:
            logger.warning(f"{LOG_PREFIX} transformers not installed — ONNX classifiers unavailable")
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to load tokenizer for {model_name}: {e}")
            return None

        # Map labels to token IDs
        label_token_ids = []
        for label in labels:
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if not token_ids:
                logger.warning(f"{LOG_PREFIX} Label '{label}' has no token ID in tokenizer")
                return None
            label_token_ids.append(token_ids[0])

        logger.info(
            f"{LOG_PREFIX} Loaded {model_name} ({version}): {onnx_path.name}, "
            f"labels={labels}, token_ids={label_token_ids}"
        )

        return _CachedModel(
            session=session,
            tokenizer=tokenizer,
            labels=labels,
            label_token_ids=label_token_ids,
            version=version,
        )


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
