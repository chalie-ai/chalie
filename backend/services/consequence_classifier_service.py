# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ConsequenceClassifierService — action consequence tier classification.

Classifies proposed actions into 4 consequence tiers based on their
reversibility and external impact:

  Tier 0 — OBSERVE:  Research, read, gather info, web search, recall, introspect.
                     Zero side-effects; always safe to auto-execute.
  Tier 1 — ORGANIZE: Create notes, update lists, file documents, memorize, associate.
                     Internal state changes; reversible with low effort.
  Tier 2 — ACT:      Send messages, schedule reminders, call external APIs.
                     External side-effects; reversible but require active cleanup.
  Tier 3 — COMMIT:   Spend money, delete data, irreversible external actions.
                     Hard or impossible to undo; always require explicit approval.

Primary path: ONNX multi-label classifier (Qwen2.5-0.5B base, ~5ms inference).
Fallback: deterministic keyword-matching rule engine (zero dependencies).

The ONNX model is optional — the service works fully without it. A warning
is logged on startup if the model directory is absent.

Integration points (Stage 6a):
  - ActDispatcherService: gate external actions by tier before execution
  - PlanAction: block Tier 3 tasks from autonomous creation
  - GoalInferenceService: decide PROPOSED vs auto-accept for inferred goals
  - PersistentTaskWorker: verify tier before each background cycle
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CONSEQUENCE]"

# Model name registered in OnnxInferenceService
_MODEL_NAME = "consequence-classifier"

# ONNX confidence threshold below which we trust the rule-based fallback
# instead of the low-confidence ONNX prediction.
_ONNX_CONFIDENCE_THRESHOLD = 0.60

# ── Keyword sets ──────────────────────────────────────────────────────────────
#
# Each set defines words/phrases whose presence in the action description
# strongly signals that tier. The rule engine scans in descending tier order
# (COMMIT → ACT → ORGANIZE → OBSERVE) and stops at the first match, which
# means higher-consequence tiers always win on ambiguous descriptions.
#
# Phrases are matched as substrings after lowercasing the description.
# Single-word entries match as substrings too, so "delete" matches "delete file".

_COMMIT_KEYWORDS = frozenset([
    "delete",
    "remove",
    "purchase",
    "buy",
    "pay",
    "cancel subscription",
    "drop",
    "destroy",
    "spend",
    "transfer money",
    "wire transfer",
    "charge",
    "irreversible",
    "permanently",
    "unrecoverable",
])

_ACT_KEYWORDS = frozenset([
    "send",
    "schedule",
    "remind",
    "notify",
    "message",
    "email",
    "post",
    "call",
    "book",
    "reserve",
    "create task",
    "submit",
    "publish",
    "reply",
    "respond",
    "invite",
    "share",
    "upload",
    "deploy",
    "api",
])

_ORGANIZE_KEYWORDS = frozenset([
    "note",
    "list",
    "memorize",
    "save",
    "store",
    "file",
    "organize",
    "tag",
    "categorize",
    "associate",
    "update list",
    "create note",
    "add to",
    "archive",
    "label",
    "index",
    "log",
])

_OBSERVE_KEYWORDS = frozenset([
    "research",
    "search",
    "read",
    "find",
    "look up",
    "recall",
    "check",
    "query",
    "introspect",
    "browse",
    "fetch",
    "gather",
    "inspect",
    "review",
    "scan",
    "summarize",
    "analyse",
    "analyze",
    "explore",
    "investigate",
    "discover",
    "monitor",
])


def _kw_matches(keyword: str, text: str) -> bool:
    """
    Return True if ``keyword`` matches in ``text`` with word-boundary awareness.

    Multi-word phrases (containing a space) are matched as plain substrings so
    that "look up" matches "look up the price" without false positives.

    Single-word keywords use \\b word boundaries so that "call" does NOT match
    inside "recall", and "read" does NOT match inside "thread".

    Both ``keyword`` and ``text`` must already be lowercased by the caller.
    """
    if " " in keyword:
        return keyword in text
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def _rule_based_classify(action_description: str) -> dict:
    """
    Deterministic keyword-matching classifier.

    Scans in COMMIT → ACT → ORGANIZE → OBSERVE order so higher-consequence
    tiers win on descriptions that contain keywords from multiple tiers.
    Returns Tier 2 (ACT) if no keywords match — erring on the side of caution.

    Single-word keywords use word-boundary matching (\\b) to avoid false
    positives from substrings (e.g. "call" inside "recall").
    Multi-word phrases are matched as plain substrings.
    """
    text = action_description.lower()

    # Tier 3: COMMIT
    for kw in _COMMIT_KEYWORDS:
        if _kw_matches(kw, text):
            return {
                "tier": ConsequenceClassifierService.COMMIT,
                "tier_name": "commit",
                "confidence": 1.0,
                "scores": {"observe": 0.0, "organize": 0.0, "act": 0.0, "commit": 1.0},
                "method": "rule_based",
                "matched_keyword": kw,
            }

    # Tier 2: ACT
    for kw in _ACT_KEYWORDS:
        if _kw_matches(kw, text):
            return {
                "tier": ConsequenceClassifierService.ACT,
                "tier_name": "act",
                "confidence": 1.0,
                "scores": {"observe": 0.0, "organize": 0.0, "act": 1.0, "commit": 0.0},
                "method": "rule_based",
                "matched_keyword": kw,
            }

    # Tier 1: ORGANIZE
    for kw in _ORGANIZE_KEYWORDS:
        if _kw_matches(kw, text):
            return {
                "tier": ConsequenceClassifierService.ORGANIZE,
                "tier_name": "organize",
                "confidence": 1.0,
                "scores": {"observe": 0.0, "organize": 1.0, "act": 0.0, "commit": 0.0},
                "method": "rule_based",
                "matched_keyword": kw,
            }

    # Tier 0: OBSERVE
    for kw in _OBSERVE_KEYWORDS:
        if _kw_matches(kw, text):
            return {
                "tier": ConsequenceClassifierService.OBSERVE,
                "tier_name": "observe",
                "confidence": 1.0,
                "scores": {"observe": 1.0, "organize": 0.0, "act": 0.0, "commit": 0.0},
                "method": "rule_based",
                "matched_keyword": kw,
            }

    # No keyword matched — default to Tier 2 (ACT) for safety.
    logger.debug(f"{LOG_PREFIX} No keywords matched, defaulting to ACT: {action_description!r}")
    return {
        "tier": ConsequenceClassifierService.ACT,
        "tier_name": "act",
        "confidence": 0.5,
        "scores": {"observe": 0.0, "organize": 0.0, "act": 1.0, "commit": 0.0},
        "method": "rule_based",
        "matched_keyword": None,
    }


class ConsequenceClassifierService:
    """
    Consequence tier classifier for proposed actions.

    Uses ONNX inference when the model is available; falls back to a
    deterministic keyword-based rule engine otherwise.

    Usage::

        svc = ConsequenceClassifierService()
        result = svc.classify("search the web for Python tutorials")
        # → {"tier": 0, "tier_name": "observe", "confidence": 0.97, ...}

        if not svc.is_reversible(result["tier"]):
            ask_user_for_confirmation(...)

    Thread-safe — the underlying OnnxInferenceService is thread-safe and
    the rule-based fallback has no shared mutable state.
    """

    # Tier constants — these are STABLE and must match the training labels.
    # Changing them requires retraining the ONNX model.
    OBSERVE = 0
    ORGANIZE = 1
    ACT = 2
    COMMIT = 3

    TIER_NAMES = {0: "observe", 1: "organize", 2: "act", 3: "commit"}

    # ONNX label order must match training/configs/consequence_classifier.yaml
    _ONNX_LABELS = ["observe", "organize", "act", "commit"]

    def __init__(self, models_dir: Optional[str] = None):
        """
        Args:
            models_dir: Path to the models directory. If None, resolved from
                        runtime_config or the default data/models path.
        """
        self._models_dir = self._resolve_models_dir(models_dir)
        self._onnx_available: Optional[bool] = None  # None = unchecked

        model_path = Path(self._models_dir) / _MODEL_NAME / "classifier_meta.json"
        if not model_path.exists():
            logger.warning(
                f"{LOG_PREFIX} ONNX model not found at "
                f"{model_path.parent} — using rule-based fallback. "
                "Train the model with: training/configs/consequence_classifier.yaml"
            )
        else:
            logger.info(f"{LOG_PREFIX} Initialized (ONNX model available: {model_path.parent})")

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, action_description: str) -> dict:
        """
        Classify a proposed action into a consequence tier.

        The ONNX model is tried first. If it is unavailable, confidence
        falls below threshold, or inference errors, the rule-based classifier
        is used as a reliable fallback.

        Args:
            action_description: Natural-language description of the action,
                                 e.g. "send an email to Alice about the meeting".

        Returns:
            dict with keys:
              - ``tier`` (int): 0–3 consequence tier
              - ``tier_name`` (str): 'observe', 'organize', 'act', or 'commit'
              - ``confidence`` (float): 0.0–1.0 classifier confidence
              - ``scores`` (dict): per-tier scores keyed by tier name
              - ``method`` (str): 'onnx' or 'rule_based'
        """
        if not action_description or not action_description.strip():
            # Empty action: default to ACT (safe direction)
            return {
                "tier": self.ACT,
                "tier_name": "act",
                "confidence": 0.5,
                "scores": {"observe": 0.0, "organize": 0.0, "act": 1.0, "commit": 0.0},
                "method": "rule_based",
            }

        onnx_result = self._classify_onnx(action_description)
        if onnx_result is not None:
            return onnx_result

        return _rule_based_classify(action_description)

    def is_reversible(self, tier: int) -> bool:
        """
        Return True if the consequence tier is considered reversible.

        Tiers 0 (OBSERVE), 1 (ORGANIZE), and 2 (ACT) are reversible.
        Tier 3 (COMMIT) is not.

        Used by the autonomous execution gate (Stage 6a Component 3) to decide
        whether to allow autonomous execution or always ask the user.
        """
        return tier < self.COMMIT

    # ── ONNX path ─────────────────────────────────────────────────────────────

    def _classify_onnx(self, action_description: str) -> Optional[dict]:
        """
        Classify via the ONNX model.

        Returns a result dict if the model is available and confident enough,
        or None to signal that the rule-based fallback should be used.
        """
        try:
            from services.onnx_inference_service import get_onnx_inference_service

            svc = get_onnx_inference_service()
            if not svc.is_available(_MODEL_NAME):
                if self._onnx_available is not False:
                    self._onnx_available = False
                    logger.debug(f"{LOG_PREFIX} ONNX model not available — using rule-based fallback")
                return None

            self._onnx_available = True
            start = time.perf_counter()

            # predict_multi_label returns [(label, score), ...] above threshold,
            # sorted descending. Use raw scores for all tiers via threshold_overrides=0.0.
            raw_results = svc.predict_multi_label(
                _MODEL_NAME,
                action_description,
                threshold_overrides={label: 0.0 for label in self._ONNX_LABELS},
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Build per-tier score map from raw_results
            scores_map = {label: 0.0 for label in self._ONNX_LABELS}
            for label, score in raw_results:
                if label in scores_map:
                    scores_map[label] = score

            # The highest-scoring tier wins (treat as ordinal selection, not multi-label).
            # If no tier exceeds threshold, fall back to rule-based.
            if not scores_map or max(scores_map.values()) < _ONNX_CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"{LOG_PREFIX} ONNX scores below threshold "
                    f"(max={max(scores_map.values(), default=0.0):.3f}) — using rule-based fallback"
                )
                return None

            best_label = max(scores_map, key=scores_map.__getitem__)
            confidence = scores_map[best_label]

            tier_map = {name: idx for idx, name in self.TIER_NAMES.items()}
            tier = tier_map.get(best_label, self.ACT)

            logger.debug(
                f"{LOG_PREFIX} ONNX: {best_label} (confidence={confidence:.3f}) "
                f"in {elapsed_ms:.1f}ms"
            )

            return {
                "tier": tier,
                "tier_name": best_label,
                "confidence": confidence,
                "scores": scores_map,
                "method": "onnx",
            }

        except ImportError:
            logger.debug(f"{LOG_PREFIX} onnxruntime not available — using rule-based fallback")
            return None
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ONNX inference error — using rule-based fallback: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_models_dir(models_dir: Optional[str]) -> str:
        if models_dir:
            return models_dir

        try:
            import runtime_config
            default = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "models"
            )
            return runtime_config.get(
                "models_dir",
                os.environ.get("MODELS_DIR", default),
            )
        except ImportError:
            return os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "models"
            )


# ── Singleton ─────────────────────────────────────────────────────────────────

import threading

_instance: Optional[ConsequenceClassifierService] = None
_instance_lock = threading.Lock()


def get_consequence_classifier_service() -> ConsequenceClassifierService:
    """Get or create the singleton ConsequenceClassifierService."""
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance
        _instance = ConsequenceClassifierService()
        return _instance
