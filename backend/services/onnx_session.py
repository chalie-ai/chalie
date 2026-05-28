# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared ONNX Runtime session factory with automatic provider fallback.

One chokepoint so every ONNX-using service picks the same provider for the
installed wheel (CPU, CUDA, or ROCm) without hardcoding provider lists. The
installer decides which wheel lands on disk; this module decides which
execution provider actually runs each session and degrades to CPU on failure.

Policy:
    1. Default providers = ``ort.get_available_providers()`` — the wheel
       exposes whichever accelerators it was built with, in preference order.
    2. Drop ``CoreMLExecutionProvider`` for models whose weight tensors exceed
       Metal's 16384-dim 2D-texture ceiling; ORT's CoreML EP partitions such
       ops into hundreds of CPU sub-graphs and blows VSZ by >20 GB.
    3. ``CPUExecutionProvider`` is appended last when missing, guaranteeing
       node-level fallback inside ORT itself.
    4. If session construction raises with the preferred set, retry once with
       CPU only and log both attempts.

Runtime inference failures (mid-session) are the caller's responsibility —
this helper only covers construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Apple's Metal feature set caps a 2D-texture dimension at 16384 across every
# Mac (Intel, M1–M4, all tiers). ORT's CoreML EP honours the limit by
# partitioning offending ops into CPU sub-graphs, which for wide embedders
# (e.g. ModernBERT's 50368×768 embedding table) produces hundreds of
# partitions and jetsam-SIGKILLs lower-RAM Macs. Check dims up-front and drop
# CoreML when we can prove the model trips the limit.
METAL_TEXTURE_LIMIT = 16384

CPU_PROVIDER = "CPUExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"


def _model_fits_coreml(model_path: Path, limit: int = METAL_TEXTURE_LIMIT) -> bool:
    """True when every weight tensor dim is within CoreML's Metal limit.

    Fail-open on inspection errors: return True so the caller keeps CoreML and
    lets ORT decide at session load. The alternative — silently dropping
    CoreML on every model when ``onnx`` is missing — punishes the happy path.
    """
    try:
        import onnx
        m = onnx.load(str(model_path), load_external_data=False)
        return not any(d > limit for init in m.graph.initializer for d in init.dims)
    except Exception as e:
        logger.debug("CoreML shape pre-check skipped (%s: %s)", type(e).__name__, e)
        return True


def choose_providers(model_path: Optional[Union[Path, str]] = None) -> List[str]:
    """Return the provider list to hand ORT, ordered by preference.

    ``model_path`` is optional. When provided, it is inspected for the Metal
    texture limit and CoreML is stripped if the model would trip it. Pass
    ``None`` for models known to be small (e.g. classifier heads, Kokoro
    phoneme decoder).
    """
    import onnxruntime as ort

    providers = list(ort.get_available_providers())

    if model_path is not None and COREML_PROVIDER in providers:
        if not _model_fits_coreml(Path(model_path)):
            providers = [p for p in providers if p != COREML_PROVIDER]
            logger.info(
                "Dropped %s: %s has weight dim > %d (Metal 2D-texture ceiling)",
                COREML_PROVIDER, Path(model_path).name, METAL_TEXTURE_LIMIT,
            )

    if CPU_PROVIDER not in providers:
        providers.append(CPU_PROVIDER)

    return providers


def build_session(
    model_path: Union[Path, str],
    sess_options=None,
    providers: Optional[List[str]] = None,
    *,
    log_prefix: str = "[ONNX]",
):
    """Construct an ``ort.InferenceSession`` with CPU fallback on failure.

    Args:
        model_path: Path to the ``.onnx`` file.
        sess_options: Optional pre-configured ``ort.SessionOptions``. A fresh
            default is built when None.
        providers: Explicit provider list. When None, ``choose_providers`` is
            consulted.
        log_prefix: Bracketed tag prepended to log lines ("[VOICE]", etc.).

    Returns:
        The constructed ``ort.InferenceSession``.

    Raises:
        The original ORT error when CPU-only construction also fails — at that
        point the host cannot run the model at all.
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    chosen = list(providers) if providers is not None else choose_providers(model_path)
    opts = sess_options if sess_options is not None else ort.SessionOptions()

    try:
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=chosen)
        logger.info(
            "%s session ready (%s, providers=%s)",
            log_prefix, model_path.name, session.get_providers(),
        )
        return session
    except Exception as e:
        if chosen == [CPU_PROVIDER]:
            raise
        logger.warning(
            "%s providers=%s failed (%s: %s) — retrying CPU-only",
            log_prefix, chosen, type(e).__name__, e,
        )
        session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=[CPU_PROVIDER]
        )
        logger.info("%s CPU fallback session ready (%s)", log_prefix, model_path.name)
        return session
