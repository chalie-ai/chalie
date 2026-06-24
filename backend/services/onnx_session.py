# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared ONNX Runtime session factory — CPU-only.

One chokepoint so every ONNX-using service builds sessions consistently.
The CPU execution provider is always used; no accelerator variants exist.
Runtime inference failures (mid-session) are the caller's responsibility —
this helper only covers construction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"


def build_session(
    model_path: Union[Path, str],
    sess_options: object = None,
    *,
    log_prefix: str = "[ONNX]",
) -> 'InferenceSession':
    import onnxruntime as ort

    model_path = Path(model_path)
    opts = sess_options if sess_options is not None else ort.SessionOptions()

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=[CPU_PROVIDER])
    logger.info(
        "%s session ready (%s, providers=%s)",
        log_prefix, model_path.name, session.get_providers(),
    )
    return session
