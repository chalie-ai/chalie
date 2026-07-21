"""In-house Silero VAD wrapper running ``silero_vad.onnx`` through the shared
ONNX Runtime session factory (:mod:`services.onnx_session`) — the same
chokepoint embeddings, the classifier heads, and rapidocr already use.

Replaces the ``silero-vad-lite`` pip dependency (deleted: no Linux arm64
wheel across its release history) with a ~50-line wrapper around the base
``onnxruntime`` CPU wheel that already ships with every install.

Model contract, introspected from the real session (never assumed — see
``get_inputs()``/``get_outputs()`` on ``silero_vad.onnx`` v6.2.1, opset 16):
    input  ``input``  float32 ``[batch, window + context]`` — a fixed-size
           audio window prefixed with the previous call's trailing
           ``context`` samples (zero-filled on the first call).
    input  ``state``  float32 ``[2, batch, 128]``  — recurrent state, carried
           between calls, reset to zero on construction (fresh per utterance,
           matching the caller's own one-instance-per-utterance lifecycle).
    input  ``sr``      int64 scalar — 8000 or 16000.
    output ``output``  float32 ``[batch, 1]``       — speech probability.
    output ``stateN``  float32 ``[2, batch, 128]``  — next ``state``.
"""

from __future__ import annotations

import numpy as np

from services.file_mapper_service import FileMapperService
from services.onnx_session import build_session

_MODEL_PATH = FileMapperService.get_voice_models_path("silero_vad.onnx")

# Window/context sizes are fixed by the model architecture, not configurable.
_WINDOW_SAMPLES = {8000: 256, 16000: 512}
_CONTEXT_SAMPLES = {8000: 32, 16000: 64}
_STATE_SHAPE = (2, 1, 128)


class SileroVad:
    """Streaming voice-activity detector. One instance per utterance — construct
    fresh (resets recurrent state), feed fixed-size chunks via ``process()``."""

    def __init__(self, sample_rate: int) -> None:
        if sample_rate not in _WINDOW_SAMPLES:
            raise ValueError(f"Unsupported Silero VAD sample rate: {sample_rate}")
        self._window = _WINDOW_SAMPLES[sample_rate]
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._context = np.zeros((1, _CONTEXT_SAMPLES[sample_rate]), dtype=np.float32)
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._session = build_session(_MODEL_PATH, log_prefix="[SileroVAD]")

    def process(self, chunk: "np.ndarray") -> float:
        """Feed one ``window``-sample float32 chunk; return its speech probability."""
        if chunk.shape[0] != self._window:
            raise ValueError(f"Expected a {self._window}-sample chunk, got {chunk.shape[0]}")
        x = np.concatenate(
            [self._context, chunk.reshape(1, -1).astype(np.float32)], axis=1
        )
        out, state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = x[:, -_CONTEXT_SAMPLES[int(self._sr)]:]
        return float(out[0, 0])
