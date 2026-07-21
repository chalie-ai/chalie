"""Feature tests for services/silero_vad_service.py — the in-house Silero VAD
wrapper that runs ``silero_vad.onnx`` through the shared onnxruntime session
factory (services.onnx_session.build_session), replacing the deleted
``silero-vad-lite`` dependency (no Linux arm64 wheel across its release
history, and it bundled a second, private ONNX Runtime).

Real model file, real onnxruntime inference, zero mocks. The model asset is
an install-time download (resources/voice-models/silero_vad.onnx, gitignored
— installer/install.sh / the Docker build fetch it) so these tests skip
cleanly when it's absent rather than fabricating an inference result.
"""

import numpy as np
import pytest

from services.file_mapper_service import FileMapperService
from services.silero_vad_service import SileroVad

_MODEL_PATH = FileMapperService.get_voice_models_path("silero_vad.onnx")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _MODEL_PATH.exists(),
        reason="silero_vad.onnx is an install-time asset (installer/install.sh / "
        "Dockerfile); not present in this environment",
    ),
]

_SR = 16000
_WINDOW = 512
_SPEECH_THRESHOLD = 0.4  # mirrors backend/api/voice.py::_VAD_SPEECH_THRESHOLD


def _synthesize_voiced_signal(duration_s: float = 2.0, f0: float = 120.0) -> "np.ndarray":
    """A deterministic, voiced-speech-shaped signal: a fundamental + four
    harmonics, amplitude-modulated at a syllable rate (~4 Hz), plus a whisper
    of noise — real numeric audio, not a fabricated model output. No TTS
    engine is required (kokoro/moonshine are not guaranteed present in every
    test environment); this is fed through the real model exactly like a real
    recording would be. Empirically verified against the real model to drive
    >95% of chunks above the speech threshold.
    """
    t = np.arange(int(_SR * duration_s)) / _SR
    sig = np.zeros_like(t)
    for harmonic in range(1, 6):
        sig += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t - np.pi / 2)
    sig = sig * envelope
    sig = sig / np.max(np.abs(sig)) * 0.6
    noise = np.random.default_rng(0).normal(0, 0.01, size=sig.shape)
    result: np.ndarray = (sig + noise).astype(np.float32)
    return result


def _run_chunks(vad: SileroVad, signal: "np.ndarray") -> "np.ndarray":
    probs = []
    for start in range(0, len(signal) - _WINDOW, _WINDOW):
        probs.append(vad.process(signal[start:start + _WINDOW]))
    return np.array(probs)


class TestSileroVadInference:
    def test_fresh_instance_on_silence_yields_low_probability(self) -> None:
        vad = SileroVad(_SR)
        prob = vad.process(np.zeros(_WINDOW, dtype=np.float32))
        assert prob < 0.05, (
            f"pure silence should score far below the {_SPEECH_THRESHOLD} "
            f"speech threshold, got {prob}"
        )

    def test_synthesized_voiced_signal_exceeds_threshold_on_most_chunks(self) -> None:
        vad = SileroVad(_SR)
        probs = _run_chunks(vad, _synthesize_voiced_signal())
        frac_above = (probs > _SPEECH_THRESHOLD).mean()
        assert frac_above > 0.8, (
            f"a real voiced signal should cross the speech threshold on most "
            f"chunks, got {frac_above:.2%} (mean prob {probs.mean():.4f})"
        )

    def test_state_does_not_leak_across_fresh_instances(self) -> None:
        # Drive one instance hard with a loud voiced signal so its recurrent
        # state is saturated, then construct two brand-new instances and feed
        # each pure silence — a fresh SileroVad must score identical input
        # identically, regardless of what any other instance has processed.
        busy = SileroVad(_SR)
        _run_chunks(busy, _synthesize_voiced_signal(duration_s=1.0))

        fresh_a = SileroVad(_SR)
        fresh_b = SileroVad(_SR)
        prob_a = fresh_a.process(np.zeros(_WINDOW, dtype=np.float32))
        prob_b = fresh_b.process(np.zeros(_WINDOW, dtype=np.float32))

        assert prob_a < 0.05 and prob_b < 0.05
        assert abs(prob_a - prob_b) < 1e-6, (
            "two fresh instances must score identical silence identically — "
            f"got {prob_a} vs {prob_b}"
        )

    def test_unsupported_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            SileroVad(44100)

    def test_wrong_chunk_size_raises(self) -> None:
        vad = SileroVad(_SR)
        with pytest.raises(ValueError):
            vad.process(np.zeros(256, dtype=np.float32))


class TestExtractSpeechEndToEnd:
    def test_extract_speech_strips_silence_and_keeps_voiced_segment(self) -> None:
        # Real end-to-end path: backend.api.voice._extract_speech() drives the
        # real SileroVad wrapper exactly as the STT route does, no mocks.
        from api.voice import _extract_speech

        silence_pad = np.zeros(_SR, dtype=np.float32)  # 1s silence each side
        voiced = _synthesize_voiced_signal(duration_s=1.0)
        audio = np.concatenate([silence_pad, voiced, silence_pad])

        result = np.asarray(_extract_speech(audio, _SR))

        assert 0 < len(result) < len(audio), (
            "the real VAD path should strip most of the surrounding silence "
            f"while keeping the voiced segment (in={len(audio)}, out={len(result)})"
        )
