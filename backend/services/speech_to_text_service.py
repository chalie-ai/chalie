"""STT service — Moonshine-based speech-to-text.

Owns transcription outright, so non-HTTP callers (the MP spine, background
tasks, unit tests) can transcribe audio without spinning up a Flask request.

The service owns:
* Moonshine model creation and lazy loading (with its own lock and loading-state
  flags that back the 503 contract)
* Dependency detection (voice packages on the import path)
* VAD-based speech extraction and spectral noise reduction
* Hallucination-dedup, filler-stripping and contraction-restoration post-processing
* WAV duration inspection and windowed transcription of arbitrarily long clips
  (up to ``MAX_AUDIO_SECONDS``)

Public API: :meth:`transcribe` runs Moonshine on raw WAV bytes;
:meth:`wav_duration_seconds` parses a WAV header; :meth:`ensure_loaded` drives
the lazy load; :meth:`missing_model_files` reports missing model files on disk.
"""

from __future__ import annotations

import logging
import re
import struct
import tempfile
import threading
from typing import TYPE_CHECKING, cast

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from typing import Protocol

    class _MoonshineModel(Protocol):
        def transcribe(self, audio: object, *, model: object) -> list[str]: ...

logger = logging.getLogger(__name__)

# ── Model paths & transcription config (hardcoded) ────────────────────────────

# Models live in resources/voice-models/ — a directory sibling to backend/,
# intentionally OUTSIDE the data/ volume so the files survive an app upgrade on
# native installs. They are downloaded once at install time (installer/install.sh
# / the Docker build), never at boot or runtime.
_MOONSHINE_DIR = FileMapperService.get_voice_models_path("moonshine", "base")

STT_MODEL_NAME = "base"

# Moonshine internally asserts duration < 64s, so a single transcribe call can
# only consume short clips. For longer narrations we split the audio into
# CHUNK_SECONDS windows and concatenate the per-chunk text. MAX_AUDIO_SECONDS
# is the hard cap on the *whole* upload (10 min) to bound model time + memory.
CHUNK_SECONDS = 60
MAX_AUDIO_SECONDS = 600
MOONSHINE_SAMPLE_RATE = 16000
# Moonshine asserts the clip is at least 0.1s; below that the model errors out.
_MIN_CHUNK_SAMPLES = int(0.1 * MOONSHINE_SAMPLE_RATE)

# Moonshine (Whisper-family) hallucinates by repeating phrases 15-20x on
# silence or noise. Collapse any n-gram (2-20 words) that appears more than
# this many consecutive times down to exactly this many occurrences.
_MAX_CONSECUTIVE_PHRASE_REPEATS = 2
_MAX_NGRAM_WORDS = 20

# Silero VAD settings (16 kHz audio, 512-sample / 32ms windows).
# Lower threshold -> fewer false negatives (borderline speech passes through).
_VAD_SPEECH_THRESHOLD = 0.4
# Pad detected speech regions on both sides to avoid clipping word edges.
_VAD_PAD_SAMPLES = 1024  # 64 ms at 16 kHz

# ── Dependency detection ────────────────────────────────────────────────────

_VOICE_REQUIRED_MODULES = (
    "kokoro_onnx", "moonshine_onnx", "soundfile", "numpy", "noisereduce",
)
_VOICE_INSTALL_HINT = (
    "Voice dependencies are missing from this install. Reinstall Chalie to restore them."
)


class SpeechToTextService:
    """STT service — the single source of truth for Moonshine-based
    speech-to-text transcription.

    Transcription serialises on an internal lock, so concurrent callers queue
    rather than loading a second copy of the model.
    """

    def __init__(self) -> None:
        # ── Lazy model state ────────────────────────────────────────────
        #
        # Kokoro loading state is owned by VoiceTranscriptService.  Moonshine
        # state lives here because STT has its own lock and its loading path is
        # only used by the transcribe route.
        self._moonshine: object = None
        self._stt_lock = threading.Lock()
        self._moonshine_load_lock = threading.Lock()
        self._moonshine_loaded = False
        self._moonshine_loading = False

        # ── Dependency state ────────────────────────────────────────────
        self._voice_missing_modules: tuple[str, ...] = self._detect_voice_modules()

    # ── Dependency introspection ──────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether all voice dependencies are importable."""
        return not self._voice_missing_modules

    @property
    def missing_modules(self) -> tuple[str, ...]:
        """The names of voice dependencies that are missing from this install."""
        return self._voice_missing_modules

    @property
    def install_hint(self) -> str:
        """A human-readable hint about how to restore voice functionality."""
        return _VOICE_INSTALL_HINT

    # ── Loading-state introspection (used by the HTTP layer to decide 503) ─

    @property
    def is_loaded(self) -> bool:
        """Whether the Moonshine model is ready for transcription."""
        return self._moonshine_loaded

    @property
    def is_loading(self) -> bool:
        """Whether the Moonshine model is currently being loaded (transient)."""
        return self._moonshine_loading

    # ── Transcription ─────────────────────────────────────────────────────

    def transcribe(self, data: bytes) -> str:
        """Run Moonshine on raw WAV bytes (blocking).

        Acquires the STT lock to serialise concurrent transcription requests.
        Clips longer than ``CHUNK_SECONDS`` are split into fixed-size windows and
        transcribed one window at a time so an arbitrarily long narration (up to
        ``MAX_AUDIO_SECONDS``) can be handled despite Moonshine's internal 64s
        assertion. Window results are concatenated with single spaces.
        """
        with self._stt_lock:
            return self._transcribe_sync(data)

    # ── Private: dependency detection ─────────────────────────────────────

    def _detect_voice_modules(self) -> tuple[str, ...]:
        import importlib.util
        return tuple(
            name for name in _VOICE_REQUIRED_MODULES
            if importlib.util.find_spec(name) is None
        )

    # ── Lazy model loading ────────────────────────────────────────────────

    def missing_model_files(self) -> list[str]:
        """Return the names of Moonshine model files that are missing on disk."""
        expected = [
            _MOONSHINE_DIR / "encoder_model.onnx",
            _MOONSHINE_DIR / "decoder_model_merged.onnx",
        ]
        return [p.name for p in expected if not p.is_file()]

    def ensure_loaded(self) -> bool:
        """Lazily load the Moonshine STT model. Returns ``False`` if files are
        missing or loading is in progress (caller should 503)."""
        if self._moonshine_loaded:
            return True

        with self._moonshine_load_lock:
            if self._moonshine_loaded:
                return True
            if self._moonshine_loading:
                return False
            self._moonshine_loading = True

        try:
            missing = self.missing_model_files()
            if missing:
                logger.error("[Voice] Moonshine model files missing: %s", missing)
                with self._moonshine_load_lock:
                    self._moonshine_loading = False
                return False

            import moonshine_onnx as mo

            logger.info("[Voice] Loading Moonshine STT from %s", _MOONSHINE_DIR)
            moonshine = mo.MoonshineOnnxModel(
                models_dir=str(_MOONSHINE_DIR),
                model_name=STT_MODEL_NAME,
            )

            with self._moonshine_load_lock:
                self._moonshine = moonshine
                self._moonshine_loaded = True
                self._moonshine_loading = False

            logger.info("[Voice] Moonshine loaded — accepting transcription requests")
            return True

        except Exception as e:
            logger.error("[Voice] Moonshine loading failed: %s", e)
            with self._moonshine_load_lock:
                self._moonshine_loading = False
            return False

    # ── Private: STT audio preprocessing ──────────────────────────────────

    def _dedup_repetitions(self, text: str) -> str:
        """Collapse Moonshine hallucination loops (consecutive identical 2–20 word phrases)."""
        if not text:
            return text
        try:
            words = text.split()
            for n in range(_MAX_NGRAM_WORDS, 1, -1):
                if len(words) < n * 3:
                    continue
                lower = [w.lower() for w in words]
                out: list[str] = []
                i = 0
                while i < len(words):
                    if i + n > len(words):
                        out.extend(words[i:])
                        break
                    phrase = tuple(lower[i:i + n])
                    j, run = i + n, 1
                    while j + n <= len(words) and tuple(lower[j:j + n]) == phrase:
                        run += 1
                        j += n
                    if run > _MAX_CONSECUTIVE_PHRASE_REPEATS:
                        for _ in range(_MAX_CONSECUTIVE_PHRASE_REPEATS):
                            out.extend(words[i:i + n])
                        i = j
                    else:
                        out.append(words[i])
                        i += 1
                words = out
            return " ".join(words)
        except Exception:
            logger.exception("[Voice] _dedup_repetitions failed on len=%d", len(text))
            return text

    def _extract_speech(self, audio: object, sr: int) -> object:
        """Strip non-speech regions using Silero VAD.

        Returns a float32 array containing only the detected speech segments
        (with padding to avoid clipping word edges), or the original audio
        unchanged when no speech is confidently detected (avoids false-negative
        drops on quiet or noisy recordings). On any error the original audio
        is returned unchanged (fail-safe).
        """
        try:
            import numpy as np
            from services.silero_vad_service import SileroVad

            vad = SileroVad(sr)
            window = 512
            # SileroVAD.process() requires a *writable* float32 array.
            audio_rw = np.array(audio, dtype=np.float32)
            n = len(audio_rw)

            # Walk in 512-sample windows, zero-padding the last partial window.
            speech_flags: list[bool] = []
            for start in range(0, n, window):
                chunk = audio_rw[start:start + window]
                if len(chunk) < window:
                    chunk = np.concatenate([chunk, np.zeros(window - len(chunk), dtype=np.float32)])
                prob = vad.process(chunk)
                speech_flags.append(prob >= _VAD_SPEECH_THRESHOLD)

            # Collect speech sample ranges (one flag per 512-sample window).
            speech_regions: list[tuple[int, int]] = []
            for idx, is_speech in enumerate(speech_flags):
                if is_speech:
                    start = max(0, idx * window - _VAD_PAD_SAMPLES)
                    end = min(n, (idx + 1) * window + _VAD_PAD_SAMPLES)
                    speech_regions.append((start, end))

            if not speech_regions:
                return audio

            merged: list[list[int]] = []
            for region in speech_regions:
                if merged and region[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], region[1])
                else:
                    merged.append([region[0], region[1]])

            segments = [audio_rw[s:e] for s, e in merged]
            return np.concatenate(segments).astype(np.float32)
        except Exception:
            logger.exception("[Voice] VAD failed — passing audio through")
            return audio

    def _denoise(self, audio: object, sr: int) -> object:
        """Apply spectral noise reduction to a float32 audio array.

        Skips clips shorter than n_fft=2048 samples (128 ms at 16 kHz) to avoid
        STFT boundary errors. Returns original audio on any error (fail-safe).
        """
        if len(cast("list[object]", audio)) < 2048:
            return audio
        try:
            import numpy as np
            import noisereduce as nr

            result = nr.reduce_noise(y=audio, sr=sr, stationary=False)
            return np.array(result, dtype=np.float32)
        except Exception:
            logger.exception("[Voice] noisereduce failed — passing audio through")
            return audio

    # Filler words produced by speakers (not by Moonshine). Match whole words only
    # — "umbrella", "error", "hermit", "ahead", and "uh-huh" must NOT be stripped.
    _FILLER_RE = re.compile(
        r"\b(u+h+|u+m+|h+m+|e+r+|a+h+|e+r+m+)\b",
        re.IGNORECASE,
    )
    _MULTI_SPACE_RE = re.compile(r" {2,}")

    def _strip_fillers(self, text: str) -> str:
        if not text:
            return text
        return self._MULTI_SPACE_RE.sub(" ", self._FILLER_RE.sub("", text)).strip()

    # Moonshine drops apostrophes from contractions.  This table maps the
    # apostrophe-free form (lowercase) to the correct spelling. Ordered
    # longest-first so a longer match is never shadowed by a shorter prefix.
    _CONTRACTIONS: list[tuple[str, str]] = [
        ("shouldnt", "shouldn't"),
        ("couldnt", "couldn't"),
        ("wouldnt", "wouldn't"),
        ("doesnt", "doesn't"),
        ("havent", "haven't"),
        ("hadnt", "hadn't"),
        ("hasnt", "hasn't"),
        ("werent", "weren't"),
        ("wasnt", "wasn't"),
        ("arent", "aren't"),
        ("didnt", "didn't"),
        ("mustnt", "mustn't"),
        ("isnt", "isn't"),
        ("dont", "don't"),
        ("cant", "can't"),
        ("wont", "won't"),
        ("thats", "that's"),
        ("whats", "what's"),
        ("whos", "who's"),
        ("youre", "you're"),
        ("youve", "you've"),
        ("youll", "you'll"),
        ("youd", "you'd"),
        ("theyre", "they're"),
        ("theyve", "they've"),
        ("theyll", "they'll"),
        ("theyd", "they'd"),
        ("weve", "we've"),
        ("hes", "he's"),
        ("shes", "she's"),
        ("ive", "I've"),
        ("im", "I'm"),
    ]

    _CONTRACTION_RES: list[tuple["re.Pattern[str]", str]] = [
        (re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE), dst)
        for src, dst in _CONTRACTIONS
    ]

    def _fix_contractions(self, text: str) -> str:
        """Restore apostrophes that Moonshine drops from common English contractions.

        Ambiguous words whose bare form is independently valid ("were", "well",
        "its", "lets") are deliberately excluded to avoid false corrections.
        Casing of the original token is preserved (lower/title/upper).
        """
        if not text:
            return text
        for pattern, replacement in self._CONTRACTION_RES:
            def _recase(m: "re.Match[str]", rep: str = replacement) -> str:
                orig = m.group(0)
                if orig.isupper():
                    return rep.upper()
                if orig[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep
            text = pattern.sub(_recase, text)
        return text

    # ── WAV helpers ───────────────────────────────────────────────────────

    def wav_duration_seconds(self, data: bytes) -> float:
        """Return the duration of a WAV blob in seconds, or 0.0 on error."""
        try:
            if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                return 0.0
            channels = cast(int, struct.unpack_from("<H", data, 22)[0])
            sample_rate = cast(int, struct.unpack_from("<I", data, 24)[0])
            bits_per_sample = cast(int, struct.unpack_from("<H", data, 34)[0])
            if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
                return 0.0
            offset = 12
            while offset < len(data) - 8:
                chunk_id = data[offset:offset + 4]
                chunk_size = cast(int, struct.unpack_from("<I", data, offset + 4)[0])
                if chunk_id == b"data":
                    bytes_per_sample = bits_per_sample // 8
                    return chunk_size / (sample_rate * channels * bytes_per_sample)
                offset += 8 + chunk_size
            return 0.0
        except Exception:
            return 0.0

    def _transcribe_sync(self, data: bytes) -> str:
        """Run Moonshine on raw WAV bytes (blocking).

        Clips longer than ``CHUNK_SECONDS`` are split into fixed-size windows and
        transcribed one window at a time so an arbitrarily long narration (up to
        ``MAX_AUDIO_SECONDS``) can be handled despite Moonshine's internal 64s
        assertion. Window results are concatenated with single spaces.
        """
        import moonshine_onnx as mo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            # ``mo.load_audio(path)`` resamples to 16 kHz and wraps the samples in
            # a batch dim -> shape ``[1, N]``. ``mo.transcribe()`` internally calls
            # ``load_audio`` again on its input, which adds a SECOND batch dim and
            # fails the ``[batch, samples]`` assertion. Strip the wrapper before
            # we hand the samples back to transcribe (or slice them).
            audio = mo.load_audio(tmp.name)[0]  # -> float32 [N] @ 16 kHz

        import numpy as np
        audio = cast("np.ndarray[tuple[int], np.dtype[np.float32]]", self._extract_speech(audio, MOONSHINE_SAMPLE_RATE))
        audio = cast("np.ndarray[tuple[int], np.dtype[np.float32]]", self._denoise(audio, MOONSHINE_SAMPLE_RATE))

        if audio.shape[0] < _MIN_CHUNK_SAMPLES:
            return ""

        n_samples = audio.shape[0]
        chunk_size = CHUNK_SECONDS * MOONSHINE_SAMPLE_RATE

        if n_samples <= chunk_size:
            texts = mo.transcribe(audio, model=self._moonshine)
            raw = " ".join(t.strip() for t in texts).strip()
            result = self._dedup_repetitions(raw)
        else:
            parts: list[str] = []
            for start in range(0, n_samples, chunk_size):
                chunk = audio[start:start + chunk_size]
                if chunk.shape[0] < _MIN_CHUNK_SAMPLES:
                    # Trailing fragment shorter than Moonshine's min duration — skip
                    # it rather than let the model assert.
                    continue
                texts = mo.transcribe(chunk, model=self._moonshine)
                part = self._dedup_repetitions(" ".join(t.strip() for t in texts).strip())
                if part:
                    parts.append(part)
            result = self._dedup_repetitions(" ".join(parts).strip())

        result = self._strip_fillers(result)
        result = self._fix_contractions(result)
        return result


# Module-level singleton, mirroring the pattern in other services.
_default_service: SpeechToTextService | None = None


def get_service() -> SpeechToTextService:
    """Return the module-level singleton."""
    global _default_service
    if _default_service is None:
        _default_service = SpeechToTextService()
    return _default_service
