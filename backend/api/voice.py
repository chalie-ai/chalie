"""
Voice namespace — STT + TTS via two single-purpose ONNX libraries.

* TTS → ``kokoro_onnx.Kokoro`` (Kokoro v1.0 + espeak-ng phonemizer) — owned by
  :class:`services.voice_transcript_service.VoiceTranscriptService`.
* STT → ``moonshine_onnx.MoonshineOnnxModel`` (Moonshine base, ONNX) — owned
  here in ``voice.py`` alongside the HTTP routes.

Voice deps and model files are installed unconditionally at install time
(``installer/install.sh`` / the Docker image build) into
``resources/voice-models/`` — nothing installs or downloads anything at boot
or runtime. If a dep or model file is somehow still missing (a broken/partial
install), every route returns ``{"status":"unavailable"}`` or 503 with a hint
to reinstall.

The playback route serves the WAV the pre-synthesis pipeline already stored for
a settled transcript row — a single ``audio/wav`` blob, no streaming, no NDJSON.
Synthesis itself never happens on the request thread: this route reads a
terminal outcome, or starts the pipeline once for a row that has none.

The transcribe route accepts a WAV upload, runs Moonshine on it (with VAD +
denoise + hallucination-dedup + filler/contraction post-processing), and
returns the text.
"""
import logging
import re
import struct
import tempfile
import threading
from typing import cast

from flask import Response, jsonify, make_response, request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from models.transcript import Transcript
from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService
from services.voice_transcript_service import get_service

from .auth import require_auth
from .dto import Error, register_dto, responds
from .dto.boundary import error
from .dto.voice import Transcription, VoiceHealth, VoiceState, VoiceUnavailable

logger = logging.getLogger(__name__)

voice_ns = Namespace("voice", description="Voice (STT + TTS) operations", path="/api/voice")

register_dto(voice_ns, Transcription, VoiceHealth, VoiceState, VoiceUnavailable, Error)
_V = voice_ns.models

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

# Models live in resources/voice-models/ — a directory sibling to backend/,
# intentionally OUTSIDE the data/ volume so the files survive an app upgrade on
# native installs. They are downloaded once at install time (installer/install.sh
# / the Docker build), never at boot or runtime.
#
# Only the Moonshine (STT) paths live here — the Kokoro (TTS) paths moved to
# VoiceTranscriptService, which the health/loading routes query for the
# combined missing-files check.
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

# Moonshine (Whisper-family) hallucinates by repeating phrases 15-20× on
# silence or noise. Collapse any n-gram (2–20 words) that appears more than
# this many consecutive times down to exactly this many occurrences.
_MAX_CONSECUTIVE_PHRASE_REPEATS = 2
_MAX_NGRAM_WORDS = 20

# Silero VAD settings (16 kHz audio, 512-sample / 32ms windows).
# Lower threshold → fewer false negatives (borderline speech passes through).
_VAD_SPEECH_THRESHOLD = 0.4
# Pad detected speech regions on both sides to avoid clipping word edges.
_VAD_PAD_SAMPLES = 1024  # 64 ms at 16 kHz

# ── Dependency detection ────────────────────────────────────────────────────

_VOICE_REQUIRED_MODULES = (
    "kokoro_onnx", "moonshine_onnx", "soundfile", "numpy", "noisereduce",
)
_VOICE_MISSING_MODULES: tuple[str, ...] = ()


def _detect_voice_modules() -> tuple[str, ...]:
    import importlib.util
    return tuple(
        name for name in _VOICE_REQUIRED_MODULES
        if importlib.util.find_spec(name) is None
    )


_VOICE_MISSING_MODULES = _detect_voice_modules()
_VOICE_AVAILABLE = not _VOICE_MISSING_MODULES

_VOICE_INSTALL_HINT = (
    "Voice dependencies are missing from this install. Reinstall Chalie to restore them."
)


def _voice_unavailable_payload() -> "dict[str, object]":
    return {
        "error": "Voice dependencies not installed",
        "reason": "deps_missing",
        "missing": list(_VOICE_MISSING_MODULES),
        "hint": _VOICE_INSTALL_HINT,
    }


def _loading_or_missing_response() -> "Response":
    """503 with a precise ``reason`` and ``Retry-After`` so the client can decide
    whether to auto-retry (transient cold-start) or give up (missing files).

    Combines missing-file and loading-state checks from both Kokoro (via the
    service) and Moonshine (local state).
    """
    tts_service = get_service()
    kokoro_missing = tts_service.missing_model_files()
    moonshine_missing = _missing_moonshine_files()
    missing = kokoro_missing + moonshine_missing
    if missing:
        return make_response(jsonify({
            "error": "Voice models not installed",
            "reason": "models_missing",
            "missing": missing,
            "hint": "Voice models are missing from this install. Reinstall Chalie to restore them.",
        }), 503)
    resp = make_response(jsonify({
        "error": "Models still loading",
        "reason": "loading",
    }), 503)
    resp.headers["Retry-After"] = "3"
    return resp


# ── Lazy model state ────────────────────────────────────────────────────────
#
# Kokoro loading state is owned by VoiceTranscriptService.  Moonshine state
# lives here because STT has its own lock and its loading path is only used by
# the transcribe route.

_moonshine: object = None
_stt_lock = threading.Lock()
_moonshine_load_lock = threading.Lock()
_moonshine_loaded = False
_moonshine_loading = False


def _missing_moonshine_files() -> list[str]:
    """Return the names of Moonshine model files that are missing on disk."""
    expected = [
        _MOONSHINE_DIR / "encoder_model.onnx",
        _MOONSHINE_DIR / "decoder_model_merged.onnx",
    ]
    return [p.name for p in expected if not p.is_file()]


def _ensure_moonshine() -> bool:
    """Lazily load the Moonshine STT model. Returns ``False`` if files are
    missing or loading is in progress (caller should 503)."""
    global _moonshine, _moonshine_loaded, _moonshine_loading

    if _moonshine_loaded:
        return True

    with _moonshine_load_lock:
        if _moonshine_loaded:
            return True
        if _moonshine_loading:
            return False
        _moonshine_loading = True

    try:
        missing = _missing_moonshine_files()
        if missing:
            logger.error("[Voice] Moonshine model files missing: %s", missing)
            with _moonshine_load_lock:
                _moonshine_loading = False
            return False

        import moonshine_onnx as mo

        logger.info("[Voice] Loading Moonshine STT from %s", _MOONSHINE_DIR)
        moonshine = mo.MoonshineOnnxModel(
            models_dir=str(_MOONSHINE_DIR),
            model_name=STT_MODEL_NAME,
        )

        with _moonshine_load_lock:
            _moonshine = moonshine
            _moonshine_loaded = True
            _moonshine_loading = False

        logger.info("[Voice] Moonshine loaded — accepting transcription requests")
        return True

    except Exception as e:
        logger.error("[Voice] Moonshine loading failed: %s", e)
        with _moonshine_load_lock:
            _moonshine_loading = False
        return False


# ── STT audio preprocessing ─────────────────────────────────────────────────

def _dedup_repetitions(text: str) -> str:
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


def _extract_speech(audio: object, sr: int) -> object:
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


def _denoise(audio: object, sr: int) -> object:
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


def _strip_fillers(text: str) -> str:
    if not text:
        return text
    return _MULTI_SPACE_RE.sub(" ", _FILLER_RE.sub("", text)).strip()


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


def _fix_contractions(text: str) -> str:
    """Restore apostrophes that Moonshine drops from common English contractions.

    Ambiguous words whose bare form is independently valid ("were", "well",
    "its", "lets") are deliberately excluded to avoid false corrections.
    Casing of the original token is preserved (lower/title/upper).
    """
    if not text:
        return text
    for pattern, replacement in _CONTRACTION_RES:
        def _recase(m: "re.Match[str]", rep: str = replacement) -> str:
            orig = m.group(0)
            if orig.isupper():
                return rep.upper()
            if orig[0].isupper():
                return rep[0].upper() + rep[1:]
            return rep
        text = pattern.sub(_recase, text)
    return text


# ── WAV helpers ─────────────────────────────────────────────────────────────

def _wav_duration_seconds(data: bytes) -> float:
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


def _transcribe_sync(data: bytes) -> str:
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
        # a batch dim → shape ``[1, N]``. ``mo.transcribe()`` internally calls
        # ``load_audio`` again on its input, which adds a SECOND batch dim and
        # fails the ``[batch, samples]`` assertion. Strip the wrapper before
        # we hand the samples back to transcribe (or slice them).
        audio = mo.load_audio(tmp.name)[0]  # → float32 [N] @ 16 kHz

    import numpy as np
    audio = cast("np.ndarray[tuple[int], np.dtype[np.float32]]", _extract_speech(audio, MOONSHINE_SAMPLE_RATE))
    audio = cast("np.ndarray[tuple[int], np.dtype[np.float32]]", _denoise(audio, MOONSHINE_SAMPLE_RATE))

    if audio.shape[0] < _MIN_CHUNK_SAMPLES:
        return ""

    n_samples = audio.shape[0]
    chunk_size = CHUNK_SECONDS * MOONSHINE_SAMPLE_RATE

    if n_samples <= chunk_size:
        texts = mo.transcribe(audio, model=_moonshine)
        raw = " ".join(t.strip() for t in texts).strip()
        result = _dedup_repetitions(raw)
    else:
        parts: list[str] = []
        for start in range(0, n_samples, chunk_size):
            chunk = audio[start:start + chunk_size]
            if chunk.shape[0] < _MIN_CHUNK_SAMPLES:
                # Trailing fragment shorter than Moonshine's min duration — skip
                # it rather than let the model assert.
                continue
            texts = mo.transcribe(chunk, model=_moonshine)
            part = _dedup_repetitions(" ".join(t.strip() for t in texts).strip())
            if part:
                parts.append(part)
        result = _dedup_repetitions(" ".join(parts).strip())

    result = _strip_fillers(result)
    result = _fix_contractions(result)
    return result


# ── Routes ──────────────────────────────────────────────────────────────────


def _health(
    status: str,
    reason: str | None = None,
    missing: list[str] | None = None,
    hint: str | None = None,
) -> dict[str, object]:
    """Serialize the always-200 readiness probe, dropping absent remediation fields.

    The foundation responder does not exclude ``None``, so the probe is dumped
    here with ``exclude_none`` to keep the ready body exactly ``{"status": "ok"}``.
    """
    return VoiceHealth(
        status=status, reason=reason, missing=missing, hint=hint,
    ).model_dump(exclude_none=True)


@voice_ns.route("/health")
class VoiceHealthResource(Resource):
    @require_auth
    @voice_ns.response(200, "Voice readiness", model=_V["VoiceHealth"])
    @responds(VoiceHealth, code=200)
    def get(self) -> dict[str, object] | ResponseReturnValue:
        """Voice service readiness probe (always 200; readiness is body-encoded)."""
        if not _VOICE_AVAILABLE:
            return _health(
                "unavailable",
                reason="deps_missing",
                missing=list(_VOICE_MISSING_MODULES),
                hint=_VOICE_INSTALL_HINT,
            )

        tts_service = get_service()
        kokoro_loaded = tts_service.is_loaded
        moonshine_loaded = _moonshine_loaded

        if kokoro_loaded and moonshine_loaded:
            return _health("ok")
        if tts_service.is_loading or _moonshine_loading:
            return _health("loading")

        kokoro_missing = tts_service.missing_model_files()
        moonshine_missing = _missing_moonshine_files()
        if kokoro_missing or moonshine_missing:
            return _health(
                "unavailable",
                reason="models_missing",
                missing=kokoro_missing + moonshine_missing,
                hint="Voice models are missing from this install. Reinstall Chalie to restore them.",
            )

        def _start_loading() -> None:
            tts_service.ensure_loaded()
            _ensure_moonshine()

        threading.Thread(target=_start_loading, daemon=True).start()
        return _health("loading")


@voice_ns.route("/transcript/<int:transcript_id>")
class VoiceTranscriptResource(Resource):
    @require_auth
    @voice_ns.produces(["audio/wav"])
    @voice_ns.response(200, "Pre-synthesized speech (audio/wav)")
    @voice_ns.response(202, "Synthesis started or still running", model=_V["VoiceState"])
    @voice_ns.response(404, "No speech is possible for this transcript row", model=_V["Error"])
    @voice_ns.response(409, "Synthesis failed for this transcript row", model=_V["VoiceState"])
    def get(self, transcript_id: int) -> Response | ResponseReturnValue:
        """Play back one transcript row's pre-synthesized speech.

        Reads the stored outcome — it never re-synthesizes a row the pipeline
        already carried to a terminal state. The one exception is a row with no
        outcome at all (history predating pre-synthesis, or a job lost to a
        restart): that starts the pipeline once and answers 202, so there is no
        backfill sweep and no stuck state.
        """
        row = VoiceTranscript.filter("transcript_id", transcript_id).first()

        if row is not None and row.file_path is not None:
            path = FileMapperService.get_voice_path(row.file_path)
            if not path.exists():
                logger.error(
                    "[Voice] transcript %d claims audio at %s but the file is gone",
                    transcript_id, path,
                )
                return VoiceState(state="failed").model_dump(), 409
            wav_bytes = path.read_bytes()
            return Response(
                wav_bytes,
                mimetype="audio/wav",
                headers={
                    "Content-Length": str(len(wav_bytes)),
                    # The audio for a transcript row never changes — the row is
                    # immutable once written, and a deleted row takes the file
                    # with it — so the browser may hold it indefinitely.
                    "Cache-Control": "public, max-age=31536000, immutable",
                },
            )

        if row is not None:
            return VoiceState(state="failed").model_dump(), 409

        transcript = Transcript.get(transcript_id)
        if transcript is None or transcript.role != "assistant" or not transcript.settled:
            return error("No speech for this message", 404)

        threading.Thread(
            target=get_service().synthesize_settled,
            args=(transcript_id,),
            daemon=True,
        ).start()
        return VoiceState(state="pending").model_dump(), 202


@voice_ns.route("/transcribe")
class VoiceTranscribeResource(Resource):
    @require_auth
    @voice_ns.param("file", "WAV audio file to transcribe", _in="formData", type="file")
    @voice_ns.response(200, "Transcribed text", model=_V["Transcription"])
    @voice_ns.response(503, "Voice unavailable (deps or models missing/loading)", model=_V["VoiceUnavailable"])
    @voice_ns.response(422, "Invalid upload (missing/empty/malformed/too-long WAV)", model=_V["Error"])
    @voice_ns.response(500, "Transcription failed", model=_V["Error"])
    @responds(Transcription, code=200)
    def post(self) -> Transcription | ResponseReturnValue:
        """Transcribe a single uploaded WAV file into text."""
        if not _VOICE_AVAILABLE:
            return _voice_unavailable_payload(), 503

        if "file" not in request.files:
            return error("No file uploaded", 422)

        file = request.files["file"]
        data = file.read()

        if not data:
            return error("Empty file", 422)

        duration = _wav_duration_seconds(data)
        if duration <= 0.0:
            return error("Malformed or unsupported WAV file", 422)
        if duration > MAX_AUDIO_SECONDS:
            return error(f"Audio exceeds {MAX_AUDIO_SECONDS}s limit ({duration:.1f}s)", 422)

        if not _ensure_moonshine():
            return _loading_or_missing_response()

        with _stt_lock:
            try:
                return Transcription(text=_transcribe_sync(data))
            except Exception as e:
                logger.error("[Voice] STT error: %s", e)
                return error("Transcription failed", 500)
