"""
Voice blueprint — STT + TTS via two single-purpose ONNX libraries.

* TTS → ``kokoro_onnx.Kokoro`` (Kokoro v1.0 + espeak-ng phonemizer)
* STT → ``moonshine_onnx.MoonshineOnnxModel`` (Moonshine base, ONNX)

Both ONNX model files ship with the install — ``installer/install.sh`` writes
them to ``resources/voice-models/`` at install time, so the first request does
not have to wait on a network download. If the deps or files are missing every
route returns ``{"status":"unavailable"}`` or 503 with a precise hint.

Kokoro uses phonemizer-fork + espeakng-loader under the hood, which preserves
punctuation as IPA pause tokens and handles numbers, abbreviations, and 2-3
letter acronyms natively. The TTS-side text pipeline collapses to:
    markdown/HTML → plaintext → URL → spoken-host → whitespace collapse.

The synthesize route returns a single ``audio/wav`` blob — no streaming, no
NDJSON, no per-sentence segmentation. Kokoro emits one coherent waveform with
native prosody from a single ``create()`` call.
"""

import io
import logging
import re
import struct
import tempfile
import threading
from pathlib import Path

from flask import Blueprint, Response, request, jsonify

from markdown_it import MarkdownIt

from services.markup import extract_plaintext

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

# Models are baked into the install by installer/install.sh into a directory
# sibling to backend/, intentionally OUTSIDE the data/ volume so the files
# travel with the image and survive `chalie update` on native installs.
_VOICE_ROOT = Path(__file__).resolve().parents[2] / "resources" / "voice-models"
_KOKORO_MODEL = _VOICE_ROOT / "kokoro" / "kokoro-v1.0.onnx"
_KOKORO_VOICES = _VOICE_ROOT / "kokoro" / "voices-v1.0.bin"
_MOONSHINE_DIR = _VOICE_ROOT / "moonshine" / "base"

TTS_VOICE = "af_heart"
TTS_LANG = "en-us"
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

# ── Dependency detection ────────────────────────────────────────────────────

_VOICE_REQUIRED_MODULES = ("kokoro_onnx", "moonshine_onnx", "soundfile", "numpy")
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
    "Voice dependencies are not installed. Run "
    "`pip install -r backend/requirements-voice.txt` (or relaunch with "
    "`./run.sh` — failed installs auto-retry on next launch)."
)


def _voice_unavailable_payload() -> dict:
    return {
        "error": "Voice dependencies not installed",
        "reason": "deps_missing",
        "missing": list(_VOICE_MISSING_MODULES),
        "hint": _VOICE_INSTALL_HINT,
    }


def _loading_or_missing_response():
    """503 with a precise ``reason`` and ``Retry-After`` so the client can decide
    whether to auto-retry (transient cold-start) or give up (missing files)."""
    missing = _missing_model_files()
    if missing:
        return jsonify({
            "error": "Voice models not installed",
            "reason": "models_missing",
            "missing": missing,
            "hint": "Re-run installer to download voice models.",
        }), 503
    return jsonify({
        "error": "Models still loading",
        "reason": "loading",
    }), 503, {"Retry-After": "3"}


def _missing_model_files() -> list[str]:
    expected = [
        _KOKORO_MODEL,
        _KOKORO_VOICES,
        _MOONSHINE_DIR / "encoder_model.onnx",
        _MOONSHINE_DIR / "decoder_model_merged.onnx",
    ]
    return [str(p) for p in expected if not p.is_file()]


# ── Lazy model state ────────────────────────────────────────────────────────

_kokoro = None
_moonshine = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False

# phonemizer-fork (espeak-ng under the hood) is not thread-safe; kokoro.create()
# calls it on every invocation. Serialise the full TTS path. STT shares its own
# lock so a long synthesis does not block a mic recording from getting
# transcribed.
_tts_lock = threading.Lock()
_stt_lock = threading.Lock()


def _ensure_models():
    """Load Kokoro + Moonshine on first use. Thread-safe."""
    global _kokoro, _moonshine, _models_loaded, _models_loading

    if _models_loaded:
        return True

    with _load_lock:
        if _models_loaded:
            return True
        if _models_loading:
            return False
        _models_loading = True

    try:
        missing = _missing_model_files()
        if missing:
            logger.error("[Voice] Model files missing: %s", missing)
            with _load_lock:
                _models_loading = False
            return False

        from kokoro_onnx import Kokoro
        import moonshine_onnx as mo

        logger.info("[Voice] Loading Kokoro TTS from %s", _KOKORO_MODEL)
        kokoro = Kokoro(str(_KOKORO_MODEL), str(_KOKORO_VOICES))

        logger.info("[Voice] Loading Moonshine STT from %s", _MOONSHINE_DIR)
        moonshine = mo.MoonshineOnnxModel(
            models_dir=str(_MOONSHINE_DIR),
            model_name=STT_MODEL_NAME,
        )

        with _load_lock:
            _kokoro = kokoro
            _moonshine = moonshine
            _models_loaded = True
            _models_loading = False

        logger.info("[Voice] Models loaded — accepting requests")
        return True

    except Exception as e:
        logger.error("[Voice] Model loading failed: %s", e)
        with _load_lock:
            _models_loading = False
        return False


# ── TTS text preprocessing ──────────────────────────────────────────────────
#
# Kokoro phonemises via phonemizer-fork (espeak-ng), which:
#   * preserves punctuation as IPA pause tokens (commas, periods, question
#     marks all produce natural prosody breaks)
#   * normalises numbers ("v0.8" reads as "vee zero point eight")
#   * pronounces 2-3 letter acronyms correctly
#   * handles abbreviations and ordinals natively
#
# So we only need to (a) strip HTML/markdown markers so the LLM's raw output
# doesn't read tags or symbols aloud, (b) rewrite bare URLs to a human-readable
# host form (otherwise espeak spells out every slash and digit), (c) collapse
# whitespace.

_md = MarkdownIt("commonmark", {"breaks": True, "html": False})
_URL_RE = re.compile(r"https?://([^\s/?#]+)\S*")
_WS_RE = re.compile(r"\s+")


def _spoken_url(match: "re.Match[str]") -> str:
    """Rewrite ``http://google.com/123`` → ``google dot com``.

    Without this, espeak reads URLs character-by-character ("h t t p
    slash slash google dot com slash one two three") — fast but unpleasant.
    Stripping protocol + path and verbalising dots produces a natural read.
    """
    host = match.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host.replace(".", " dot ")


def _clean_for_tts(text: str) -> str:
    """Markdown/HTML → plaintext; rewrite URLs; collapse whitespace."""
    if not text:
        return ""
    # HTML pre-pass — strip LLM-emitted tags before markdown-it sees them so
    # `<p>foo</p>` doesn't survive as literal "less-than p greater-than".
    if "<" in text and ">" in text:
        text = extract_plaintext(text)
    plain = extract_plaintext(_md.render(text))
    plain = _URL_RE.sub(_spoken_url, plain)
    return _WS_RE.sub(" ", plain).strip()


# ── WAV helpers ─────────────────────────────────────────────────────────────

def _wav_duration_seconds(data: bytes) -> float:
    """Parse a WAV header for duration without decoding the audio payload."""
    try:
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return 0.0
        channels = struct.unpack_from("<H", data, 22)[0]
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]
        if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
            return 0.0
        offset = 12
        while offset < len(data) - 8:
            chunk_id = data[offset:offset + 4]
            chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
            if chunk_id == b"data":
                bytes_per_sample = bits_per_sample // 8
                return chunk_size / (sample_rate * channels * bytes_per_sample)
            offset += 8 + chunk_size
        return 0.0
    except Exception:
        return 0.0


def _audio_to_wav_bytes(samples, sample_rate: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


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

    n_samples = audio.shape[0]
    chunk_size = CHUNK_SECONDS * MOONSHINE_SAMPLE_RATE

    if n_samples <= chunk_size:
        texts = mo.transcribe(audio, model=_moonshine)
        return " ".join(t.strip() for t in texts).strip()

    parts: list[str] = []
    for start in range(0, n_samples, chunk_size):
        chunk = audio[start:start + chunk_size]
        if chunk.shape[0] < _MIN_CHUNK_SAMPLES:
            # Trailing fragment shorter than Moonshine's min duration — skip
            # it rather than let the model assert.
            continue
        texts = mo.transcribe(chunk, model=_moonshine)
        part = " ".join(t.strip() for t in texts).strip()
        if part:
            parts.append(part)
    return " ".join(parts).strip()


# ── Routes ──────────────────────────────────────────────────────────────────

@voice_bp.route("/voice/health", methods=["GET"])
def voice_health():
    """Voice service health check.

    Returns ``status`` ∈ {``ok``, ``loading``, ``unavailable``}. When models
    or deps are missing the response also carries ``reason`` + ``missing`` +
    ``hint`` so the UI can surface actionable install guidance.
    """
    if not _VOICE_AVAILABLE:
        return jsonify({
            "status": "unavailable",
            "reason": "deps_missing",
            "missing": list(_VOICE_MISSING_MODULES),
            "hint": _VOICE_INSTALL_HINT,
        }), 200

    if _models_loaded:
        return jsonify({"status": "ok"}), 200
    if _models_loading:
        return jsonify({"status": "loading"}), 200

    missing_files = _missing_model_files()
    if missing_files:
        return jsonify({
            "status": "unavailable",
            "reason": "models_missing",
            "missing": missing_files,
            "hint": "Re-run installer to download voice models.",
        }), 200

    threading.Thread(target=_ensure_models, daemon=True).start()
    return jsonify({"status": "loading"}), 200


@voice_bp.route("/voice/synthesize", methods=["POST"])
def voice_synthesize():
    """Synthesise speech and return a single WAV blob."""
    if not _VOICE_AVAILABLE:
        return jsonify(_voice_unavailable_payload()), 503

    if not _ensure_models():
        return _loading_or_missing_response()

    data = request.get_json(silent=True) or {}
    text = _clean_for_tts((data.get("text") or "").strip())

    if not text:
        return jsonify({"error": "Text is required"}), 400

    logger.info("[Voice] synthesize: len=%d head=%r", len(text), text[:80])

    try:
        with _tts_lock:
            samples, sr = _kokoro.create(
                text, voice=TTS_VOICE, speed=1.0, lang=TTS_LANG,
            )
    except Exception as e:
        logger.error("[Voice] TTS synthesis failed: %s", e)
        return jsonify({"error": "TTS synthesis failed"}), 500

    wav_bytes = _audio_to_wav_bytes(samples, sr)
    return Response(
        wav_bytes,
        mimetype="audio/wav",
        headers={
            "Content-Length": str(len(wav_bytes)),
            "Cache-Control": "no-cache",
        },
    )


@voice_bp.route("/voice/transcribe", methods=["POST"])
def voice_transcribe():
    """Transcribe uploaded audio (WAV). Returns ``{"text": "..."}``."""
    if not _VOICE_AVAILABLE:
        return jsonify(_voice_unavailable_payload()), 503

    # Validate the request shape BEFORE waiting on model load — a malformed
    # upload should get a 400 instantly instead of stalling on cold-start
    # phonemizer/ONNX initialisation.
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    data = file.read()

    if not data:
        return jsonify({"error": "Empty file"}), 400

    duration = _wav_duration_seconds(data)
    if duration <= 0.0:
        # _wav_duration_seconds returns 0.0 on malformed/non-WAV input.
        # Reject before we hand the bytes to Moonshine so the user gets a
        # clear error instead of a downstream load_audio crash.
        return jsonify({"error": "Malformed or unsupported WAV file"}), 400
    if duration > MAX_AUDIO_SECONDS:
        return jsonify({
            "error": f"Audio exceeds {MAX_AUDIO_SECONDS}s limit ({duration:.1f}s)"
        }), 400

    if not _ensure_models():
        return _loading_or_missing_response()

    with _stt_lock:
        try:
            text = _transcribe_sync(data)
            return jsonify({"text": text})
        except Exception as e:
            logger.error("[Voice] STT error: %s", e)
            return jsonify({"error": "Transcription failed"}), 500
