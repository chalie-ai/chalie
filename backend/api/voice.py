"""
Voice blueprint — native STT (faster-whisper) + TTS (KittenTTS).

Auto-detects voice dependencies at import time. If faster-whisper or kittentts
are not installed, all routes return {"status": "unavailable"} / 503.
No Docker required — voice runs in-process.

Models are loaded lazily on first request (not at startup) to avoid blocking
the Flask server while large models download.
"""

import io
import logging
import struct
import tempfile
import threading

from flask import Blueprint, request, jsonify, Response

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

WHISPER_MODEL = "base"
KITTEN_VOICE = "Jasper"
KITTEN_MODEL = "KittenML/kitten-tts-mini-0.8"
MAX_AUDIO_SECONDS = 60
MAX_TTS_CHARS = 5000
STT_CONCURRENCY = 1
TTS_CONCURRENCY = 2

# ── Dependency detection ────────────────────────────────────────────────────

_VOICE_AVAILABLE = False
try:
    import faster_whisper  # noqa: F401
    import kittentts  # noqa: F401
    import soundfile  # noqa: F401
    import numpy  # noqa: F401
    _VOICE_AVAILABLE = True
except ImportError:
    pass

# ── Lazy model state ────────────────────────────────────────────────────────

_stt_model = None
_tts_model = None
_stt_sem = None
_tts_sem = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False


def _ensure_models():
    """Load STT and TTS models on first use. Thread-safe, blocks concurrent loaders."""
    global _stt_model, _tts_model, _stt_sem, _tts_sem, _models_loaded, _models_loading

    if _models_loaded:
        return True

    with _load_lock:
        if _models_loaded:
            return True
        if _models_loading:
            return False  # Another thread is loading — return "loading" status

        _models_loading = True

    # Load outside the lock to avoid blocking other routes
    try:
        logger.info("[Voice] Loading STT model: %s", WHISPER_MODEL)
        from faster_whisper import WhisperModel
        stt = WhisperModel(WHISPER_MODEL, compute_type="int8")

        logger.info("[Voice] Loading TTS model: %s (voice=%s)", KITTEN_MODEL, KITTEN_VOICE)
        from kittentts import KittenTTS
        tts = KittenTTS(KITTEN_MODEL)

        with _load_lock:
            _stt_model = stt
            _tts_model = tts
            _stt_sem = threading.Semaphore(STT_CONCURRENCY)
            _tts_sem = threading.Semaphore(TTS_CONCURRENCY)
            _models_loaded = True
            _models_loading = False

        logger.info("[Voice] Models loaded — accepting requests")
        return True

    except Exception as e:
        logger.error("[Voice] Model loading failed: %s", e)
        with _load_lock:
            _models_loading = False
        return False


# ── Helpers (ported from voice/app.py) ──────────────────────────────────────

def _wav_duration_seconds(data: bytes) -> float:
    """Parse WAV header to get duration without decoding the full file."""
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


def _audio_to_wav_bytes(audio_array, sample_rate: int = 24000) -> bytes:
    """Encode a numpy audio array as PCM WAV bytes."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _transcribe_sync(data: bytes) -> str:
    """Run faster-whisper transcription on raw WAV bytes (blocking)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = _stt_model.transcribe(tmp.name)
        return " ".join(seg.text.strip() for seg in segments).strip()


# ── Routes ──────────────────────────────────────────────────────────────────

@voice_bp.route("/voice/health", methods=["GET"])
def voice_health():
    """Voice service health check."""
    if not _VOICE_AVAILABLE:
        return jsonify({"status": "unavailable"}), 503
    if _models_loaded:
        return jsonify({"status": "ok"}), 200
    if _models_loading:
        return jsonify({"status": "loading"}), 200

    # First health check triggers lazy model loading in background
    thread = threading.Thread(target=_ensure_models, daemon=True)
    thread.start()
    return jsonify({"status": "loading"}), 200


@voice_bp.route("/voice/synthesize", methods=["POST"])
def voice_synthesize():
    """Generate speech from text. Returns binary audio/wav."""
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400
    if len(text) > MAX_TTS_CHARS:
        return jsonify({"error": f"Text exceeds {MAX_TTS_CHARS} character limit"}), 400

    if not _tts_sem.acquire(blocking=False):
        return jsonify({"error": "TTS busy — try again shortly"}), 503

    try:
        import numpy as np
        audio = _tts_model.generate(text, voice=KITTEN_VOICE)
        # Pad with 300ms of silence so the last phoneme isn't clipped at the
        # hardware buffer boundary — a common TTS tail-cutoff artefact.
        silence = np.zeros(int(24000 * 0.3), dtype=audio.dtype)
        audio = np.concatenate([audio, silence])
        wav_bytes = _audio_to_wav_bytes(audio)
        return Response(wav_bytes, mimetype="audio/wav")
    except Exception as e:
        logger.error("[Voice] TTS error: %s", e)
        return jsonify({"error": "TTS generation failed"}), 500
    finally:
        _tts_sem.release()


@voice_bp.route("/voice/transcribe", methods=["POST"])
def voice_transcribe():
    """Transcribe uploaded audio (WAV). Returns {"text": "..."}."""
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    data = file.read()

    if not data:
        return jsonify({"error": "Empty file"}), 400

    duration = _wav_duration_seconds(data)
    if duration > MAX_AUDIO_SECONDS:
        return jsonify({
            "error": f"Audio exceeds {MAX_AUDIO_SECONDS}s limit ({duration:.1f}s)"
        }), 400

    if not _stt_sem.acquire(blocking=False):
        return jsonify({"error": "STT busy — try again shortly"}), 503

    try:
        text = _transcribe_sync(data)
        return jsonify({"text": text})
    except Exception as e:
        logger.error("[Voice] STT error: %s", e)
        return jsonify({"error": "Transcription failed"}), 500
    finally:
        _stt_sem.release()
