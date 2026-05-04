"""
Voice blueprint — native STT (Moonshine Voice) + TTS (Kokoro).

Auto-detects voice dependencies at import time. If moonshine_voice or kokoro_onnx
are not installed, all routes return {"status": "unavailable"} / 503.
No Docker required — voice runs in-process.

Models are loaded lazily on first request (not at startup) to avoid blocking
the Flask server while large models download.
"""

import io
import logging
import os
import struct
import tempfile
import threading

from flask import Blueprint, Response, request, jsonify, stream_with_context

import paths
from services.markup import extract_plaintext

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

MOONSHINE_LANG = "en"
KOKORO_VOICE = "af_heart"
MAX_AUDIO_SECONDS = 660

# Kokoro model files — downloaded lazily into data/models/kokoro/
_KOKORO_MODEL_DIR = str(paths.MODELS_DIR / "kokoro")
_KOKORO_MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx"
_KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# ── Dependency detection ────────────────────────────────────────────────────

_VOICE_AVAILABLE = False
try:
    import moonshine_voice  # noqa: F401
    import kokoro_onnx  # noqa: F401
    import soundfile  # noqa: F401
    import numpy  # noqa: F401
    _VOICE_AVAILABLE = True
except ImportError:
    pass

# ── Lazy model state ────────────────────────────────────────────────────────

_stt_model = None
_tts_model = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False
# Serialises access to the shared Moonshine / Kokoro ONNX sessions. The model
# is shared across request threads; these locks prevent two simultaneous
# requests from corrupting session state.
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()


def _download_kokoro_models():
    """Download Kokoro model files if not present."""
    import requests

    os.makedirs(_KOKORO_MODEL_DIR, exist_ok=True)

    model_path = os.path.join(_KOKORO_MODEL_DIR, "kokoro-v1.0-fp16.onnx")
    voices_path = os.path.join(_KOKORO_MODEL_DIR, "voices-v1.0.bin")

    for url, path in [(_KOKORO_MODEL_URL, model_path), (_KOKORO_VOICES_URL, voices_path)]:
        if os.path.exists(path):
            continue
        fname = os.path.basename(path)
        logger.info("[Voice] Downloading %s …", fname)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        os.rename(tmp_path, path)
        logger.info("[Voice] Downloaded %s (%.1f MB)", fname, os.path.getsize(path) / 1e6)

    return model_path, voices_path


def _patch_kokoro_speed_dtype(tts):
    """Fix upstream ``kokoro-onnx==0.5.0`` dtype bug for the HF fp16 schema.

    The library's ``_create_audio`` hardcodes ``speed: int32`` when the model
    exposes an ``input_ids`` input (the newer
    ``onnx-community/Kokoro-82M-v1.0-ONNX`` export), but the fp16 graph
    actually expects ``speed: float32``. ONNXRuntime then rejects every
    inference call with::

        Unexpected input data type. Actual: (tensor(int32)),
        expected: (tensor(float))

    We override the bound method on the loaded instance with a corrected
    version so we don't have to vendor or fork the library. Drop this once
    upstream releases a fix.
    """
    import time
    import types
    import numpy as np
    from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE

    def _create_audio_fixed(self, phonemes, voice, speed):
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
        start_t = time.time()
        tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
        assert len(tokens) <= MAX_PHONEME_LENGTH
        voice = voice[len(tokens)]
        tokens = [[0, *tokens, 0]]
        if "input_ids" in [i.name for i in self.sess.get_inputs()]:
            inputs = {
                "input_ids": tokens,
                "style": np.array(voice, dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),  # was int32 — upstream bug
            }
        else:
            inputs = {
                "tokens": tokens,
                "style": voice,
                "speed": np.ones(1, dtype=np.float32) * speed,
            }
        audio = self.sess.run(None, inputs)[0]
        # The HF fp16 export returns shape (1, N) float16; kokoro-onnx's
        # downstream concatenation + soundfile encoding both expect a 1-D
        # float32 waveform (the legacy thewh1teagle fp32 model returned that
        # natively). Squeeze and cast so existing chunk-stitching keeps working.
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        logger.debug("[Voice] kokoro chunk %.2fs in %.2fs",
                     len(audio) / SAMPLE_RATE, time.time() - start_t)
        return audio, SAMPLE_RATE

    tts._create_audio = types.MethodType(_create_audio_fixed, tts)


def _ensure_models():
    """Load STT and TTS models on first use. Thread-safe, blocks concurrent loaders."""
    global _stt_model, _tts_model, _models_loaded, _models_loading

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
        import moonshine_voice as mv

        logger.info("[Voice] Loading STT model (Moonshine, lang=%s)", MOONSHINE_LANG)
        model_path, model_arch = mv.get_model_for_language(MOONSHINE_LANG)
        stt = mv.Transcriber(model_path=model_path, model_arch=model_arch)

        logger.info("[Voice] Loading TTS model (Kokoro, voice=%s)", KOKORO_VOICE)
        model_file, voices_file = _download_kokoro_models()
        from kokoro_onnx import Kokoro
        from services.onnx_session import build_session

        # Kokoro's default __init__ hardcodes CPUExecutionProvider. The installer
        # lands the right wheel (CPU/CUDA/ROCm) for the host; build_session picks
        # the matching EP and falls back to CPU if session construction fails.
        kokoro_sess = build_session(model_file, log_prefix="[Voice/Kokoro]")
        tts = Kokoro.from_session(kokoro_sess, voices_file)
        _patch_kokoro_speed_dtype(tts)

        with _load_lock:
            _stt_model = stt
            _tts_model = tts
            _models_loaded = True
            _models_loading = False

        logger.info("[Voice] Models loaded — accepting requests")
        return True

    except Exception as e:
        logger.error("[Voice] Model loading failed: %s", e)
        with _load_lock:
            _models_loading = False
        return False


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def _clean_for_tts(text: str) -> str:
    """Convert response content to TTS-safe plaintext.

    Inputs may be raw plaintext (legacy callers) or XML markup. Both produce
    a clean speakable string. XML tags are stripped, <actions> contents
    dropped entirely, <img> alt text used.
    """
    if not text:
        return ""
    if "<" in text and ">" in text:
        return extract_plaintext(text)
    return " ".join(text.split())


def _transcribe_sync(data: bytes) -> str:
    """Run Moonshine transcription on raw WAV bytes (blocking)."""
    import moonshine_voice as mv

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        audio_data, sample_rate = mv.load_wav_file(tmp.name)
        transcript = _stt_model.transcribe_without_streaming(audio_data, sample_rate=sample_rate)
        return " ".join(line.text.strip() for line in transcript.lines).strip()


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


def _audio_to_wav_bytes(audio_array, sample_rate: int = 24000) -> bytes:
    """Encode a numpy audio array as PCM WAV bytes."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@voice_bp.route("/voice/synthesize", methods=["POST"])
def voice_synthesize():
    """Synthesise speech and return the full WAV as a single NDJSON line.

    The response body is two NDJSON lines: one ``{index:0, total:1, audio}``
    payload carrying the entire base64-encoded WAV, then a
    ``{done:true, total:1}`` sentinel so the client can distinguish a
    complete response from a truncated one.

    We synthesise the whole message in a single Kokoro call — Kokoro's
    internal phoneme batching (MAX_PHONEME_LENGTH=510) handles long text
    safely. The previous sentence-chunked streaming caused mid-message
    gaps whenever synthesis ran slower than playback: the client would
    stall between chunks, then resume awkwardly. One blob, one playback,
    no gaps.
    """
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    data = request.get_json(silent=True) or {}
    text = _clean_for_tts((data.get("text") or "").strip())

    if not text:
        return jsonify({"error": "Text is required"}), 400

    @stream_with_context
    def stream():
        import base64
        import json

        with _tts_lock:
            try:
                samples, _sr = _tts_model.create(
                    text, voice=KOKORO_VOICE, lang="en-us"
                )
            except Exception as e:
                logger.error("[Voice] TTS synthesis failed: %s", e)
                yield json.dumps({"done": True, "total": 0}) + "\n"
                return
            wav_bytes = _audio_to_wav_bytes(samples)
            yield json.dumps({
                "index": 0,
                "total": 1,
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
            }) + "\n"
        yield json.dumps({"done": True, "total": 1}) + "\n"

    return Response(
        stream(),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


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

    with _stt_lock:
        try:
            text = _transcribe_sync(data)
            return jsonify({"text": text})
        except Exception as e:
            logger.error("[Voice] STT error: %s", e)
            return jsonify({"error": "Transcription failed"}), 500
