"""
Voice blueprint — native STT (Moonshine Voice) + TTS (Kokoro).

Auto-detects voice dependencies at import time. If moonshine_voice or kokoro_onnx
are not installed, all routes return {"status": "unavailable"} / 503.
No Docker required — voice runs in-process.

Models are loaded lazily on first request (not at startup) to avoid blocking
the Flask server while large models download.

TTS architecture: one Kokoro instance, one lock. ``Kokoro.create()`` already
phonemizes, batches by ``MAX_PHONEME_LENGTH``, runs ORT inference, trims
inter-batch silence, and concatenates — so this module is just the Flask
glue (text cleanup → kokoro.create → WAV encode → NDJSON envelope). The
single dtype patch in ``_patch_kokoro_speed_dtype`` works around an upstream
``kokoro-onnx==0.5.0`` int32/float32 mismatch on the HF fp16 export. Drop
the patch when upstream releases a fix.
"""

import io
import logging
import os
import re
import struct
import tempfile
import threading

from flask import Blueprint, Response, request, jsonify

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
# Moonshine STT is single-instance; one user clicks the mic at a time.
# Kokoro is single-instance too — ``create()`` mutates the phonemizer's
# stateful espeak backend and the ORT session is not safe for concurrent
# ``run()`` calls on a shared Kokoro wrapper. Serialise both.
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
        # The library's ``create()`` pre-batches phonemes via
        # ``_split_phonemes()`` so each call here is already under budget.
        # If something upstream ever skips that, fail loudly rather than
        # silently truncating audio.
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise RuntimeError(
                f"Kokoro phoneme overflow: {len(phonemes)} > {MAX_PHONEME_LENGTH}. "
                "Library should have batched via _split_phonemes()."
            )
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
        # natively).
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        logger.debug("[Voice] kokoro batch %.2fs in %.2fs",
                     len(audio) / SAMPLE_RATE, time.time() - start_t)
        return audio, SAMPLE_RATE

    tts._create_audio = types.MethodType(_create_audio_fixed, tts)


def _build_kokoro_instance(model_file: str, voices_file: str):
    """Construct one Kokoro instance with its own ONNX session and dtype patch."""
    from kokoro_onnx import Kokoro
    from services.onnx_session import build_session

    sess = build_session(model_file, log_prefix="[Voice/Kokoro]")
    inst = Kokoro.from_session(sess, voices_file)
    _patch_kokoro_speed_dtype(inst)
    return inst


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
        tts = _build_kokoro_instance(model_file, voices_file)

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


# ── Markdown strip patterns (compiled once) ─────────────────────────────────
#
# Kokoro pronounces literal punctuation, so ``*example*`` is spoken as
# "asterisk example asterisk". The LLM is supposed to emit our HTML subset,
# but markdown leaks through legacy paths, fenced tool output, and quoted
# user input. We strip it unconditionally — gating on "<" + ">" was the
# previous bug (any markdown without HTML tags reached the synthesiser raw).

_MD_FENCE_RE = re.compile(r"```[\w-]*\n?((?:[^`]|`(?!``))*+)```", re.MULTILINE)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_BARE_URL_RE = re.compile(r"https?://\S+")
_MD_BOLD_STAR_RE = re.compile(r"\*\*([^\s*][^*]*?[^\s*]|\S)\*\*")
_MD_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*([^\s*][^*]*?[^\s*]|\S)\*(?!\w)")
_MD_BOLD_UNDER_RE = re.compile(r"__([^\s_][^_]*?[^\s_]|\S)__")
_MD_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_([^\s_][^_]*?[^\s_]|\S)_(?!\w)")
_MD_HEADER_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_MD_BLOCKQUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s?")
_MD_LIST_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_LIST_NUM_RE = re.compile(r"(?m)^\s*\d+\.\s+")
_MD_HRULE_RE = re.compile(r"(?m)^\s*(?:[-*_]\s*){3,}\s*$")


def _strip_markdown(text: str) -> str:
    """Strip markdown markers so they aren't pronounced literally.

    Kokoro speaks raw punctuation. Without this, ``*example*`` becomes
    "asterisk example asterisk" and ``[click](https://x)`` becomes
    "open bracket click close bracket open paren ...".

    Order matters: images before links (``![alt](url)`` shares the link
    syntax), fenced/inline code before star/underscore (so backtick
    contents are preserved as-is), bold before italic (so ``**`` consumes
    before ``*`` matches its own pair).
    """
    if not text:
        return ""
    text = _MD_HRULE_RE.sub("", text)
    text = _MD_FENCE_RE.sub(r"\1", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_IMAGE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_BARE_URL_RE.sub("", text)
    text = _MD_BOLD_STAR_RE.sub(r"\1", text)
    text = _MD_BOLD_UNDER_RE.sub(r"\1", text)
    text = _MD_ITALIC_STAR_RE.sub(r"\1", text)
    text = _MD_ITALIC_UNDER_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_LIST_BULLET_RE.sub("", text)
    text = _MD_LIST_NUM_RE.sub("", text)
    return text


def _clean_for_tts(text: str) -> str:
    """Convert response content to TTS-safe plaintext.

    Pipeline: HTML subset (if present) → markdown strip → whitespace
    collapse. Markdown stripping runs on every input — the LLM emits
    HTML, but markdown still leaks through quoted user content, tool
    output, and legacy paths.
    """
    if not text:
        return ""
    if "<" in text and ">" in text:
        text = extract_plaintext(text)
    text = _strip_markdown(text)
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


def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """Encode a numpy audio array as PCM WAV bytes."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


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
    """Synthesise speech and return the full WAV as a single NDJSON line.

    The response body is two NDJSON lines: one ``{index:0, total:1, audio}``
    payload carrying the entire base64-encoded WAV, then a
    ``{done:true, total:1}`` sentinel so the client can distinguish a
    complete response from a truncated one.

    ``Kokoro.create()`` handles phonemization, batching by
    ``MAX_PHONEME_LENGTH``, ORT inference, per-batch silence trimming, and
    concatenation. We just clean the input text, lock around the call (the
    instance is not thread-safe), and ship the WAV.
    """
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    data = request.get_json(silent=True) or {}
    text = _clean_for_tts((data.get("text") or "").strip())

    if not text:
        return jsonify({"error": "Text is required"}), 400

    logger.info("[Voice] synthesize: len=%d head=%r", len(text), text[:80])

    import base64
    import json
    import time

    try:
        t0 = time.time()
        with _tts_lock:
            samples, sample_rate = _tts_model.create(
                text, voice=KOKORO_VOICE, speed=1.0, lang="en-us",
            )
        wav_bytes = _audio_to_wav_bytes(samples, sample_rate)
        logger.info(
            "[Voice] synthesized %d samples in %.2fs",
            len(samples), time.time() - t0,
        )
    except Exception as e:
        logger.error("[Voice] TTS synthesis failed: %s", e)
        body = (
            json.dumps({"done": True, "total": 0, "error": str(e)}) + "\n"
        )
        return Response(body, mimetype="application/x-ndjson", status=500)

    payload = json.dumps({
        "index": 0,
        "total": 1,
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
    }) + "\n"
    sentinel = json.dumps({"done": True, "total": 1}) + "\n"

    return Response(
        payload + sentinel,
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
