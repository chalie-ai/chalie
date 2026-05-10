"""
Voice blueprint — STT + TTS via the unified ``moonshine_voice`` package.

One pip dependency owns G2P, ONNX runtime wiring, voice asset download, and
inference for both directions:

* STT  → ``moonshine_voice.Transcriber`` (Moonshine encoder/decoder)
* TTS  → ``moonshine_voice.TextToSpeech`` (Kokoro voice "am_adam")

If the package is not installed, all routes return ``{"status":"unavailable"}``
or 503. Models are loaded lazily on first request to avoid blocking Flask
while assets download; ``run.py`` fires a daemon thread at boot to warm them
ahead of the first user message.
"""

import io
import logging
import re
import struct
import tempfile
import threading

from flask import Blueprint, Response, request, jsonify

from services.markup import extract_plaintext

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

MOONSHINE_LANG = "en"
TTS_LANG = "en-us"
TTS_VOICE = "kokoro_am_adam"
MAX_AUDIO_SECONDS = 660

# ── Dependency detection ────────────────────────────────────────────────────
#
# We track _which_ voice package is missing so the UI can surface a precise
# install hint instead of a generic "unavailable" — the fresh-install case.
# The shipped requirements-voice.txt covers all three; if any are missing the
# user (or installer) skipped the voice pip step.

_VOICE_REQUIRED_MODULES = ("moonshine_voice", "soundfile", "numpy")
_VOICE_MISSING_MODULES: tuple[str, ...] = ()


def _detect_voice_modules() -> tuple[str, ...]:
    import importlib.util
    missing = []
    for name in _VOICE_REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return tuple(missing)


_VOICE_MISSING_MODULES = _detect_voice_modules()
_VOICE_AVAILABLE = not _VOICE_MISSING_MODULES

# Hint surfaced in /voice/health and route 503s. Same string in both places so
# the frontend has a single canonical install instruction to render.
_VOICE_INSTALL_HINT = (
    "Voice dependencies are not installed. Run "
    "`pip install -r backend/requirements-voice.txt` (or relaunch with "
    "`./run.sh` — failed installs auto-retry on next launch)."
)


def _voice_unavailable_payload() -> dict:
    """Canonical 503 envelope when voice deps are missing."""
    return {
        "error": "Voice dependencies not installed",
        "reason": "deps_missing",
        "missing": list(_VOICE_MISSING_MODULES),
        "hint": _VOICE_INSTALL_HINT,
    }

# ── Lazy model state ────────────────────────────────────────────────────────

_stt_model = None
_tts_model = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False
# Both engines are single-instance: one user clicks the mic at a time, and
# ``TextToSpeech.synthesize()`` is not documented as thread-safe (it shares
# G2P + ORT state under the hood). Serialise both.
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()


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

        logger.info("[Voice] Loading TTS model (lang=%s, voice=%s)", TTS_LANG, TTS_VOICE)
        tts = mv.TextToSpeech(TTS_LANG, voice=TTS_VOICE)

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
# The TTS engine pronounces literal punctuation, so ``*example*`` is spoken as
# "asterisk example asterisk". The LLM is supposed to emit our HTML subset,
# but markdown leaks through legacy paths, fenced tool output, and quoted
# user input. We strip it unconditionally — gating on "<" + ">" was the
# previous bug (any markdown without HTML tags reached the synthesiser raw).

_MD_FENCE_RE = re.compile(r"```[\w-]*\n?((?:[^`]|`(?!``))*+)```", re.MULTILINE)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
# Possessive quantifiers (`++`/`*+`) prevent the regex engine from
# back-tracking inside the bracket and paren bodies, which avoids the
# polynomial-ReDoS path on adversarial inputs like ``[[[[[…``.
_MD_LINK_RE = re.compile(r"\[([^\]]++)\]\([^)]*+\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*+)\]\([^)]*+\)")
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

    The TTS engine speaks raw punctuation. Without this, ``*example*`` becomes
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
    """Voice service health check.

    Returns ``status`` ∈ {``ok``, ``loading``, ``unavailable``} for backwards
    compatibility with the existing frontend poller and integration tests.
    When deps are missing we also include ``reason`` + ``missing`` + ``hint``
    so the UI can show actionable install guidance instead of a silent hide.
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

    ``TextToSpeech.synthesize()`` handles G2P, batching, ORT inference, and
    waveform concatenation. We just clean the input text, lock around the
    call (the instance is not thread-safe), and ship the WAV.
    """
    if not _VOICE_AVAILABLE:
        return jsonify(_voice_unavailable_payload()), 503

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
            samples, sample_rate = _tts_model.synthesize(text)
        wav_bytes = _audio_to_wav_bytes(samples, sample_rate)
        logger.info(
            "[Voice] synthesized %d samples in %.2fs",
            len(samples), time.time() - t0,
        )
    except Exception as e:
        # Log the exception detail server-side; the response body must NOT
        # echo ``str(e)`` because it can carry stack-frame artefacts that
        # leak filesystem paths or library internals to the caller.
        logger.error("[Voice] TTS synthesis failed: %s", e)
        body = (
            json.dumps({"done": True, "total": 0, "error": "TTS synthesis failed"}) + "\n"
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
        return jsonify(_voice_unavailable_payload()), 503

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
