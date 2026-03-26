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
import re
import struct
import tempfile
import threading

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

MOONSHINE_LANG = "en"
KOKORO_VOICE = "af_heart"
MAX_AUDIO_SECONDS = 660
STT_CONCURRENCY = 1
TTS_CONCURRENCY = 2

# Kokoro model files — downloaded lazily into backend/data/models/kokoro/
_KOKORO_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "models", "kokoro",
)
_KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
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
_stt_sem = None
_tts_sem = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False


def _download_kokoro_models():
    """Download Kokoro model files if not present."""
    import requests

    os.makedirs(_KOKORO_MODEL_DIR, exist_ok=True)

    model_path = os.path.join(_KOKORO_MODEL_DIR, "kokoro-v1.0.onnx")
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
        import moonshine_voice as mv

        logger.info("[Voice] Loading STT model (Moonshine, lang=%s)", MOONSHINE_LANG)
        model_path, model_arch = mv.get_model_for_language(MOONSHINE_LANG)
        stt = mv.Transcriber(model_path=model_path, model_arch=model_arch)

        logger.info("[Voice] Loading TTS model (Kokoro, voice=%s)", KOKORO_VOICE)
        model_file, voices_file = _download_kokoro_models()
        from kokoro_onnx import Kokoro
        tts = Kokoro(model_file, voices_file)

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
    """Convert markdown to natural spoken text for TTS synthesis.

    Transforms structured markdown into fluid speech while preserving meaning:
    - Code blocks  → "See code in chat message."
    - Numbered lists → kept numbered as sentences
    - Bullet lists  → individual sentences
    - Tables        → row-by-row natural reading
    - Headers       → spoken as topic sentences
    - Links         → label only, URLs dropped
    """
    # ── Phase 1: Block-level replacements (before line processing) ──────

    # Code blocks → spoken placeholder
    text = re.sub(r"```[\s\S]*?```", "\nSee code in chat message.\n", text)

    # Inline code — keep the content, drop backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove markdown emphasis — keep the content
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)

    # Markdown links — keep the label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Bare URLs — not speakable
    text = re.sub(r"https?://\S+", "", text)

    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # ── Phase 2: Tables → natural row reading ───────────────────────────

    def _speak_table(match: re.Match) -> str:
        """Convert a markdown table to speakable row descriptions."""
        table_text = match.group(0)
        rows = [r.strip() for r in table_text.strip().split("\n") if r.strip()]
        # Extract header row
        headers = [c.strip() for c in rows[0].strip("|").split("|")] if rows else []
        # Skip separator row (---|---) and process data rows
        spoken = []
        for row in rows[1:]:
            if re.match(r"^\|?[\s:|-]+\|?$", row):
                continue  # separator row
            cells = [c.strip() for c in row.strip("|").split("|")]
            parts = []
            for i, cell in enumerate(cells):
                if not cell:
                    continue
                if i < len(headers) and headers[i]:
                    parts.append(f"{headers[i]}: {cell}")
                else:
                    parts.append(cell)
            if parts:
                spoken.append(", ".join(parts) + ".")
        return "\n".join(spoken) if spoken else ""

    text = re.sub(
        r"(?:^\|.+\|$\n?){2,}",
        _speak_table,
        text,
        flags=re.MULTILINE,
    )

    # ── Phase 3: Line-by-line processing ────────────────────────────────

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Headers → spoken as topic intro
        header_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if header_match:
            content = header_match.group(1).rstrip(".!?")
            lines.append(content + ".")
            continue

        # Blockquote markers — strip, keep content
        line = re.sub(r"^>\s?", "", line)

        # Numbered list items — keep the number, ensure sentence ending
        num_match = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if num_match:
            num, content = num_match.group(1), num_match.group(2)
            content = content.rstrip(".!?,;:")
            lines.append(f"{num}, {content}.")
            continue

        # Bullet list items — convert to sentence
        bullet_match = re.match(r"^[-*+]\s+(.+)$", line)
        if bullet_match:
            content = bullet_match.group(1)
            if content and content[-1] not in ".!?":
                content += "."
            lines.append(content)
            continue

        # Regular line — ensure sentence ending
        if line and line[-1] not in ".!?,;:":
            line += "."
        lines.append(line)

    text = " ".join(lines)

    # ── Phase 4: Final cleanup ──────────────────────────────────────────

    text = re.sub(r"\.{2,}", ".", text)           # repeated periods
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)  # space before punctuation
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_CHUNK_CHARS = 400


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized chunks safe for the ONNX TTS model."""
    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > _MAX_CHUNK_CHARS:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [text]


def _audio_to_wav_bytes(audio_array, sample_rate: int = 24000) -> bytes:
    """Encode a numpy audio array as PCM WAV bytes."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


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


@voice_bp.route("/voice/synthesize", methods=["POST"])
def voice_synthesize():
    """Generate speech from text.

    Synthesizes sentence-by-sentence in a background thread and pushes each
    WAV chunk as a ``tts_chunk`` event through the ``output:events`` pub/sub
    channel (picked up by the WebSocket and forwarded to the client).

    Returns immediately with ``{"ok": true, "total": N}`` so the caller knows
    how many chunks to expect.
    """
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Clean markdown → natural spoken text
    text = _clean_for_tts(text)
    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunks = _split_sentences(text)
    total = len(chunks)

    if not _tts_sem.acquire(blocking=False):
        return jsonify({"error": "TTS busy — try again shortly"}), 503

    def _synthesize_and_push():
        """Synthesize each sentence and publish WAV chunks via pub/sub."""
        import base64
        import json
        import numpy as np
        from services.memory_store import get_shared_store

        store = get_shared_store()
        silence = np.zeros(int(24000 * 0.3), dtype=np.float32)
        try:
            for i, chunk in enumerate(chunks):
                try:
                    samples, _sr = _tts_model.create(
                        chunk, voice=KOKORO_VOICE, lang="en-us"
                    )
                    # Inter-sentence pause (skip on last chunk)
                    if i < len(chunks) - 1:
                        samples = np.concatenate([samples, silence])
                    wav_bytes = _audio_to_wav_bytes(samples)
                    store.publish("output:events", json.dumps({
                        "type": "tts_chunk",
                        "index": i,
                        "total": total,
                        "text": chunk,
                        "audio": base64.b64encode(wav_bytes).decode("ascii"),
                    }))
                except Exception as chunk_err:
                    logger.warning("[Voice] TTS chunk failed (%d chars): %s — text: %.80s",
                                   len(chunk), chunk_err, chunk)
                    continue
            # End-of-stream marker
            store.publish("output:events", json.dumps({
                "type": "tts_done",
            }))
        finally:
            _tts_sem.release()

    thread = threading.Thread(target=_synthesize_and_push, daemon=True)
    thread.start()

    return jsonify({"ok": True, "total": total})


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
