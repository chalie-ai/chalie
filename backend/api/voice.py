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
import queue
import re
import struct
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, Response, request, jsonify, stream_with_context

import paths
from services.markup import extract_plaintext

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

MOONSHINE_LANG = "en"
KOKORO_VOICE = "af_heart"
MAX_AUDIO_SECONDS = 660

# Number of Kokoro instances loaded at boot, dispatched via a queue checkout.
# Long messages chunk into N pieces; with a pool of 3, the first 3 chunks
# synthesise concurrently. Memory cost is ~3× the model RAM (~600MB total
# for fp16 Kokoro). Each instance owns its own ONNX session — sharing one
# session across threads is documented-safe at the ORT layer but the
# Kokoro wrapper above it is not, and the original lock comment claims
# state corruption was observed under concurrent ``create()`` calls.
TTS_POOL_SIZE = 3

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
_tts_pool: "queue.Queue" = queue.Queue()
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False
# Moonshine STT is single-instance; one user clicks the mic at a time.
# Kokoro is concurrent via the pool — no global lock, instances are
# checked out from ``_tts_pool`` and returned after each synthesis.
_stt_lock = threading.Lock()
# phonemizer's EspeakBackend is cached in a process-global dict and carries
# mutable instance state (_count_txt / _count_phn).  Concurrent threads
# clobber that state, producing "number of lines in input and output must be
# equal" errors on multi-chunk requests.  Serialise the phonemize step only;
# the ONNX inference step is genuinely parallel-safe per-instance.
_phonemize_lock = threading.Lock()

# Per-chunk retry budget. Phonemizer occasionally returns truncated/empty
# phonemes on chunks dominated by punctuation, em-dashes, or stage-direction
# brackets — Kokoro then produces near-silent audio. One retry is cheap and
# clears most transient cases (e.g., the espeak backend's stateful counters
# self-correcting after a previous mismatch). Persistent failures fall back
# to a short pad of silence so the rest of the message still plays.
_TTS_MAX_ATTEMPTS = 2
# Below ~50ms of audio (24kHz × 0.05) the chunk is effectively silent — the
# user hears a gap. Treat as failure and retry.
_TTS_MIN_SAMPLES = 1200
_TTS_FALLBACK_SAMPLES = 2400  # 0.1s of silence preserves chunk ordering


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
        # Defense in depth: callers must pre-chunk text so each segment
        # phonemizes within MAX_PHONEME_LENGTH. The previous version
        # silently truncated overlong phoneme strings, dropping audio for
        # any chunk Kokoro's internal segmenter handed us over budget.
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise RuntimeError(
                f"Kokoro phoneme overflow: {len(phonemes)} > {MAX_PHONEME_LENGTH}. "
                "Caller must pre-chunk text before synthesis."
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
        # natively). Squeeze and cast so existing chunk-stitching keeps working.
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        logger.debug("[Voice] kokoro chunk %.2fs in %.2fs",
                     len(audio) / SAMPLE_RATE, time.time() - start_t)
        return audio, SAMPLE_RATE

    tts._create_audio = types.MethodType(_create_audio_fixed, tts)


def _build_kokoro_instance(model_file: str, voices_file: str):
    """Construct one Kokoro instance with its own ONNX session and patch."""
    from kokoro_onnx import Kokoro
    from services.onnx_session import build_session

    sess = build_session(model_file, log_prefix="[Voice/Kokoro]")
    inst = Kokoro.from_session(sess, voices_file)
    _patch_kokoro_speed_dtype(inst)
    return inst


def _ensure_models():
    """Load STT and TTS models on first use. Thread-safe, blocks concurrent loaders."""
    global _stt_model, _models_loaded, _models_loading

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

        logger.info(
            "[Voice] Loading TTS pool (Kokoro x%d, voice=%s)",
            TTS_POOL_SIZE, KOKORO_VOICE,
        )
        model_file, voices_file = _download_kokoro_models()
        # Build N independent Kokoro instances in parallel. Each has its
        # own ONNX session (~200MB) — the cost is RAM, the gain is true
        # concurrent synthesis without serialising on a shared wrapper.
        with ThreadPoolExecutor(max_workers=TTS_POOL_SIZE) as pool:
            instances = list(pool.map(
                lambda _: _build_kokoro_instance(model_file, voices_file),
                range(TTS_POOL_SIZE),
            ))

        with _load_lock:
            _stt_model = stt
            for inst in instances:
                _tts_pool.put(inst)
            _models_loaded = True
            _models_loading = False

        logger.info(
            "[Voice] Models loaded — TTS pool size=%d, accepting requests",
            _tts_pool.qsize(),
        )
        return True

    except Exception as e:
        logger.error("[Voice] Model loading failed: %s", e)
        with _load_lock:
            _models_loading = False
        return False


def _synthesise_chunk(chunk: str):
    """Synthesise one text chunk on a pooled Kokoro instance.

    Two-phase execution:
    1. Phonemize under ``_phonemize_lock`` — espeak's process-global
       EspeakBackend cache has mutable instance state that is not
       thread-safe; serialising this step eliminates the race.
    2. ONNX inference outside the lock — each pooled Kokoro instance
       owns its own ORT session; inference is genuinely parallel.

    Checks an instance out of ``_tts_pool`` for the inference step and
    always returns it to the pool, even on exception.
    """
    import numpy as np

    inst = _tts_pool.get()
    try:
        with _phonemize_lock:
            phonemes = inst.tokenizer.phonemize(chunk, lang="en-us")
        voice_style = inst.get_voice_style(KOKORO_VOICE)
        samples, _sr = inst._create_audio(phonemes, voice_style, speed=1.0)
        return np.asarray(samples, dtype=np.float32).reshape(-1)
    finally:
        _tts_pool.put(inst)


def _synthesise_chunk_safe(idx_chunk):
    """Synthesise one chunk with retry-on-failure and retry-on-silent.

    A bare ``_synthesise_chunk`` call can raise (phonemizer mismatch) or
    return effectively-empty audio (chunk dominated by punctuation that
    phonemizes to nothing). Both used to either kill the whole synthesis
    via ``pool.map``'s exception propagation, or land in the concat as a
    silent gap with no log. We now retry once and, if still bad, fall
    back to a short pad of silence so neighbouring chunks still play.

    Takes ``(idx, chunk)`` so log lines can identify the offending chunk
    by position; returns the audio array. Order is preserved by the
    caller using ``pool.map`` over the enumerated list.
    """
    import numpy as np

    idx, chunk = idx_chunk
    for attempt in range(1, _TTS_MAX_ATTEMPTS + 1):
        try:
            audio = _synthesise_chunk(chunk)
        except Exception as exc:
            logger.warning(
                "[Voice] chunk %d raised on attempt %d/%d (%s) head=%r — retrying",
                idx, attempt, _TTS_MAX_ATTEMPTS, exc, chunk[:60],
            )
            continue
        if audio.size >= _TTS_MIN_SAMPLES:
            return audio
        logger.warning(
            "[Voice] chunk %d empty on attempt %d/%d (samples=%d) head=%r — retrying",
            idx, attempt, _TTS_MAX_ATTEMPTS, int(audio.size), chunk[:60],
        )
    logger.error(
        "[Voice] chunk %d permanently silent after %d attempts — emitting fallback silence (head=%r)",
        idx, _TTS_MAX_ATTEMPTS, chunk[:60],
    )
    return np.zeros(_TTS_FALLBACK_SAMPLES, dtype=np.float32)


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


# ── TTS chunking ────────────────────────────────────────────────────────────
#
# Kokoro's ONNX graph caps phoneme input at 510 tokens per call. Its public
# ``create()`` segments on punctuation, but a long un-punctuated stretch
# still produces an over-budget chunk that the model rejects (or, before
# the patch hardened, silently clipped). We pre-segment here so every chunk
# Kokoro sees is comfortably under budget, then concatenate the resulting
# float32 audio arrays into a single WAV — preserving the no-gap
# single-blob delivery contract.
#
# Budget is in characters, not phonemes, because phonemizing twice (once
# to size, once to synthesise) is wasted work. 400 chars maps to roughly
# 320–450 phonemes for typical English prose, leaving headroom under 510.

_TTS_CHUNK_TARGET = 400
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def _word_pack(text: str, target: int = _TTS_CHUNK_TARGET) -> list[str]:
    """Pack words into chunks <= target chars. Hard-cuts any single token
    larger than target — the only case where we slice mid-content, and
    only for pathological input (no whitespace in 400 chars)."""
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in text.split():
        while len(word) > target:
            if buf:
                out.append(" ".join(buf))
                buf = []
                buf_len = 0
            out.append(word[:target])
            word = word[target:]
        sep = 1 if buf else 0
        if buf_len + sep + len(word) > target and buf:
            out.append(" ".join(buf))
            buf = [word]
            buf_len = len(word)
        else:
            buf.append(word)
            buf_len += sep + len(word)
    if buf:
        out.append(" ".join(buf))
    return out


def _chunk_text_for_tts(text: str, target: int = _TTS_CHUNK_TARGET) -> list[str]:
    """Split text into Kokoro-safe chunks: newline → sentence → clause → word.

    Newline is the primary boundary — paragraph breaks are the natural
    split for chat output. Lines that exceed ``target`` cascade through
    sentence → clause → word-pack so no chunk ever blows the 510-phoneme
    cap. Short atoms coalesce up to ``target`` chars so we don't fan out
    a 50-call worker storm on a bullet list.
    """
    text = text.strip()
    if not text:
        return []

    atoms: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= target:
            atoms.append(line)
            continue
        for sent in _SENTENCE_SPLIT_RE.split(line):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= target:
                atoms.append(sent)
                continue
            for clause in _CLAUSE_SPLIT_RE.split(sent):
                clause = clause.strip()
                if not clause:
                    continue
                if len(clause) <= target:
                    atoms.append(clause)
                else:
                    atoms.extend(_word_pack(clause, target))

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for atom in atoms:
        sep = 1 if buf else 0
        if buf_len + sep + len(atom) > target and buf:
            chunks.append(" ".join(buf))
            buf = [atom]
            buf_len = len(atom)
        else:
            buf.append(atom)
            buf_len += sep + len(atom)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


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

    Long text is pre-chunked on sentence/clause boundaries so each
    Kokoro call stays under the 510-phoneme budget; the resulting audio
    arrays are concatenated server-side and shipped as one WAV. This
    preserves the no-gap single-blob contract — the previous
    sentence-chunked *streaming* design caused mid-message gaps when
    synthesis lagged playback, but server-side concat-then-blob has no
    such race.
    """
    if not _VOICE_AVAILABLE:
        return jsonify({"error": "Voice dependencies not installed"}), 503

    if not _ensure_models():
        return jsonify({"error": "Models still loading"}), 503

    data = request.get_json(silent=True) or {}
    text = _clean_for_tts((data.get("text") or "").strip())

    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunks = _chunk_text_for_tts(text)
    if not chunks:
        return jsonify({"error": "Text is required"}), 400

    logger.info(
        "[Voice] synthesize: len=%d chunks=%d head=%r",
        len(text),
        len(chunks),
        text[:80],
    )

    @stream_with_context
    def stream():
        import base64
        import json
        import time
        import numpy as np

        try:
            t0 = time.time()
            # Cap concurrency at the pool size — submitting more would just
            # have workers block on _tts_pool.get(). Equal sizes mean every
            # worker has an instance the moment its turn comes up.
            with ThreadPoolExecutor(max_workers=TTS_POOL_SIZE) as pool:
                # ``map`` preserves submission order — critical for concat.
                # ``_synthesise_chunk_safe`` retries on error/empty per chunk
                # and degrades to a short pad of silence on persistent failure
                # so a single bad chunk no longer kills the whole synthesis.
                pieces = list(pool.map(
                    _synthesise_chunk_safe,
                    list(enumerate(chunks)),
                ))
            full_audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
            logger.info(
                "[Voice] synthesized %d chunks in %.2fs (%d samples total)",
                len(chunks), time.time() - t0, len(full_audio),
            )
        except Exception as e:
            logger.error("[Voice] TTS synthesis failed: %s", e)
            yield json.dumps({"done": True, "total": 0, "error": str(e)}) + "\n"
            return

        wav_bytes = _audio_to_wav_bytes(full_audio)
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
