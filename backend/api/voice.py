"""
Voice blueprint — STT + TTS via two single-purpose ONNX libraries.

* TTS → ``kokoro_onnx.Kokoro`` (Kokoro v1.0 + espeak-ng phonemizer)
* STT → ``moonshine_onnx.MoonshineOnnxModel`` (Moonshine base, ONNX)

The ONNX model files are downloaded on demand, not bundled with the install.
Turning voice on in Settings (``PUT /api/voice-settings``) calls
``RuntimeDepsService.enable_voice()``, which background-installs the voice deps
and downloads the models into ``resources/voice-models/``. Until the deps or
model files are present every route returns ``{"status":"unavailable"}`` or 503
with a precise hint pointing the user back to that setting.

Kokoro uses phonemizer-fork + espeakng-loader under the hood, which preserves
punctuation as IPA pause tokens and handles numbers, abbreviations, and 2-3
letter acronyms natively. The TTS-side text pipeline collapses to:
    markdown/HTML → plaintext → URL → spoken-host → whitespace collapse.

The synthesize route returns a single ``audio/wav`` blob — no streaming, no
NDJSON. Long text is split into sentence-level chunks (≤320 chars each) before
being passed to Kokoro so the 510-phoneme hard limit is never hit. Chunks are
concatenated with short silence pads and encoded as a single waveform.
"""

import io
import logging
import re
import struct
import tempfile
import threading
from typing import TYPE_CHECKING, cast

from flask import Blueprint, Response, request, jsonify

if TYPE_CHECKING:
    from typing import Protocol
    from flask.typing import ResponseReturnValue

    class _KokoroModel(Protocol):
        def create(self, text: str, voice: str, speed: float, lang: str) -> "tuple[object, int]": ...

from services.file_mapper_service import FileMapperService
from services.markup import extract_plaintext, markdown_to_html

from .auth import require_auth

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

# Models live in resources/voice-models/ — a directory sibling to backend/,
# intentionally OUTSIDE the data/ volume so the files survive `chalie update` on
# native installs. They are downloaded at runtime by
# RuntimeDepsService.enable_voice() when the user turns voice on in Settings,
# not baked in by the installer.
_KOKORO_MODEL = FileMapperService.get_voice_models_path("kokoro", "kokoro-v1.0.onnx")
_KOKORO_VOICES = FileMapperService.get_voice_models_path("kokoro", "voices-v1.0.bin")
_MOONSHINE_DIR = FileMapperService.get_voice_models_path("moonshine", "base")

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
    "kokoro_onnx", "moonshine_onnx", "soundfile", "numpy",
    "silero_vad_lite", "noisereduce",
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
    "Voice dependencies are not installed. Run "
    "`uv pip install --system -e backend[voice]` (or relaunch with "
    "`./run.sh` — failed installs auto-retry on next launch)."
)


def _voice_unavailable_payload() -> "dict[str, object]":
    return {
        "error": "Voice dependencies not installed",
        "reason": "deps_missing",
        "missing": list(_VOICE_MISSING_MODULES),
        "hint": _VOICE_INSTALL_HINT,
    }


def _loading_or_missing_response() -> "ResponseReturnValue":
    """503 with a precise ``reason`` and ``Retry-After`` so the client can decide
    whether to auto-retry (transient cold-start) or give up (missing files)."""
    missing = _missing_model_files()
    if missing:
        return jsonify({
            "error": "Voice models not installed",
            "reason": "models_missing",
            "missing": missing,
            "hint": "Enable voice in Settings to download voice models.",
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

_kokoro: object = None
_moonshine: object = None
_load_lock = threading.Lock()
_models_loaded = False
_models_loading = False

# phonemizer-fork (espeak-ng under the hood) is not thread-safe; kokoro.create()
# calls it on every invocation. Serialise the full TTS path. STT shares its own
# lock so a long synthesis does not block a mic recording from getting
# transcribed.
_tts_lock = threading.Lock()
_stt_lock = threading.Lock()


def _ensure_models() -> bool:
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

# Block-level markdown the LLM occasionally leaks into responses. We do NOT run a
# full markdown parser for TTS — we strip the block markers and route list items
# through ``<li>`` so they share the HTML list-pause logic below. Images drop
# entirely (alt text isn't spoken); links keep their text, not the URL.
_FENCE_RE = re.compile(r"^[ \t]*```[^\n]*$", re.MULTILINE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+(.*)$", re.MULTILINE)
_URL_RE = re.compile(r"https?://([^\s/?#]+)\S*")
_WS_RE = re.compile(r"\s+")
# Append a period before ``</li>`` when the item doesn't already end in
# sentence-terminating punctuation. ``extract_plaintext`` strips ``<li>`` tags
# down to a single space, which espeak runs together without a beat — items
# need real punctuation to produce a natural between-item pause.
_LI_NEEDS_TERMINATOR_RE = re.compile(
    r"([^\s.!?,;:])\s*</li\s*>", re.IGNORECASE,
)


def _spoken_url(match: "re.Match[str]") -> str:
    """Rewrite ``http://google.com/123`` → ``google dot com``.

    Without this, espeak reads URLs character-by-character — fast but unpleasant.
    """
    host = match.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host.replace(".", " dot ")


def _clean_for_tts(text: str) -> str:
    if not text:
        return ""
    # ONE common path: strip block markdown the LLM leaks (lists become <li> so
    # they join the HTML list-pause logic), rewrite leaked inline emphasis via the
    # shared markup helper, then flatten every tag through extract_plaintext.
    text = _FENCE_RE.sub("", text)
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_ITEM_RE.sub(r"<li>\1</li>", text)
    text = markdown_to_html(text)
    text = _LI_NEEDS_TERMINATOR_RE.sub(r"\1.</li>", text)
    plain = extract_plaintext(text)
    plain = _URL_RE.sub(_spoken_url, plain)
    return _WS_RE.sub(" ", plain).strip()


# Kokoro ONNX hard limit is 510 phonemes (~1.5 chars/phoneme → 340 chars).
# 320 gives headroom for phoneme-heavy words.
_MAX_TTS_CHUNK_CHARS = 320
_TTS_SILENCE_PAD_SECONDS = 0.15
_TTS_SPLIT_RE = re.compile(r"(?<=[.!?,;:—])\s+")


def _segment_for_tts(text: str) -> list[str]:
    if not text:
        return []
    limit = _MAX_TTS_CHUNK_CHARS
    chunks: list[str] = []
    current = ""
    for fragment in _TTS_SPLIT_RE.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue
        candidate = f"{current} {fragment}".strip() if current else fragment
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Fragment itself over limit — hard-split on whitespace.
            if len(fragment) > limit:
                for word in fragment.split():
                    word = word[:limit]
                    wc = f"{current} {word}".strip() if current else word
                    if len(wc) <= limit:
                        current = wc
                    else:
                        if current:
                            chunks.append(current)
                        current = word
            else:
                current = fragment
    if current:
        chunks.append(current)
    return chunks or [text]


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


# ── STT audio preprocessing ─────────────────────────────────────────────────

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
        from silero_vad_lite import SileroVAD

        vad = SileroVAD(sr)
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


def _audio_to_wav_bytes(samples: object, sample_rate: int) -> bytes:
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

@voice_bp.route("/voice/health", methods=["GET"])
@require_auth
def voice_health() -> "ResponseReturnValue":
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
            "hint": "Enable voice in Settings to download voice models.",
        }), 200

    threading.Thread(target=_ensure_models, daemon=True).start()
    return jsonify({"status": "loading"}), 200


@voice_bp.route("/voice/synthesize", methods=["POST"])
@require_auth
def voice_synthesize() -> "ResponseReturnValue":
    if not _VOICE_AVAILABLE:
        return jsonify(_voice_unavailable_payload()), 503

    if not _ensure_models():
        return _loading_or_missing_response()

    data = request.get_json(silent=True) or {}
    text = _clean_for_tts((data.get("text") or "").strip())

    if not text:
        return jsonify({"error": "Text is required"}), 400

    chunks = _segment_for_tts(text)
    logger.info(
        "[Voice] synthesize: len=%d chunks=%d head=%r",
        len(text), len(chunks), text[:80],
    )

    try:
        if len(chunks) == 1:
            with _tts_lock:
                samples, sr = cast("_KokoroModel", _kokoro).create(
                    chunks[0], voice=TTS_VOICE, speed=1.0, lang=TTS_LANG,
                )
        else:
            import numpy as np
            all_samples: list[object] = []
            sr = None
            with _tts_lock:
                for i, chunk in enumerate(chunks):
                    chunk_samples, chunk_sr = cast("_KokoroModel", _kokoro).create(
                        chunk, voice=TTS_VOICE, speed=1.0, lang=TTS_LANG,
                    )
                    if sr is None:
                        sr = chunk_sr
                    all_samples.append(chunk_samples)
                    if i < len(chunks) - 1:
                        pad_len = int(_TTS_SILENCE_PAD_SECONDS * sr)
                        all_samples.append(np.zeros(pad_len, dtype=cast("np.ndarray[tuple[int], np.dtype[np.float32]]", chunk_samples).dtype))
            samples = np.concatenate(cast("list[np.ndarray[tuple[int], np.dtype[np.float32]]]", all_samples))
    except Exception as e:
        logger.error("[Voice] TTS synthesis failed: %s", e)
        return jsonify({"error": "TTS synthesis failed"}), 500

    wav_bytes = _audio_to_wav_bytes(samples, cast(int, sr))
    return Response(
        wav_bytes,
        mimetype="audio/wav",
        headers={
            "Content-Length": str(len(wav_bytes)),
            "Cache-Control": "no-cache",
        },
    )


@voice_bp.route("/voice/transcribe", methods=["POST"])
@require_auth
def voice_transcribe() -> "ResponseReturnValue":
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
