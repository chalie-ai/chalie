"""
Voice blueprint — STT + TTS via the unified ``moonshine_voice`` package.

One pip dependency owns G2P, ONNX runtime wiring, voice asset download, and
inference for both directions:

* STT  → ``moonshine_voice.Transcriber`` (Moonshine encoder/decoder)
* TTS  → ``moonshine_voice.TextToSpeech`` (Kokoro voice "kokoro_af_heart")

If the package is not installed, all routes return ``{"status":"unavailable"}``
or 503. Models are loaded lazily on first request to avoid blocking Flask
while assets download; ``run.py`` fires a daemon thread at boot to warm them
ahead of the first user message.

TTS text preprocessing pipeline:
  HTML pre-pass (if tags present) → markdown-it-py render →
  extract_plaintext (nh3 strip + entity decode) → URL strip →
  ordinal expansion → acronym expansion → whitespace collapse.

Prosody is restored via pysbd sentence segmentation: each sentence is
synthesised independently and silence pads (sized by terminal punctuation) are
concatenated between them. This is necessary because Moonshine's neural G2P
strips all punctuation before IPA reaches Kokoro, so a single synthesize()
call on multi-sentence text produces a flat, run-on read with no pauses.

The HTTP route streams one NDJSON chunk per sentence as soon as it is
synthesised, so the frontend can begin playback before the full text is
processed. VoicePlayer._consumeStream already handles multi-chunk NDJSON.
"""

import io
import logging
import re
import struct
import tempfile
import threading

from flask import Blueprint, Response, request, jsonify
from markdown_it import MarkdownIt
from num2words import num2words
import pysbd

from services.markup import extract_plaintext

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__)

# ── Constants (hardcoded — no env vars) ─────────────────────────────────────

MOONSHINE_LANG = "en"
TTS_LANG = "en-us"
TTS_VOICE = "kokoro_af_heart"
MAX_AUDIO_SECONDS = 660

# ── Dependency detection ────────────────────────────────────────────────────
#
# We track _which_ voice package is missing so the UI can surface a precise
# install hint instead of a generic "unavailable" — the fresh-install case.
# The shipped requirements-voice.txt covers all; if any are missing the user
# (or installer) skipped the voice pip step.

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

# Probe result from warmup: moonshine_voice.TextToSpeech.synthesize() returns
# a Python list of float64 values, not a numpy ndarray. We coerce to float32
# on every synthesis call so silence pads (np.zeros, dtype=float32) can be
# concatenated without a dtype mismatch. The coercion is intentional and
# load-bearing — do not remove it without re-verifying the probe at warmup.
_SYNTH_NEEDS_COERCE: bool = True  # set to False by _ensure_models if probe shows ndarray float32


def _ensure_models():
    """Load STT and TTS models on first use. Thread-safe, blocks concurrent loaders."""
    global _stt_model, _tts_model, _models_loaded, _models_loading, _SYNTH_NEEDS_COERCE

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
        import numpy as np

        logger.info("[Voice] Loading STT model (Moonshine, lang=%s)", MOONSHINE_LANG)
        model_path, model_arch = mv.get_model_for_language(MOONSHINE_LANG)
        stt = mv.Transcriber(model_path=model_path, model_arch=model_arch)

        logger.info("[Voice] Loading TTS model (lang=%s, voice=%s)", TTS_LANG, TTS_VOICE)
        tts = mv.TextToSpeech(TTS_LANG, voice=TTS_VOICE)

        # Probe: synthesize a short string and inspect the output type/dtype.
        # moonshine_voice 0.0.59 returns a Python list of float64 — coercion is
        # needed. If a future version returns an ndarray of float32, skip it.
        probe_samples, _probe_sr = tts.synthesize("Test.")
        probe_arr = np.asarray(probe_samples)
        needs_coerce = not (
            isinstance(probe_samples, np.ndarray) and probe_samples.dtype == np.float32
        )
        logger.info(
            "[Voice] synthesize probe: type=%s dtype=%s coerce=%s",
            type(probe_samples).__name__, probe_arr.dtype, needs_coerce,
        )

        with _load_lock:
            _stt_model = stt
            _tts_model = tts
            _SYNTH_NEEDS_COERCE = needs_coerce
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


# ── TTS text preprocessing ───────────────────────────────────────────────────
#
# Pipeline:
#   1. markdown-it-py renders markdown to HTML (handles headers, bold, italic,
#      fenced/inline code, links, images, lists, blockquotes)
#   2. extract_plaintext (services.markup) strips all tags via nh3 and decodes
#      HTML entities — same function used by the speak button
#   3. Bare URLs are rewritten to their spoken host form post-plaintext
#      (commonmark does not autolink, so bare https://... survives tag-
#      stripping as a text node). Protocol and path are dropped so
#      "http://google.com/123" reads as "google dot com".
#   4. Ordinal integers (1st, 21st, …) are expanded to words via num2words —
#      only ordinal-suffixed integers; bare integers are NEVER expanded
#   5. Short ALL-CAPS acronyms and digit-letter joins are letter-spaced so the
#      neural G2P pronounces them correctly
#   6. Whitespace is collapsed

_md = MarkdownIt("commonmark", {"breaks": True, "html": False})
_segmenter = pysbd.Segmenter(language="en", clean=False)

_URL_RE = re.compile(r"https?://([^\s/?#]+)\S*")
_WS_RE = re.compile(r"\s+")


def _spoken_url(match: "re.Match[str]") -> str:
    """Render a URL as its host read aloud, dropping protocol and path.

    ``http://google.com/123`` → ``google dot com``. Leading ``www.`` is
    dropped so ``www.example.com`` reads as ``example dot com``.
    """
    host = match.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    return host.replace(".", " dot ")

# Ordinal suffix pattern — matches ONLY integers with an ordinal suffix.
# Bare integers (phone numbers, years, version numbers) are not touched.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")

# Acronym helpers (unchanged from prior version — neural G2P treats 2-3 letter
# ALL-CAPS tokens as fake one-syllable words; letter-spacing forces per-letter
# pronunciation; 4+ letter tokens pronounce correctly from the dictionary).
_DIGIT_LETTER_RE = re.compile(r"(\d)([A-Za-z])")
_LETTER_DIGIT_RE = re.compile(r"([A-Za-z])(\d)")
_SHORT_ACRONYM_RE = re.compile(r"\b[A-Z]{2,3}\b")


def _expand_ordinals_for_tts(text: str) -> str:
    """Replace ordinal-suffixed integers with their spelled-out form.

    ``1st`` → ``first``, ``21st`` → ``twenty-first``. Only integers followed
    by an ordinal suffix (st/nd/rd/th) are touched. Bare integers, phone
    numbers, version strings, and years are left unchanged.
    """
    def _replace(m: re.Match) -> str:
        return num2words(int(m.group(1)), to="ordinal")

    return _ORDINAL_RE.sub(_replace, text)


def _expand_acronyms_for_tts(text: str) -> str:
    """Letter-space short ALL-CAPS tokens and split digit-letter joins.

    Moonshine's neural G2P treats 2- and 3-letter ALL-CAPS tokens as fake
    one-syllable words (``MB``→"eb", ``VM``→"em"). Splitting the letters with
    whitespace forces per-letter pronunciation (``M B``→"em bee"). Number-
    letter joins like ``228MB`` confuse it further; a space restores both.

    Acronyms of length 4+ test correctly against moonshine's dictionary
    (``NASA``, ``HTTP``, ``HTML``) so they are left alone.
    """
    text = _DIGIT_LETTER_RE.sub(r"\1 \2", text)
    text = _LETTER_DIGIT_RE.sub(r"\1 \2", text)
    text = _SHORT_ACRONYM_RE.sub(lambda m: " ".join(m.group(0)), text)
    return text


def _clean_for_tts(text: str) -> str:
    """Convert response content to TTS-safe plaintext.

    Pipeline:
      1. If the text contains HTML tags (LLM output path), call
         extract_plaintext first to strip them and decode entities — prevents
         literal ``<p>`` / ``<b>`` tag text from surviving into the output.
      2. Run through markdown-it-py so any markdown markers (headers, bold,
         italic, fenced code, links, images, lists) become clean HTML.
      3. extract_plaintext again to strip those tags + decode any entities
         introduced by markdown-it-py (e.g. ``&quot;`` inside code spans).
      4. Rewrite bare URLs to their spoken host form (commonmark does not
         autolink, so https://... survives as a text node through tag
         stripping). ``http://google.com/123`` → ``google dot com``.
      5. Expand ordinal integers (``1st`` → ``first``).
      6. Letter-space short ALL-CAPS acronyms and split digit-letter joins.
      7. Collapse whitespace.

    Plain text (no markdown markers, no HTML) round-trips cleanly through
    a paragraph wrap — markdown-it-py is a no-op on prose without markers.
    """
    if not text:
        return ""
    # Step 1: HTML pre-pass — strip LLM-emitted tags and decode entities
    # before markdown-it sees the text. Without this, '<p>foo</p>' becomes
    # '&lt;p&gt;foo&lt;/p&gt;' and the literal tag survives as output text.
    if "<" in text and ">" in text:
        text = extract_plaintext(text)
    # Step 2-3: markdown render → tag strip + entity decode
    html = _md.render(text)
    plain = extract_plaintext(html)
    # Step 4: URL → spoken host (e.g. "google dot com")
    plain = _URL_RE.sub(_spoken_url, plain)
    # Steps 5-6: text expansions
    plain = _expand_ordinals_for_tts(plain)
    plain = _expand_acronyms_for_tts(plain)
    # Step 7: whitespace collapse
    return _WS_RE.sub(" ", plain).strip()


# ── Prosody-aware synthesis ─────────────────────────────────────────────────
#
# Moonshine's neural G2P strips every punctuation mark before IPA reaches
# Kokoro, so a single synthesize() call on a multi-sentence string yields a
# flat, run-on read. The previous espeak-ng path preserved punctuation as IPA
# pause tokens. We restore prosody by:
#   1. Segmenting with pysbd (handles "Dr. Smith", U.S.A., etc. correctly)
#   2. Synthesising each sentence independently under _tts_lock
#   3. Yielding each sentence's WAV (plus a trailing silence pad) as a
#      separate NDJSON chunk — the frontend begins playback on the first chunk
#
# Pause values are keyed off terminal punctuation and sized for natural speech.

_PAUSE_BY_TERMINAL = {
    ".": 0.28, "!": 0.30, "?": 0.30,
    ":": 0.20, ";": 0.20,
    ",": 0.14,
}


def _segment_for_prosody(text: str) -> list[tuple[str, float]]:
    """Split ``text`` into ``(sentence, trailing_pause_seconds)`` pairs.

    Uses pysbd for sentence boundary detection — correctly handles
    abbreviations (``Dr. Smith`` → one sentence), decimal numbers, U.S.A.,
    and other patterns that fool a char-scanner approach.

    The last sentence always gets 0.0 pause (no trailing silence after the
    final word). Pause length is keyed off the sentence's terminal punctuation.
    """
    if not text:
        return []
    sentences = _segmenter.segment(text)
    out: list[tuple[str, float]] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        last_char = s[-1]
        pause = _PAUSE_BY_TERMINAL.get(last_char, 0.10)
        out.append((s, pause))
    if out:
        # No trailing silence after the final sentence
        out[-1] = (out[-1][0], 0.0)
    return out


def _synth_to_float32(text: str):
    """Call _tts_model.synthesize and coerce output to (np.ndarray[float32], sr).

    moonshine_voice.TextToSpeech.synthesize() returns a Python list of float64
    values (observed at warmup probe). We coerce to float32 so silence pads
    (np.zeros, dtype=float32) can be concatenated without a dtype mismatch.
    The coercion is skipped if the probe showed the model already returns a
    float32 ndarray (_SYNTH_NEEDS_COERCE=False).
    """
    import numpy as np
    samples, sr = _tts_model.synthesize(text)
    if _SYNTH_NEEDS_COERCE:
        samples = np.asarray(samples, dtype=np.float32)
    return samples, sr


def _synthesize_sentence_wav(sentence: str, pause: float) -> bytes:
    """Synthesise one sentence and return WAV bytes with silence pad baked in.

    Sample rate is taken from the model's actual output (``_synth_to_float32``);
    callers do not pass one because the silence pad and WAV encoding must use
    the same rate the model produced — passing a different value would create
    a pad/audio mismatch.
    """
    import numpy as np
    samples, sr = _synth_to_float32(sentence)
    if pause > 0:
        silence = np.zeros(int(pause * sr), dtype=np.float32)
        samples = np.concatenate([samples, silence])
    return _audio_to_wav_bytes(samples, sr)


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
    """Synthesise speech and stream one NDJSON chunk per sentence.

    Each sentence is synthesised (under _tts_lock) and yielded as a
    ``{"index": i, "total": N, "audio": "<base64-WAV>"}`` line as soon as it
    is ready. A final ``{"done": true, "total": N}`` sentinel closes the
    stream. The frontend's VoicePlayer._consumeStream handles multi-chunk
    NDJSON and begins playback on the first chunk received.

    Silence pads (sized by terminal punctuation) are baked into each chunk's
    WAV so there is no gap in playback between sentences.
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

    segments = _segment_for_prosody(text)
    total = len(segments)

    def _generate():
        t0 = time.time()
        for i, (sentence, pause) in enumerate(segments):
            try:
                with _tts_lock:
                    wav_bytes = _synthesize_sentence_wav(sentence, pause)
            except Exception as e:
                logger.error("[Voice] TTS synthesis failed on segment %d: %s", i, e)
                # NB: error sentinel intentionally does NOT carry ``done: true``.
                # The frontend treats ``done`` as the success terminator; if we
                # set it on errors the spinner stays up and no error is shown.
                # See voice_player.js _enqueueLine — it routes ``error`` lines
                # through the fetch's abort/catch path instead.
                error_line = json.dumps({
                    "error": "TTS synthesis failed", "total": total, "index": i,
                }) + "\n"
                yield error_line
                return

            chunk = json.dumps({
                "index": i,
                "total": total,
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
            }) + "\n"
            yield chunk
            logger.debug("[Voice] yielded segment %d/%d", i + 1, total)

        logger.info("[Voice] synthesized %d segments in %.2fs", total, time.time() - t0)
        yield json.dumps({"done": True, "total": total}) + "\n"

    return Response(
        _generate(),
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
