"""TTS synthesis service — Kokoro-based text-to-speech.

Extracted from ``api.voice`` so non-HTTP callers (the MP spine, background
tasks, unit tests) can synthesize speech without spinning up a Flask request.

:class:`VoiceTranscriptService` owns everything in one place:
* Kokoro model creation and lazy loading (with its own lock and loading-state
  flags that back the 503 contract)
* TTS text preprocessing (markdown/HTML stripping, URL spoken-form rewrite,
  whitespace collapse)
* Sentence-level chunking to stay under Kokoro's 510-phoneme / ~320-char limit
* Per-chunk synthesis with silence-pad stitching
* WAV encoding of the final waveform

Public API: :meth:`instance` returns the process-wide handle;
:meth:`synthesize` turns text into a raw sample array, sample rate and WAV
bytes; :meth:`synthesize_settled` drives the whole pipeline for one settled
transcript row (attempts, silence check, storage, websocket signal);
:meth:`delete_for_transcripts` is the single owner of voice cleanup.
"""

from __future__ import annotations

import io
import logging
import re
import threading
from typing import TYPE_CHECKING, cast

from models.transcript import Transcript
from models.voice_signal import VoiceSignal
from models.voice_transcript import VoiceTranscript
from services.file_mapper_service import FileMapperService
from services.markup import extract_plaintext, markdown_to_html
from services.websocket import Websocket

if TYPE_CHECKING:
    from typing import Protocol

    class _KokoroModel(Protocol):
        def create(self, text: str, voice: str, speed: float, lang: str) -> "tuple[object, int]": ...

logger = logging.getLogger(__name__)


class VoiceTranscriptService:
    """TTS synthesis service — the single source of truth for Kokoro-based
    speech synthesis.

    Synthesis is never on a request thread: :meth:`synthesize_settled` runs it
    in the background when a turn's final reply settles, and the playback route
    in ``api.voice`` only serves the outcome this service already stored.
    """

    # ── Model paths & voice config ───────────────────────────────────────────

    KOKORO_MODEL = FileMapperService.get_voice_models_path("kokoro", "kokoro-v1.0.onnx")
    KOKORO_VOICES = FileMapperService.get_voice_models_path("kokoro", "voices-v1.0.bin")

    TTS_VOICE = "af_heart"
    TTS_LANG = "en-us"

    MAX_SYNTHESIS_ATTEMPTS = 3
    SILENCE_PEAK_THRESHOLD = 0.001

    # Kokoro ONNX hard limit is 510 phonemes (~1.5 chars/phoneme → 340 chars).
    # 320 gives headroom for phoneme-heavy words.
    MAX_TTS_CHUNK_CHARS = 320
    TTS_SILENCE_PAD_SECONDS = 0.15

    # ── TTS text preprocessing patterns ──────────────────────────────────────
    #
    # Kokoro phonemises via phonemizer-fork (espeak-ng), which:
    #   * preserves punctuation as IPA pause tokens (commas, periods, question
    #     marks all produce natural prosody breaks)
    #   * normalises numbers ("v0.8" reads as "vee zero point eight")
    #   * pronounces 2-3 letter acronyms correctly
    #   * handles abbreviations and ordinals natively
    #
    # So we only need to (a) strip HTML/markdown markers so the LLM's raw output
    # doesn't read tags or symbols aloud, (b) rewrite bare URLs to a
    # human-readable host form (otherwise espeak spells out every slash and
    # digit), (c) collapse whitespace.
    #
    # Block-level markdown the LLM occasionally leaks into responses. We do NOT
    # run a full markdown parser for TTS — we strip the block markers and route
    # list items through ``<li>`` so they share the HTML list-pause logic below.
    # Images drop entirely (alt text isn't spoken); links keep their text, not
    # the URL.
    _FENCE_RE = re.compile(r"^[ \t]*```[^\n]*$", re.MULTILINE)
    _MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
    _MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
    _HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
    _BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
    _LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+(.*)$", re.MULTILINE)
    _URL_RE = re.compile(r"https?://([^\s/?#]+)\S*")
    _WS_RE = re.compile(r"\s+")
    # Append a period before ``</li>`` when the item doesn't already end in
    # sentence-terminating punctuation. ``extract_plaintext`` strips ``<li>``
    # tags down to a single space, which espeak runs together without a beat —
    # items need real punctuation to produce a natural between-item pause.
    _LI_NEEDS_TERMINATOR_RE = re.compile(
        r"([^\s.!?,;:])\s*</li\s*>", re.IGNORECASE,
    )
    _TTS_SPLIT_RE = re.compile(r"(?<=[.!?,;:—])\s+")

    # ── Process-wide model state ─────────────────────────────────────────────
    #
    # Class-level, not per-instance: exactly one Kokoro model per process no
    # matter how many handles exist. A second instance loading its own copy
    # would silently double the model's resident footprint.
    _kokoro: object = None
    _tts_lock = threading.Lock()
    _load_lock = threading.Lock()
    _models_loaded = False
    _models_loading = False

    _synthesize_in_flight: set[int] = set()
    _synthesize_in_flight_lock = threading.Lock()

    _instance: "VoiceTranscriptService | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "VoiceTranscriptService":
        """Return the process-wide singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Model lifecycle ──────────────────────────────────────────────────────

    @classmethod
    def missing_model_files(cls) -> list[str]:
        """Return the names of Kokoro model files that are missing on disk."""
        expected = [cls.KOKORO_MODEL, cls.KOKORO_VOICES]
        return [p.name for p in expected if not p.is_file()]

    @classmethod
    def _build_session(cls) -> object:
        """Build Kokoro's ONNX session ourselves so we control the memory arena.

        ``kokoro_onnx.Kokoro`` constructs its session with no ``SessionOptions``
        at all, which silently accepts ORT's default
        ``enable_cpu_mem_arena=True``. That arena never returns scratch to the
        OS: it grows to the largest activation ever allocated — a function of
        the longest reply ever spoken — and holds it for the process lifetime.
        Measured over an alternating short/long synthesis workload: +816 MiB of
        held scratch with the arena on, and none with it off.
        """
        import onnxruntime as ort

        from services.onnx_session import build_session

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        return build_session(
            cls.KOKORO_MODEL,
            sess_options=opts,
            log_prefix="[VoiceTranscript]",
        )

    @classmethod
    def ensure_loaded(cls) -> bool:
        """Lazily load the Kokoro model. Returns ``False`` if files are missing
        or loading is in progress (caller should 503). Sets the shared
        loading-state flags so the HTTP layer can distinguish "still loading"
        from "permanently unavailable"."""
        if cls._models_loaded:
            return True

        with cls._load_lock:
            if cls._models_loaded:
                return True
            if cls._models_loading:
                return False
            cls._models_loading = True

        try:
            missing = cls.missing_model_files()
            if missing:
                logger.error("[VoiceTranscript] Kokoro model files missing: %s", missing)
                with cls._load_lock:
                    cls._models_loading = False
                return False

            from kokoro_onnx import Kokoro

            logger.info("[VoiceTranscript] Loading Kokoro TTS from %s", cls.KOKORO_MODEL)
            kokoro = Kokoro.from_session(cls._build_session(), str(cls.KOKORO_VOICES))

            with cls._load_lock:
                cls._kokoro = kokoro
                cls._models_loaded = True
                cls._models_loading = False

            logger.info("[VoiceTranscript] Kokoro loaded — accepting synthesis requests")
            return True

        except Exception as e:
            logger.error("[VoiceTranscript] Kokoro loading failed: %s", e)
            with cls._load_lock:
                cls._models_loading = False
            return False

    @property
    def is_loaded(self) -> bool:
        """Whether the Kokoro model is ready for synthesis."""
        return type(self)._models_loaded

    @property
    def is_loading(self) -> bool:
        """Whether the Kokoro model is currently being loaded (transient)."""
        return type(self)._models_loading

    # ── Text preprocessing ───────────────────────────────────────────────────

    @staticmethod
    def _spoken_url(match: "re.Match[str]") -> str:
        """Rewrite ``http://google.com/123`` → ``google dot com``.

        Without this, espeak reads URLs character-by-character — fast but
        unpleasant.
        """
        host = match.group(1).lower()
        if host.startswith("www."):
            host = host[4:]
        return host.replace(".", " dot ")

    @classmethod
    def clean_for_tts(cls, text: str) -> str:
        """Strip markup from *text* and rewrite it into a spoken-friendly form."""
        if not text:
            return ""
        # ONE common path: strip block markdown the LLM leaks (lists become <li>
        # so they join the HTML list-pause logic), rewrite leaked inline emphasis
        # via the shared markup helper, then flatten every tag through
        # extract_plaintext.
        text = cls._FENCE_RE.sub("", text)
        text = cls._MD_IMAGE_RE.sub("", text)
        text = cls._MD_LINK_RE.sub(r"\1", text)
        text = cls._HEADING_RE.sub("", text)
        text = cls._BLOCKQUOTE_RE.sub("", text)
        text = cls._LIST_ITEM_RE.sub(r"<li>\1</li>", text)
        text = markdown_to_html(text)
        text = cls._LI_NEEDS_TERMINATOR_RE.sub(r"\1.</li>", text)
        plain = extract_plaintext(text)
        plain = cls._URL_RE.sub(cls._spoken_url, plain)
        return cls._WS_RE.sub(" ", plain).strip()

    @classmethod
    def segment_for_tts(cls, text: str) -> list[str]:
        """Split *text* into chunks that fit Kokoro's per-call length limit.

        Greedy sentence-level accumulation: walk the text on sentence terminators
        (``.!?,;:—``) and keep adding the next sentence to the current chunk as
        long as it fits. When a sentence would overflow, flush the current chunk
        and start a new one. If a single sentence itself exceeds the limit,
        hard-split it on whitespace.
        """
        if not text:
            return []
        limit = cls.MAX_TTS_CHUNK_CHARS
        chunks: list[str] = []
        current = ""
        for fragment in cls._TTS_SPLIT_RE.split(text):
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

    # ── Synthesis ────────────────────────────────────────────────────────────

    @staticmethod
    def _audio_to_wav_bytes(samples: object, sample_rate: int) -> bytes:
        """Encode a float32 sample array (or list thereof) to a WAV blob."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def synthesize(self, text: str) -> tuple[object, int, bytes]:
        """Synthesize *text* into speech.

        Returns:
            A ``(samples, sample_rate, wav_bytes)`` tuple:
            * ``samples`` — raw float32 numpy array
            * ``sample_rate`` — Hz (typically 24000 for Kokoro)
            * ``wav_bytes`` — PCM_16 WAV blob ready to stream

        Raises:
            ValueError: If *text* is empty after TTS preprocessing.
            RuntimeError: If the Kokoro model is missing or failed to load.
            Exception: Propagated from Kokoro if synthesis fails (caller
                should map to 500).
        """
        cls = type(self)
        cleaned = cls.clean_for_tts(text.strip())
        if not cleaned:
            raise ValueError("Text is required")

        chunks = cls.segment_for_tts(cleaned)
        logger.info(
            "[VoiceTranscript] synthesize: len=%d chunks=%d head=%r",
            len(cleaned), len(chunks), cleaned[:80],
        )

        if not cls.ensure_loaded():
            raise RuntimeError("Kokoro model not available")

        if len(chunks) == 1:
            with cls._tts_lock:
                samples, sr = cast("_KokoroModel", cls._kokoro).create(
                    chunks[0], voice=cls.TTS_VOICE, speed=1.0, lang=cls.TTS_LANG,
                )
        else:
            import numpy as np

            all_samples: list[object] = []
            rate: int | None = None
            with cls._tts_lock:
                for i, chunk in enumerate(chunks):
                    chunk_samples, chunk_sr = cast(
                        "_KokoroModel", cls._kokoro,
                    ).create(chunk, voice=cls.TTS_VOICE, speed=1.0, lang=cls.TTS_LANG)
                    if rate is None:
                        rate = chunk_sr
                    all_samples.append(chunk_samples)
                    if i < len(chunks) - 1:
                        pad_len = int(cls.TTS_SILENCE_PAD_SECONDS * rate)
                        all_samples.append(
                            np.zeros(
                                pad_len,
                                dtype=cast(
                                    "np.ndarray[tuple[int], np.dtype[np.float32]]",
                                    chunk_samples,
                                ).dtype,
                            )
                        )
            samples = np.concatenate(
                cast("list[np.ndarray[tuple[int], np.dtype[np.float32]]]", all_samples)
            )
            if rate is None:
                raise RuntimeError("Kokoro returned no sample rate")
            sr = rate

        wav_bytes = self._audio_to_wav_bytes(samples, sr)
        return samples, sr, wav_bytes

    def synthesize_settled(self, transcript_id: int) -> None:
        """Synthesize speech for one settled transcript row, exactly once to a
        terminal state. Retries up to 3 times on silent or failed synthesis,
        persists the WAV path on success or NULL on failure, and broadcasts
        the pipeline state transition."""
        import numpy as np

        cls = type(self)
        with cls._synthesize_in_flight_lock:
            if transcript_id in cls._synthesize_in_flight:
                return
            if VoiceTranscript.filter("transcript_id", transcript_id).exists():
                return
            cls._synthesize_in_flight.add(transcript_id)
        try:
            row = Transcript.get(transcript_id)
            if row is None:
                return
            text = cls.clean_for_tts(row.content)
            Websocket.broadcast(VoiceSignal.pending(transcript_id))
            if not text:
                VoiceTranscript(
                    transcript_id=transcript_id, file_path=None,
                ).record()
                Websocket.broadcast(VoiceSignal.failed(transcript_id))
                logger.error(
                    "[VoiceTranscript] synthesize_settled(%d): empty text — marking failed",
                    transcript_id,
                )
                return
            for attempt in range(1, cls.MAX_SYNTHESIS_ATTEMPTS + 1):
                try:
                    samples, _sr, wav_bytes = self.synthesize(text)
                    samples_arr = cast(
                        "np.ndarray[tuple[int], np.dtype[np.float32]]", samples,
                    )
                    if (
                        len(samples_arr) > 0
                        and float(np.max(np.abs(samples_arr))) > cls.SILENCE_PEAK_THRESHOLD
                    ):
                        file_name = f"{transcript_id}.wav"
                        path = FileMapperService.get_voice_path(file_name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(wav_bytes)
                        VoiceTranscript(
                            transcript_id=transcript_id, file_path=file_name,
                        ).record()
                        Websocket.broadcast(VoiceSignal.ready(transcript_id))
                        return
                    logger.error(
                        "[VoiceTranscript] synthesize_settled(%d): attempt %d/%d produced silent audio",
                        transcript_id, attempt, cls.MAX_SYNTHESIS_ATTEMPTS,
                    )
                except Exception:
                    logger.exception(
                        "[VoiceTranscript] synthesize_settled(%d): attempt %d/%d raised",
                        transcript_id, attempt, cls.MAX_SYNTHESIS_ATTEMPTS,
                    )
            VoiceTranscript(
                transcript_id=transcript_id, file_path=None,
            ).record()
            Websocket.broadcast(VoiceSignal.failed(transcript_id))
            logger.error(
                "[VoiceTranscript] synthesize_settled(%d): all attempts exhausted — marking failed",
                transcript_id,
            )
        finally:
            with cls._synthesize_in_flight_lock:
                cls._synthesize_in_flight.discard(transcript_id)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def delete_for_transcripts(self, transcript_ids: list[int]) -> None:
        """Delete the stored WAV files and ``voice_transcript`` rows for the
        given transcript ids — the single owner of voice cleanup. The rows
        alone would cascade away with their transcript, but SQLite cannot see
        the file on disk: the paths must be read and unlinked *before* the
        transcript delete removes the only record of where they live. A failed
        unlink is logged with its path and never blocks the delete — a stray
        WAV is recoverable, a half-deleted transcript is not."""
        for row in VoiceTranscript.for_transcripts(transcript_ids):
            if row.file_path is not None:
                path = FileMapperService.get_voice_path(row.file_path)
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.error(
                        "[VoiceTranscript] delete_for_transcripts: unlink %s failed: %s",
                        path, exc,
                    )
        VoiceTranscript.delete_by_transcripts(transcript_ids)
