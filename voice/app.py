"""
Chalie Voice Service — local STT (faster-whisper) + TTS (KittenTTS).

Single FastAPI container, zero configuration.  Models are pre-cached at
Docker build time so the first real request is fast.
"""

import asyncio
import io
import logging
import os
import struct
import tempfile

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

# ── Configuration (all optional, sensible defaults) ──────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
KITTEN_VOICE = os.getenv("KITTEN_VOICE", "Jasper")
KITTEN_MODEL = os.getenv("KITTEN_MODEL", "KittenML/kitten-tts-mini-0.8")
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "60"))
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "5000"))
STT_CONCURRENCY = int(os.getenv("STT_CONCURRENCY", "1"))
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "2"))

logger = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Globals (loaded at startup) ──────────────────────────────────────────────

stt_model = None
tts_model = None
stt_sem = None
tts_sem = None

app = FastAPI(title="Chalie Voice", docs_url=None, redoc_url=None)


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global stt_model, tts_model, stt_sem, tts_sem

    stt_sem = asyncio.Semaphore(STT_CONCURRENCY)
    tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)

    logger.info("Loading STT model: %s", WHISPER_MODEL)
    from faster_whisper import WhisperModel
    stt_model = WhisperModel(WHISPER_MODEL, compute_type="int8")

    logger.info("Loading TTS model: %s (voice=%s)", KITTEN_MODEL, KITTEN_VOICE)
    from kittentts import KittenTTS
    tts_model = KittenTTS(KITTEN_MODEL)

    # Warm both models so the first real request is fast.
    logger.info("Warming STT model...")
    _warmup_stt()
    logger.info("Warming TTS model...")
    _warmup_tts()
    logger.info("Voice service ready.")


def _warmup_stt():
    """Run a tiny silent WAV through Whisper to warm CTranslate2 layers."""
    sample_rate = 16000
    silence = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, silence, sample_rate)
        segments, _ = stt_model.transcribe(tmp.name)
        list(segments)  # consume the generator


def _warmup_tts():
    """Run a short phrase through KittenTTS to warm the ONNX runtime."""
    tts_model.generate("hello", voice=KITTEN_VOICE)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _wav_duration_seconds(data: bytes) -> float:
    """Parse WAV header to get duration without decoding the full file.

    Reads the sample rate and data chunk size from the RIFF header to compute
    duration.  Returns 0.0 if the header cannot be parsed (the caller should
    still attempt transcription — Whisper is resilient).
    """
    try:
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return 0.0
        channels = struct.unpack_from("<H", data, 22)[0]
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        bits_per_sample = struct.unpack_from("<H", data, 34)[0]
        if sample_rate == 0 or channels == 0 or bits_per_sample == 0:
            return 0.0
        # Find the data chunk (it's usually at offset 36, but not always)
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


def _audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode a numpy audio array as PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


class SynthesizeRequest(BaseModel):
    text: str


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Generate speech from text.  Returns binary audio/wav."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    if len(req.text) > MAX_TTS_CHARS:
        raise HTTPException(status_code=400, detail=f"Text exceeds {MAX_TTS_CHARS} character limit")

    try:
        await asyncio.wait_for(tts_sem.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="TTS busy — try again shortly")

    try:
        audio = await asyncio.get_event_loop().run_in_executor(
            None, lambda: tts_model.generate(req.text.strip(), voice=KITTEN_VOICE)
        )
    except Exception as e:
        logger.error("TTS error: %s", e)
        raise HTTPException(status_code=500, detail="TTS generation failed")
    finally:
        tts_sem.release()

    wav_bytes = _audio_to_wav_bytes(audio)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio (WAV).  Returns {"text": "..."}."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Server-side duration check to prevent long-audio abuse.
    duration = _wav_duration_seconds(data)
    if duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds {MAX_AUDIO_SECONDS}s limit ({duration:.1f}s)",
        )

    try:
        await asyncio.wait_for(stt_sem.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="STT busy — try again shortly")

    try:
        text = await asyncio.get_event_loop().run_in_executor(None, _transcribe_sync, data)
    except Exception as e:
        logger.error("STT error: %s", e)
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        stt_sem.release()

    return {"text": text}


def _transcribe_sync(data: bytes) -> str:
    """Run faster-whisper transcription on raw WAV bytes (blocking)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        segments, _ = stt_model.transcribe(tmp.name)
        return " ".join(seg.text.strip() for seg in segments).strip()
