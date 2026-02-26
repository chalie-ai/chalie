FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time so the container starts fast.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
RUN python -c "from kittentts import KittenTTS; KittenTTS('KittenML/kitten-tts-mini-0.8')"

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
