FROM python:3.12-slim

# System deps:
#   build-essential + libffi-dev  → native extensions (cryptography/pywebpush)
#   libsndfile1                   → soundfile (voice STT/TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching — only re-runs when requirements change)
COPY backend/requirements.txt backend/requirements-voice.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt \
 && pip install --no-cache-dir -r backend/requirements-voice.txt

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Data directory for SQLite DB and runtime files
RUN mkdir -p /data
VOLUME ["/data"]

ENV CHALIE_DB_PATH=/data/chalie.db

EXPOSE 8081

CMD ["python", "backend/run.py", "--port=8081", "--host=0.0.0.0"]
