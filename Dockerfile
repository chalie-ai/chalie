FROM python:3.12-slim

# System deps:
#   build-essential + libffi-dev  → native extensions (cryptography/pywebpush)
#   libsndfile1                   → soundfile (voice STT/TTS)
#   libsqlite3-dev + gettext-base + curl → build sqlite-vec from source (PyPI wheel ships broken aarch64 binary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libsndfile1 \
    libsqlite3-dev \
    gettext-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching — only re-runs when requirements change)
COPY backend/requirements.txt backend/requirements-voice.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt \
 && pip install --no-cache-dir -r backend/requirements-voice.txt

# Fix sqlite-vec: the PyPI aarch64 wheel ships a 32-bit ARM .so (upstream bug).
# Build the native extension from source and replace the broken binary.
ARG SQLITE_VEC_VERSION=0.1.6
RUN cd /tmp \
 && curl -sL "https://github.com/asg017/sqlite-vec/archive/refs/tags/v${SQLITE_VEC_VERSION}.tar.gz" | tar xz \
 && cd "sqlite-vec-${SQLITE_VEC_VERSION}" \
 && echo "${SQLITE_VEC_VERSION}" > VERSION \
 && VERSION=${SQLITE_VEC_VERSION} DATE=docker SOURCE=local \
    VERSION_MAJOR=$(echo ${SQLITE_VEC_VERSION} | cut -d. -f1) \
    VERSION_MINOR=$(echo ${SQLITE_VEC_VERSION} | cut -d. -f2) \
    VERSION_PATCH=$(echo ${SQLITE_VEC_VERSION} | cut -d. -f3) \
    envsubst < sqlite-vec.h.tmpl > sqlite-vec.h \
 && mkdir -p dist \
 && cc -fPIC -shared -O3 -lm -I/usr/include sqlite-vec.c -o dist/vec0.so \
 && SITE_PKG=$(python3 -c "import sqlite_vec, os; print(os.path.dirname(sqlite_vec.__file__))") \
 && cp dist/vec0.so "$SITE_PKG/vec0.so" \
 && python3 -c "import sqlite3, sqlite_vec; conn = sqlite3.connect(':memory:'); conn.enable_load_extension(True); sqlite_vec.load(conn); conn.execute('CREATE VIRTUAL TABLE t USING vec0(e float[4])'); print('sqlite-vec: OK')" \
 && cd / && rm -rf /tmp/sqlite-vec-*

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY run.sh ./

# Data directory for SQLite DB and runtime files
RUN mkdir -p /data
VOLUME ["/data"]

ENV CHALIE_DB_PATH=/data/chalie.db

EXPOSE 8081

CMD ["bash", "run.sh", "--port=8081", "--host=0.0.0.0"]
