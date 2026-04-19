FROM python:3.12-slim

# Copy source to the path the installer treats as canonical.
# install.sh auto-detects it via backend/requirements.txt and skips the download.
COPY . /root/.chalie/app/

# Single install path — identical to native user installs.
# Installs: system build deps, voice deps, Deno, Python venv, playwright
# browsers, and the sqlite-vec aarch64 patch (no-op on x86_64).
RUN bash /root/.chalie/app/installer/install.sh

# Deno was installed to /root/.deno by the script; expose its bin for runtime.
ENV PATH="/root/.deno/bin:${PATH}"
ENV DENO_DIR=/tmp/deno

# Runtime data directory (separate from the source tree).
RUN mkdir -p /data
VOLUME ["/data"]
ENV CHALIE_DB_PATH=/data/chalie.db

EXPOSE 8081

CMD ["bash", "/root/.chalie/app/run.sh", "--port=8081", "--host=0.0.0.0"]
