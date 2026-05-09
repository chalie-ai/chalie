FROM python:3.12-slim

# Build-time install.sh flags.
# One ARG per install.sh flag, named after the flag itself.
#   --disable-voice  → ARG DISABLE_VOICE=1 (any non-empty value enables)
#   --branch=NAME    → ARG BRANCH=NAME
# Example: docker build --build-arg DISABLE_VOICE=1 --build-arg BRANCH=rc-0.4.0 .
ARG DISABLE_VOICE=
ARG BRANCH=

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates bash tar \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://raw.githubusercontent.com/chalie-ai/chalie/main/installer/install.sh -o /tmp/install.sh \
 && bash /tmp/install.sh ${DISABLE_VOICE:+--disable-voice} ${BRANCH:+--branch="$BRANCH"} \
 && rm /tmp/install.sh

ENV PATH="/root/.deno/bin:${PATH}"
ENV DENO_DIR=/tmp/deno

# Persistent runtime state. The application code lives at /root/.chalie/app/,
# so data/ is alongside backend/ and resolves the same way as a local checkout.
RUN mkdir -p /root/.chalie/app/data
VOLUME ["/root/.chalie/app/data"]
EXPOSE 8081

# Runtime run.sh flags pass through unchanged via ENTRYPOINT + CMD:
#   docker run chalieai/chalie                       → uses CMD defaults
#   docker run chalieai/chalie --port=9000           → run.sh --port=9000
#   docker run chalieai/chalie --no-voice            → run.sh --no-voice
#   docker run chalieai/chalie --host=127.0.0.1      → run.sh --host=127.0.0.1
ENTRYPOINT ["bash", "/root/.chalie/app/run.sh"]
CMD ["--port=8081", "--host=0.0.0.0"]
