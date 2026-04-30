FROM python:3.12-slim

# install.sh passthrough args (build-time).
# Use the SAME flag names install.sh accepts: --disable-voice, --branch=NAME.
# Example: docker build --build-arg INSTALL_ARGS="--disable-voice --branch=rc-0.4.0" .
ARG INSTALL_ARGS=

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates bash tar \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://raw.githubusercontent.com/chalie-ai/chalie/main/installer/install.sh -o /tmp/install.sh \
 && bash /tmp/install.sh $INSTALL_ARGS \
 && rm /tmp/install.sh

ENV PATH="/root/.deno/bin:${PATH}"
ENV DENO_DIR=/tmp/deno
ENV CHALIE_DB_PATH=/data/chalie.db

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8081

# run.sh passthrough args (runtime).
# Use the SAME flag names run.sh accepts: --port[=]N, --host[=]H, --no-voice.
# Example: docker run chalieai/chalie --port=9000 --no-voice
ENTRYPOINT ["bash", "/root/.chalie/app/run.sh"]
CMD ["--port=8081", "--host=0.0.0.0"]
