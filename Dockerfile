# ── Runtime image ─────────────────────────────────────────────────────
# Single stage, no Node/pnpm. install.sh fetches ALL source from the GitHub
# tarball, which now ships the committed Vue dist (frontend/apps/{interface,brain}/dist,
# in-repo per TKT-993) — so the app's routes resolve on a fresh install with no
# frontend build step. End users never build the frontend; shipping dist serves
# everyone, this image included.
FROM python:3.12-slim

# Build-time install.sh flags.
#   --branch=NAME    → ARG BRANCH=NAME    (uses refs/heads/NAME tarball)
#   --tag=NAME       → ARG TAG=NAME       (uses refs/tags/NAME tarball, no API lookup)
# Example: docker build --build-arg BRANCH=rc-0.8.0 .
# The release workflow passes TAG=${{ github.ref_name }} so install.sh skips the
# GitHub release-API lookup, which races with release-publication on tag pushes.
ARG BRANCH=
ARG TAG=

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates bash tar \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://raw.githubusercontent.com/chalie-ai/chalie/main/installer/install.sh -o /tmp/install.sh \
 && bash /tmp/install.sh ${BRANCH:+--branch="$BRANCH"} ${TAG:+--tag="$TAG"} \
 && rm /tmp/install.sh

# Persistent runtime state. The application code lives at /root/.chalie/app/,
# so data/ is alongside backend/ and resolves the same way as a local checkout.
RUN mkdir -p /root/.chalie/app/data
VOLUME ["/root/.chalie/app/data"]
EXPOSE 31025

# Voice and playwright are installed at runtime by RuntimeDepsService
# based on user settings — not baked into the image.
ENTRYPOINT ["bash", "/root/.chalie/app/run.sh"]
CMD ["--port=31025", "--host=0.0.0.0"]
