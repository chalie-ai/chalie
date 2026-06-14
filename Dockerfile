# ── Stage 1: build the Vue 3 frontend ────────────────────────────────
# dist/ is gitignored, so the source tarball install.sh extracts in the runtime
# stage has no Vue build. This stage produces it. The two apps build to
# apps/interface/dist (chat SPA + the login & on-boarding multi-page entries)
# and apps/brain/dist (admin SPA, asset base /brain/).
FROM node:22-slim AS frontend-builder

# Pin pnpm to the version that generated pnpm-lock.yaml (lockfileVersion 9.0)
# and understands pnpm-workspace.yaml's allowBuilds key. Installing via npm
# avoids corepack's stale-signing-key failures on newer pnpm releases and keeps
# --frozen-lockfile deterministic regardless of the base image's bundled pnpm.
RUN npm install -g pnpm@11.6.0

WORKDIR /build
# Copy frontend source only (the .dockerignore keeps node_modules/ and dist/ out).
COPY frontend/ ./frontend/
WORKDIR /build/frontend
RUN pnpm install --frozen-lockfile \
 && pnpm -r build

# ── Stage 2: runtime ─────────────────────────────────────────────────
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

# Overlay the pre-built Vue dist dirs onto the source tree install.sh extracted.
# install.sh leaves frontend/apps/{interface,brain}/ in place (only dist/ is
# absent, being gitignored) and there is no Node in this stage — this COPY is
# what makes the app's routes resolve on a fresh install instead of 404ing.
COPY --from=frontend-builder /build/frontend/apps/interface/dist /root/.chalie/app/frontend/apps/interface/dist
COPY --from=frontend-builder /build/frontend/apps/brain/dist /root/.chalie/app/frontend/apps/brain/dist

# Persistent runtime state. The application code lives at /root/.chalie/app/,
# so data/ is alongside backend/ and resolves the same way as a local checkout.
RUN mkdir -p /root/.chalie/app/data
VOLUME ["/root/.chalie/app/data"]
EXPOSE 31025

# Voice and playwright are installed at runtime by RuntimeDepsService
# based on user settings — not baked into the image.
ENTRYPOINT ["bash", "/root/.chalie/app/run.sh"]
CMD ["--port=31025", "--host=0.0.0.0"]
