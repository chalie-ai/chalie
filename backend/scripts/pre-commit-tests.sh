#!/usr/bin/env bash
# Pre-commit gate: run unit tests before allowing commit.
# Fails fast on first error — run `pytest -m unit -v` for full diagnostics.

set -e

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BACKEND_DIR"

echo "Running unit tests..."
python -m pytest -x -q --tb=line -m unit

echo "All unit tests passed."
