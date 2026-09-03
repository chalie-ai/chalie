#!/usr/bin/env bash
# Host driver for the install-verification matrix.
#
# For each row in matrix.tsv it launches a pristine distro container, mounts in
# the branch installer (installer/install.sh) plus verify-inside.sh, runs the
# real install end to end inside that container, and records the per-row log and
# final VERDICT under results/. A one-line summary is printed at the end.
#
# The container base images are pristine official distros — nothing is pre-baked,
# so what the installer must do (build deps, uv, Python venv, native wheels,
# Playwright, voice models, Deno) is exactly what a first-time user's machine
# would require.
#
# Usage:
#   ./run-matrix.sh                         # whole matrix, branch rc-1.2.0
#   ./run-matrix.sh --branch main           # a different branch ref
#   ./run-matrix.sh debian12 fedora41       # only the named rows
#   ./run-matrix.sh --results /tmp/out ...  # custom results dir
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
INSTALLER="$REPO/installer/install.sh"
MATRIX="$HERE/matrix.tsv"

BRANCH="rc-1.2.0"
RESULTS="$HERE/results"
rows=()
while [ $# -gt 0 ]; do
  case "$1" in
    --branch)  BRANCH="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    -*)        echo "unknown flag: $1" >&2; exit 2 ;;
    *)         rows+=("$1"); shift ;;
  esac
done
mkdir -p "$RESULTS"

[ -f "$INSTALLER" ] || { echo "installer not found: $INSTALLER" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "docker not found" >&2; exit 1; }

run_row() {
  local name="$1" image="$2" plat="$3" expect="$4"
  local log="$RESULTS/$name.log"
  echo ">>> $name  ($image  $plat  expect=$expect)"
  docker run --rm --platform "$plat" \
    -v "$INSTALLER:/verify/install.sh:ro" \
    -v "$HERE/verify-inside.sh:/verify/verify-inside.sh:ro" \
    -e CHALIE_BRANCH="$BRANCH" -e ROW_NAME="$name" \
    -e ROW_EXPECT="$expect" \
    "$image" bash /verify/verify-inside.sh \
    >"$log" 2>&1
  local verdict
  verdict="$(grep -oE 'VERDICT=[A-Z-]+' "$log" | tail -1 | cut -d= -f2)"
  echo "${verdict:-NO-VERDICT}" >"$RESULTS/$name.verdict"
  echo "<<< $name  ->  ${verdict:-NO-VERDICT}"
}

selected() {
  [ ${#rows[@]} -eq 0 ] && return 0
  printf '%s\n' "${rows[@]}" | grep -qx "$1"
}

while IFS=$'\t' read -r name image plat expect; do
  case "$name" in ''|\#*) continue ;; esac
  selected "$name" || continue
  run_row "$name" "$image" "$plat" "$expect"
done < "$MATRIX"

echo
echo "=== SUMMARY (branch $BRANCH) ==="
bad=0
for v in "$RESULTS"/*.verdict; do
  [ -f "$v" ] || continue
  verdict="$(cat "$v")"
  printf '  %-22s %s\n' "$(basename "$v" .verdict)" "$verdict"
  # Only PASS (everything installed, booted and every runtime probe green) and
  # REFUSED (a row whose whole claim is that the installer refuses old Python)
  # are acceptable. Anything else gates, a missing verdict included: a row that
  # produced no result has proved nothing, and a silent non-result is exactly
  # how a broken install reaches a user.
  case "$verdict" in PASS|REFUSED) ;; *) bad=1 ;; esac
done
exit "$bad"
