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
  local name="$1" image="$2" plat="$3" expect="$4" do_boot="$5"
  local log="$RESULTS/$name.log"
  echo ">>> $name  ($image  $plat  expect=$expect  boot=$do_boot)"
  # If bash is absent the installer (bash shebang + bashisms) cannot run at all —
  # record that as a confirmed gap rather than a container-exec error.
  docker run --rm --platform "$plat" \
    -v "$INSTALLER:/verify/install.sh:ro" \
    -v "$HERE/verify-inside.sh:/verify/verify-inside.sh:ro" \
    -e CHALIE_BRANCH="$BRANCH" -e ROW_NAME="$name" \
    -e ROW_EXPECT="$expect" -e DO_BOOT="$do_boot" \
    "$image" \
    sh -c 'if command -v bash >/dev/null 2>&1; then exec bash /verify/verify-inside.sh; else printf "RESULT installer_rc=n/a\nRESULT reason=bash-absent(installer requires bash)\nVERDICT=GAP-CONFIRMED %s\n" "$ROW_NAME"; fi' \
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

while IFS=$'\t' read -r name image plat expect do_boot; do
  case "$name" in ''|\#*) continue ;; esac
  selected "$name" || continue
  run_row "$name" "$image" "$plat" "$expect" "$do_boot"
done < "$MATRIX"

echo
echo "=== SUMMARY (branch $BRANCH) ==="
bad=0
for v in "$RESULTS"/*.verdict; do
  [ -f "$v" ] || continue
  verdict="$(cat "$v")"
  printf '  %-22s %s\n' "$(basename "$v" .verdict)" "$verdict"
  # PASS / REFUSED / GAP-CONFIRMED / UNEXPECTED-PASS are all as-expected-or-better.
  # INCONCLUSIVE means the environment (not the installer) could not be established
  # — e.g. a package manager that will not run under CPU emulation — so it is
  # surfaced but never gates; a native-arch CI runner turns it into a real verdict.
  # FAIL (install broke where it should have worked) and NO-VERDICT (the run never
  # produced a verdict) are the only real problems — fail the driver so CI gates.
  case "$verdict" in FAIL|NO-VERDICT|'') bad=1 ;; esac
done
exit "$bad"
