# Install verification

Proves the Chalie install experience works on a real, pristine system — one
distro at a time — by running the actual installer end to end and then proving
the installed instance genuinely works, subsystem by subsystem.

Each row spins up an **official, unmodified distro image** (nothing pre-baked),
mounts in `installer/install.sh`, and runs it exactly as a first-time user would
(`install.sh --branch <ref>`, which self-fetches that branch's source). It then
audits that the dependencies actually installed — `onnxruntime` imports and lists
its execution providers, the core modules import, and the installer-managed
artifacts (voice models, Playwright browser, Deno, the CLI) are present — boots
Chalie, and finally exercises every subsystem a user touches on day one.

## What a row has to prove

A row passes only when **all** of these hold. "The process started" is not the
claim being tested; "a user can use it" is.

| Probe | Proof |
|---|---|
| readiness | `GET /ready` returns `200`, meaning database, memory store, embeddings and ONNX classifier heads each report `ok` |
| webserver | `GET /` serves the built interface bundle |
| chromium | Playwright launches Chromium, navigates to the live instance and reads its title |
| voice | Kokoro synthesises speech and Moonshine transcribes that same audio back to text (running Silero VAD on the way) |
| deno | the bundled runtime executes a script and returns its stdout |

The voice probe is a round trip on purpose: importing `kokoro_onnx` proves a
wheel landed, but synthesising audio and reading it back proves both ONNX
sessions, the model files, `soundfile` and the VAD are all genuinely working.
Likewise Chromium navigates a real page rather than merely launching, because
launching only proves its shared objects resolved — not that the renderer runs.

## What it covers

The installer supports **macOS (Apple Silicon)** and **Linux (amd64/arm64)** on
two package managers: **apt** (Debian/Ubuntu) and **dnf** (Fedora/RHEL-family).
Every other platform is refused outright by the installer, so there is nothing
for this harness to verify there. The matrix mirrors exactly those code paths:

| Row group | Distros | Expectation |
|---|---|---|
| apt path | Debian 12, Ubuntu 24.04 (arm64 + amd64) | install, boot and every probe pass |
| dnf path | Fedora 40, Fedora 41 | install, boot and every probe pass |
| stock Python too old | Ubuntu 22.04 (3.10), AlmaLinux 9 (3.9) | same, from a uv-managed CPython 3.12 |
| no `python3` at all | Debian 12, Fedora 41 | same — on Debian the build deps supply 3.11, on Fedora nothing does, so a managed CPython 3.12 is provisioned |

`expect` values in `matrix.tsv`:

- **pass** — installer succeeds, dependencies resolve, Chalie boots, and every
  probe in the table above passes.
- **pass-managed-python** — everything `pass` requires, plus one assertion on top:
  the venv's interpreter resolves under `~/.local/share/uv/python` and reports
  3.12.x. A suitable stock `python3` is always preferred; only when the host has
  none — too old, or absent — does the installer fetch a self-contained CPython
  3.12 with `uv` and build the virtualenv from that. These rows are the hosts
  where that second path is the one taken, so they are what proves it works.

The `python` column says what the harness sets up *before* running the installer:
`stock` installs the distro's own `python3` first (official base images are more
stripped than a real machine — the Debian, Ubuntu and Fedora bases ship none at
all), while `none` leaves the host without one, which is a real starting state
and no longer a refusal.

Adding a row means committing to supporting that platform. Unsupported distros
are absent by design — the installer refuses them rather than producing an
instance that looks healthy but whose browser cannot start.

macOS (Apple Silicon) cannot run as a container — Docker cannot host macOS — so
the CI workflow covers it with a dedicated `macos-14` job that runs the same
`verify-inside.sh` directly on a real Apple-Silicon runner (`INSTALLER_PATH`
points the script at the checked-out installer instead of a container mount). The
container matrix here covers Linux; the same script runs unchanged on any Mac.

## Running it

Requires Docker. Rows run on their native architecture in CI; running the amd64
rows on an arm64 host (or vice versa) falls back to emulation, which is slow and
can make a package manager fail for reasons that have nothing to do with Chalie.

```bash
# whole matrix, current release-candidate branch
installer/verify/run-matrix.sh

# a specific branch ref
installer/verify/run-matrix.sh --branch main

# only certain rows
installer/verify/run-matrix.sh debian12 fedora41
```

Per-row logs and a one-word verdict land in `installer/verify/results/`
(`<row>.log`, `<row>.verdict`). Verdicts: `PASS`, `FAIL`. Only `PASS` is
acceptable; anything else — including a row that never produced a verdict — fails
the driver, because a silent non-result is exactly how a broken install reaches a
user.

Each run also records what an install actually costs, so the system requirements
published in the README stay measured numbers rather than estimates:

- **disk** — `du` over `~/.chalie`, the Playwright browser cache and `~/.local/bin`.
- **memory** — resident set of the running instance, sampled *after* every probe,
  by which point the daemon has warmed its voice and embedding models and served
  real traffic. That steady state is the honest figure; RSS at boot is not.

## Files

- `matrix.tsv` — the distro rows (image, platform, expectation, stock python).
- `run-matrix.sh` — host driver: one container per row, collects logs + verdicts.
- `verify-inside.sh` — runs inside each container: install → dependency audit →
  boot → the five runtime probes.
- `results/` — generated logs and verdicts (git-ignored).
