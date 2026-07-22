# Install verification

Proves the Chalie install experience works on a real, pristine system — one
distro at a time — by running the actual installer end to end and confirming
every dependency resolves and the app reaches its readiness contract.

Each row spins up an **official, unmodified distro image** (nothing pre-baked),
mounts in `installer/install.sh`, and runs it exactly as a first-time user would
(`install.sh --branch <ref>`, which self-fetches that branch's source). It then
audits that the dependencies actually installed — `onnxruntime` imports and lists
its execution providers, the core modules import, and the installer-managed
artifacts (voice models, Playwright browser, Deno, the CLI) are present — and,
where a green boot is the claim, starts Chalie and waits for `GET /ready` to
return `200` with every preflight component healthy.

## What it covers

The installer supports **macOS (Apple Silicon)** and **Linux (amd64/arm64)**, and
knows two Linux package managers: **apt** (Debian/Ubuntu) and **dnf**
(Fedora/RHEL-family). Anything else is an unhandled distro. The matrix is built
around those real code paths:

| Row group | Distros | Expectation |
|---|---|---|
| apt path | Debian 12, Ubuntu 24.04 | install + boot succeed |
| dnf path | Fedora 40/41 | install + boot succeed |
| stock Python too old | Ubuntu 22.04 (3.10), AlmaLinux 9 (3.9) | installer **refuses** — correct behaviour |
| unhandled distro | Arch, openSUSE Tumbleweed, Alpine | documented **gap** — no apt/dnf branch |

`expect` values in `matrix.tsv`:

- **pass** — installer succeeds, dependencies resolve, and (when `do_boot=1`)
  `/ready` returns `200`.
- **refuse-old-python** — the distro's stock `python3` is older than 3.11, so the
  installer must refuse. The installer checks for Python 3.11+ but deliberately
  never installs it; the harness first installs each distro's stock `python3`
  (no version bump) to reproduce a realistic machine, so the pass/refuse outcome
  is driven by what that distro actually ships.
- **gap** — an unhandled distro (no apt/dnf branch, or no `bash` for the installer
  to run under). Failure is the documented expectation; `UNEXPECTED-PASS` means
  prebuilt wheels happened to cover it.

macOS (Apple Silicon) cannot run as a container — Docker cannot host macOS — so
the repeatable mechanism for it is a real Apple-Silicon runner (GitHub's
`macos-14`) or running the installer directly on a Mac; the container matrix here
covers Linux only. macOS on Intel and any non-Linux/macOS OS are refused by the
installer by design.

## Running it

Requires Docker with `linux/amd64` + `linux/arm64` (Buildx emulation covers the
non-native arch).

```bash
# whole matrix, current release-candidate branch
installer/verify/run-matrix.sh

# a specific branch ref
installer/verify/run-matrix.sh --branch main

# only certain rows
installer/verify/run-matrix.sh debian12 fedora41
```

Per-row logs and a one-word verdict land in `installer/verify/results/`
(`<row>.log`, `<row>.verdict`). Verdicts: `PASS`, `REFUSED`, `GAP-CONFIRMED`,
`UNEXPECTED-PASS`, `FAIL`. Only `FAIL` (and a missing verdict) is a real problem —
the others are the expected outcome for their row.

## Files

- `matrix.tsv` — the distro rows (image, platform, expectation, whether to boot).
- `run-matrix.sh` — host driver: one container per row, collects logs + verdicts.
- `verify-inside.sh` — runs inside each container: install → dependency audit →
  boot + `/ready`.
- `results/` — generated logs and verdicts (git-ignored).
