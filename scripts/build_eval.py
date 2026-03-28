#!/usr/bin/env python3
"""Deterministic eval harness for the /build skill.

This script is the SOLE AUTHORITY on whether a micro-improvement passes or fails.
Claude's judgment is irrelevant — only exit codes matter.

Modes:
  baseline  — Capture current metrics -> .build_baseline.json
  check     — Re-measure and compare against baseline -> exit 0 (pass) or 1 (fail)
  refresh   — Update baseline to current state (after a successful commit)

Gates (tuned for feature building, not cleanup):
  0. schema.sql and API blueprints must not be modified
  1. SLOC increase must be <= 300 (keeps it "micro")
  2. Total cyclomatic complexity increase must be <= 30
  3. No test that was passing may start failing (differential)
  4. Test count must not decrease (forces writing tests for new code)
  5. Import memory must not increase beyond 5%
  6. Import time must not increase beyond 15%
  7. No new import failures
  8. No new ruff F-series lint issues

Reuses measurement functions from simplify_eval.py to avoid duplication.
"""

import json
import sys
import time
from pathlib import Path

# Import shared measurement functions from simplify_eval
sys.path.insert(0, str(Path(__file__).parent))
from simplify_eval import (  # noqa: E402
    BACKEND_DIR,
    _check_dev_deps,
    check_protected_files,
    measure_complexity,
    measure_imports,
    measure_ruff,
    measure_sloc,
    run_tests,
)

BASELINE_FILE = BACKEND_DIR / ".build_baseline.json"

# --- Gate thresholds ---
MAX_SLOC_INCREASE = 300       # Keeps changes "micro"
MAX_CC_INCREASE = 30          # Bounded complexity growth
MEMORY_TOLERANCE = 1.05       # 5% noise tolerance
TIME_TOLERANCE = 1.15         # 15% noise tolerance (higher than simplify — build adds code)


# --- Orchestration ---

def capture_all():
    """Capture all metrics. Returns dict."""
    _check_dev_deps()
    print("=== BUILD EVAL: Capturing metrics ===")

    print("[1/5] SLOC (radon raw)...")
    sloc = measure_sloc()
    print(f"  Total SLOC: {sloc['total']}")

    print("[2/5] Complexity (radon cc)...")
    cc = measure_complexity()
    print(f"  Total CC: {cc['total']}")

    print("[3/5] Unit tests (pytest)...")
    tests = run_tests()
    print(f"  Passed: {tests['pass_count']}, Failed: {tests['fail_count']}, Errors: {tests['error_count']}")

    print("[4/5] Import health...")
    imports = measure_imports()
    print(f"  Modules: {imports['modules']}, Time: {imports['import_time_s']}s, Peak: {imports['memory_peak_mb']}MB")
    if imports["import_fails"]:
        print(f"  Import failures: {imports['import_fails']}")

    print("[5/5] Ruff lint issues...")
    ruff = measure_ruff()
    print(f"  Total issues: {ruff['total']} {ruff['breakdown']}")

    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "sloc": sloc,
        "complexity": cc,
        "tests": tests,
        "imports": imports,
        "ruff": ruff,
    }


def cmd_baseline():
    """Capture and save baseline."""
    metrics = capture_all()
    with open(BASELINE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBaseline saved to {BASELINE_FILE}")
    print(f"  SLOC={metrics['sloc']['total']}  CC={metrics['complexity']['total']}  "
          f"Tests={metrics['tests']['pass_count']}p/{metrics['tests']['fail_count']}f  "
          f"Mem={metrics['imports']['memory_peak_mb']}MB  Ruff={metrics['ruff']['total']}")


def cmd_check():
    """Check current state against baseline. Exit 0=pass, 1=fail, 2=error."""
    _check_dev_deps()
    if not BASELINE_FILE.exists():
        print("ERROR: No baseline found. Run 'baseline' mode first.")
        sys.exit(2)

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    print(f"Baseline from: {baseline['captured_at']}")

    failures = []

    # Gate 0: Protected files
    print("\n[Gate 0] Protected files...")
    violations = check_protected_files()
    if violations:
        failures.append(f"PROTECTED FILES MODIFIED: {violations}")
        print(f"  FAIL: {violations}")
    else:
        print("  PASS")

    # Gate 1: SLOC increase bounded
    print(f"[Gate 1] SLOC (max +{MAX_SLOC_INCREASE})...")
    sloc = measure_sloc()
    delta = sloc["total"] - baseline["sloc"]["total"]
    if delta > MAX_SLOC_INCREASE:
        failures.append(f"SLOC increased by {delta} (max {MAX_SLOC_INCREASE}): {baseline['sloc']['total']} -> {sloc['total']}")
        print(f"  FAIL: +{delta} lines (max +{MAX_SLOC_INCREASE})")
    else:
        print(f"  PASS: {delta:+d} lines ({baseline['sloc']['total']} -> {sloc['total']})")

    # Gate 2: Complexity increase bounded
    print(f"[Gate 2] Complexity (max +{MAX_CC_INCREASE})...")
    cc = measure_complexity()
    cc_delta = cc["total"] - baseline["complexity"]["total"]
    if cc_delta > MAX_CC_INCREASE:
        failures.append(f"Complexity increased by {cc_delta} (max {MAX_CC_INCREASE}): {baseline['complexity']['total']} -> {cc['total']}")
        print(f"  FAIL: +{cc_delta} (max +{MAX_CC_INCREASE})")
    else:
        print(f"  PASS: {cc_delta:+d} ({baseline['complexity']['total']} -> {cc['total']})")

    # Gate 3: No test regressions (differential)
    print("[Gate 3] Test regressions...")
    tests = run_tests()
    baseline_passed = set(baseline["tests"]["passed"])
    current_passed = set(tests["passed"])
    regressions = sorted(baseline_passed - current_passed)
    if regressions:
        show = regressions[:10]
        failures.append(f"Test regressions ({len(regressions)}): {show}")
        print(f"  FAIL: {len(regressions)} regressions")
        for t in show:
            print(f"    - {t}")
    else:
        new_passes = sorted(current_passed - baseline_passed)
        print(f"  PASS: {tests['pass_count']}p (was {baseline['tests']['pass_count']}p, +{len(new_passes)} new)")

    # Gate 4: Test count must not decrease
    print("[Gate 4] Test count...")
    if tests["pass_count"] < baseline["tests"]["pass_count"]:
        failures.append(
            f"Test count decreased: {baseline['tests']['pass_count']} -> {tests['pass_count']}"
        )
        print(f"  FAIL: {baseline['tests']['pass_count']} -> {tests['pass_count']}")
    else:
        print(f"  PASS: {tests['pass_count']}p (was {baseline['tests']['pass_count']}p)")

    # Gate 5: Import memory
    print("[Gate 5] Import memory...")
    imports = measure_imports()
    mem_baseline = baseline["imports"]["memory_peak_mb"]
    mem_threshold = mem_baseline * MEMORY_TOLERANCE
    if imports["memory_peak_mb"] > mem_threshold:
        failures.append(
            f"Memory increased: {mem_baseline}MB -> {imports['memory_peak_mb']}MB "
            f"(threshold {mem_threshold:.2f}MB)"
        )
        print(f"  FAIL: {mem_baseline}MB -> {imports['memory_peak_mb']}MB (limit {mem_threshold:.2f}MB)")
    else:
        print(f"  PASS: {imports['memory_peak_mb']}MB (baseline {mem_baseline}MB)")

    # Gate 6: Import time
    print("[Gate 6] Import time...")
    time_baseline = baseline["imports"]["import_time_s"]
    time_threshold = time_baseline * TIME_TOLERANCE
    if imports["import_time_s"] > time_threshold:
        failures.append(
            f"Import time increased: {time_baseline}s -> {imports['import_time_s']}s "
            f"(threshold {time_threshold:.2f}s)"
        )
        print(f"  FAIL: {time_baseline}s -> {imports['import_time_s']}s (limit {time_threshold:.2f}s)")
    else:
        print(f"  PASS: {imports['import_time_s']}s (baseline {time_baseline}s)")

    # Gate 7: No new import failures
    print("[Gate 7] Import health...")
    baseline_fails = set(baseline["imports"].get("import_fails", []))
    current_fails = set(imports.get("import_fails", []))
    new_fails = sorted(current_fails - baseline_fails)
    if new_fails:
        failures.append(f"New import failures ({len(new_fails)}): {new_fails[:5]}")
        print(f"  FAIL: {len(new_fails)} new import failures")
        for f_item in new_fails[:5]:
            print(f"    - {f_item}")
    else:
        print(f"  PASS: {imports['modules']} modules import OK")

    # Gate 8: No new ruff F-series issues
    print("[Gate 8] Ruff lint...")
    ruff = measure_ruff()
    ruff_delta = ruff["total"] - baseline["ruff"]["total"]
    if ruff_delta > 0:
        failures.append(f"New ruff issues: {ruff_delta} new ({baseline['ruff']['total']} -> {ruff['total']})")
        print(f"  FAIL: +{ruff_delta} new issues ({ruff['breakdown']})")
    else:
        print(f"  PASS: {ruff['total']} issues ({ruff_delta:+d} from baseline)")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED -- {len(failures)} gate(s) violated:\n")
        for i, f_msg in enumerate(failures, 1):
            print(f"  {i}. {f_msg}")
        print("\nVerdict: Fix the violations or simplify the implementation.")
        sys.exit(1)
    else:
        print("ALL GATES PASSED")
        print(f"  SLOC: {delta:+d}  CC: {cc_delta:+d}  Ruff: {ruff_delta:+d}  "
              f"Mem: {imports['memory_peak_mb']}MB  Tests: {tests['pass_count']}p")
        print("\nVerdict: Safe to commit.")
        sys.exit(0)


def cmd_refresh():
    """Update baseline to current state (call after successful commit)."""
    print("Refreshing baseline after successful commit...")
    cmd_baseline()


# --- Entry point ---

COMMANDS = {
    "baseline": cmd_baseline,
    "check": cmd_check,
    "refresh": cmd_refresh,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} [{' | '.join(COMMANDS)}]")
        sys.exit(2)
    COMMANDS[sys.argv[1]]()
