#!/usr/bin/env python3
"""Deterministic eval harness for the /simplify skill.

This script is the SOLE AUTHORITY on whether a change passes or fails.
Claude's judgment is irrelevant — only exit codes matter.

Modes:
  baseline  — Capture current metrics -> .simplify_baseline.json
  check     — Re-measure and compare against baseline -> exit 0 (pass) or 1 (fail)
  refresh   — Update baseline to current state (after a successful commit)

Non-negotiable gates:
  1. SLOC must not increase (excluding comments/docstrings)
  2. Cyclomatic complexity must not increase
  3. No test that was passing may start failing (differential)
  4. Import memory must not increase beyond noise threshold (5%)
  5. Import time must not increase beyond noise threshold (10%)
  6. schema.sql and API blueprints must not be modified
"""

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
BASELINE_FILE = BACKEND_DIR / ".simplify_baseline.json"


def _check_dev_deps():
    """Verify dev tools are installed (from requirements-dev.txt)."""
    missing = []
    for mod in ("ruff", "radon"):
        r = subprocess.run(
            [sys.executable, "-m", mod, "--version"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            missing.append(mod)
    if missing:
        print(f"ERROR: Missing dev tools: {missing}")
        print("Install with: pip install -r backend/requirements-dev.txt")
        sys.exit(2)
TEST_XML = BACKEND_DIR / ".simplify_test_results.xml"
PROTECTED_PATTERNS = [
    "schema.sql",
    "api/",
]

# --- Measurement functions ---

def measure_sloc():
    """Count source lines of code via radon raw (excludes comments, docstrings, blanks)."""
    result = subprocess.run(
        [sys.executable, "-m", "radon", "raw", "-s", "-j",
         "-e", "data/*,__pycache__/*,.venv/*,venv/*,tests/*",
         str(BACKEND_DIR)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"radon raw failed: {result.stderr[:500]}")
        sys.exit(2)
    data = json.loads(result.stdout)
    total_sloc = 0
    per_file = {}
    for filepath, metrics in data.items():
        sloc = metrics.get("sloc", 0)
        total_sloc += sloc
        per_file[filepath] = sloc
    return {"total": total_sloc, "per_file": per_file}


def measure_complexity():
    """Sum cyclomatic complexity across all functions/methods via radon cc."""
    result = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "-s", "-j", "-a",
         "-e", "data/*,__pycache__/*,.venv/*,venv/*,tests/*",
         str(BACKEND_DIR)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"radon cc failed: {result.stderr[:500]}")
        sys.exit(2)
    data = json.loads(result.stdout)
    total_cc = 0
    per_file = {}
    for filepath, blocks in data.items():
        if not isinstance(blocks, list):
            continue
        file_cc = sum(b.get("complexity", 0) for b in blocks if isinstance(b, dict))
        total_cc += file_cc
        per_file[filepath] = file_cc
    return {"total": total_cc, "per_file": per_file}


def run_tests():
    """Run unit tests, capture individual pass/fail via JUnit XML."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "unit", "-q", "--tb=no",
         f"--junitxml={TEST_XML}", "-x"],
        capture_output=True, text=True,
        cwd=str(BACKEND_DIR),
        timeout=600
    )

    if not TEST_XML.exists():
        print(f"pytest did not produce JUnit XML. stderr: {result.stderr[:500]}")
        sys.exit(2)

    tree = ET.parse(TEST_XML)
    root = tree.getroot()

    passed, failed, errored, skipped = set(), set(), set(), set()
    for suite in root.iter("testsuite"):
        for case in suite.iter("testcase"):
            name = f"{case.get('classname', '')}.{case.get('name', '')}"
            if case.find("failure") is not None:
                failed.add(name)
            elif case.find("error") is not None:
                errored.add(name)
            elif case.find("skipped") is not None:
                skipped.add(name)
            else:
                passed.add(name)

    return {
        "passed": sorted(passed),
        "failed": sorted(failed),
        "errored": sorted(errored),
        "skipped": sorted(skipped),
        "pass_count": len(passed),
        "fail_count": len(failed),
        "error_count": len(errored),
    }


def measure_imports():
    """Measure import time and peak memory for all service modules."""
    script = """
import tracemalloc, importlib, pkgutil, time, json, sys
sys.path.insert(0, ".")
tracemalloc.start()
start = time.monotonic()
count, fails = 0, []
for info in pkgutil.walk_packages(["services"], prefix="services."):
    try:
        importlib.import_module(info.name)
        count += 1
    except Exception as e:
        fails.append(f"{info.name}: {e}")
elapsed = time.monotonic() - start
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({
    "modules": count,
    "import_fails": fails,
    "import_time_s": round(elapsed, 3),
    "memory_peak_mb": round(peak / 1024 / 1024, 2),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        cwd=str(BACKEND_DIR),
        timeout=120
    )
    if result.returncode != 0:
        print(f"Import measurement failed: {result.stderr[:500]}")
        sys.exit(2)
    return json.loads(result.stdout.strip().split("\n")[-1])


def check_protected_files():
    """Ensure schema.sql and API blueprints have no uncommitted changes."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True,
        cwd=str(BACKEND_DIR.parent)
    )
    changed = result.stdout.strip().split("\n") if result.stdout.strip() else []
    violations = []
    for f in changed:
        for pattern in PROTECTED_PATTERNS:
            if pattern in f:
                violations.append(f)
    return violations


def measure_ruff():
    """Count ruff lint issues (F-series: pyflakes)."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=F", "--statistics",
         "-q", "--exclude=data,__pycache__,.venv,venv",
         str(BACKEND_DIR)],
        capture_output=True, text=True, timeout=60
    )
    # ruff returns non-zero when it finds issues, that's expected
    total = 0
    breakdown = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            count = int(parts[0])
            code = parts[1]
            total += count
            breakdown[code] = count
    return {"total": total, "breakdown": breakdown}


# --- Orchestration ---

def capture_all():
    """Capture all metrics. Returns dict."""
    _check_dev_deps()
    print("=== SIMPLIFY EVAL: Capturing metrics ===")

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

    # Gate 1: SLOC must not increase
    print("[Gate 1] SLOC...")
    sloc = measure_sloc()
    delta = sloc["total"] - baseline["sloc"]["total"]
    if delta > 0:
        failures.append(f"SLOC increased by {delta}: {baseline['sloc']['total']} -> {sloc['total']}")
        print(f"  FAIL: +{delta} lines ({baseline['sloc']['total']} -> {sloc['total']})")
    else:
        print(f"  PASS: {delta} lines ({baseline['sloc']['total']} -> {sloc['total']})")

    # Gate 2: Complexity must not increase
    print("[Gate 2] Complexity...")
    cc = measure_complexity()
    cc_delta = cc["total"] - baseline["complexity"]["total"]
    if cc_delta > 0:
        failures.append(f"Complexity increased by {cc_delta}: {baseline['complexity']['total']} -> {cc['total']}")
        print(f"  FAIL: +{cc_delta} ({baseline['complexity']['total']} -> {cc['total']})")
    else:
        print(f"  PASS: {cc_delta} ({baseline['complexity']['total']} -> {cc['total']})")

    # Gate 3: No test regressions (differential)
    print("[Gate 3] Test differential...")
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

    # Gate 4: Resource usage — memory
    print("[Gate 4] Import memory...")
    imports = measure_imports()
    mem_baseline = baseline["imports"]["memory_peak_mb"]
    mem_threshold = mem_baseline * 1.05  # 5% tolerance for measurement noise
    if imports["memory_peak_mb"] > mem_threshold:
        failures.append(
            f"Memory increased: {mem_baseline}MB -> {imports['memory_peak_mb']}MB "
            f"(threshold {mem_threshold:.2f}MB)"
        )
        print(f"  FAIL: {mem_baseline}MB -> {imports['memory_peak_mb']}MB (limit {mem_threshold:.2f}MB)")
    else:
        print(f"  PASS: {imports['memory_peak_mb']}MB (baseline {mem_baseline}MB)")

    # Gate 5: Resource usage — import time
    print("[Gate 5] Import time...")
    time_baseline = baseline["imports"]["import_time_s"]
    time_threshold = time_baseline * 1.10  # 10% tolerance
    if imports["import_time_s"] > time_threshold:
        failures.append(
            f"Import time increased: {time_baseline}s -> {imports['import_time_s']}s "
            f"(threshold {time_threshold:.2f}s)"
        )
        print(f"  FAIL: {time_baseline}s -> {imports['import_time_s']}s (limit {time_threshold:.2f}s)")
    else:
        print(f"  PASS: {imports['import_time_s']}s (baseline {time_baseline}s)")

    # Gate 6: New import failures
    print("[Gate 6] Import health...")
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

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED — {len(failures)} gate(s) violated:\n")
        for i, f_msg in enumerate(failures, 1):
            print(f"  {i}. {f_msg}")
        print("\nVerdict: REVERT changes and try a different approach.")
        sys.exit(1)
    else:
        ruff = measure_ruff()
        ruff_delta = ruff["total"] - baseline["ruff"]["total"]
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
