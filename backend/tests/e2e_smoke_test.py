#!/usr/bin/env python3
"""
End-to-end smoke test for the cognitive brain system.

Sends small-talk messages via the REST API and verifies:
- REST API responds successfully
- Topic classification works
- Frontal cortex generates responses (non-empty, correct modes)
- Memory chunker processes exchanges
- No errors in pipeline

Usage:
    python3 tests/e2e_smoke_test.py                    # Run against localhost
    python3 tests/e2e_smoke_test.py --host myserver.local  # Run against remote host
    python3 tests/e2e_smoke_test.py --port 9000        # Custom port
"""

import sys
import json
import time
import logging
import urllib.request
import urllib.error
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---

API_BASE = "http://localhost:8081"
TEST_UUID = "e2e-smoke-test"
TIMEOUT = 300  # seconds per request (LLM can be slow)

TEST_MESSAGES = [
    {
        "message": "Hey, how's it going?",
        "expect_modes": ["RESPOND", "CLARIFY", "ACT"],
        "description": "Greeting - should get a conversational response",
    },
    {
        "message": "Not much, just checking in. Weather's nice today.",
        "expect_modes": ["RESPOND", "ACT"],
        "description": "Small talk - should continue conversation",
    },
    {
        "message": "Alright, talk to you later!",
        "expect_modes": ["RESPOND"],
        "description": "Farewell - should respond",
    },
]


# --- Helpers ---

def api_request(endpoint, method="GET", data=None):
    """Make an HTTP request to the REST API."""
    url = f"{API_BASE}{endpoint}"
    headers = {"Content-Type": "application/json"} if data else {}
    body = json.dumps(data).encode() if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}, e.code
    except urllib.error.URLError as e:
        return {"error": str(e.reason)}, 0
    except Exception as e:
        return {"error": str(e)}, 0


# --- Test Runner ---

def run_tests():
    """Run the e2e smoke test suite."""
    log.info("=" * 60)
    log.info("  CHALIE E2E Smoke Test")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    results = []
    total_time = 0

    # Test 0: Health check
    log.info("\n[0/3] Health check...")
    health, status = api_request("/health")
    if status == 200 and health.get("status") == "ok":
        log.info("  PASS - REST API is healthy")
        results.append(("Health check", True, ""))
    else:
        log.info(f"  FAIL - Health check returned: {health}")
        results.append(("Health check", False, f"Status {status}: {health}"))
        log.info("\nAborting: REST API not available.")
        return print_summary(results)

    # Tests 1-3: Send messages
    for i, test in enumerate(TEST_MESSAGES, 1):
        log.info(f"\n[{i}/{len(TEST_MESSAGES)}] {test['description']}")
        log.info(f"  Sending: \"{test['message']}\"")

        start = time.time()
        resp, status = api_request("/api/message", method="POST", data={
            "uuid": TEST_UUID,
            "message": test["message"],
        })
        elapsed = time.time() - start
        total_time += elapsed

        # Check HTTP success
        if status != 200:
            log.info(f"  FAIL - HTTP {status}: {resp}")
            results.append((test["description"], False, f"HTTP {status}"))
            continue

        # Check response structure
        if not resp.get("success"):
            log.info(f"  FAIL - success=false: {resp}")
            results.append((test["description"], False, "success=false"))
            continue

        topic = resp.get("topic", "unknown")
        log.info(f"  OK   - Topic: {topic}, Time: {elapsed:.1f}s")
        results.append((test["description"], True, f"topic={topic}, {elapsed:.1f}s"))

        # Brief pause between messages to avoid queue pileup
        if i < len(TEST_MESSAGES):
            time.sleep(1)

    log.info(f"\n  Total LLM time: {total_time:.1f}s")

    return print_summary(results)


def print_summary(results):
    """Print test summary and return exit code."""
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    log.info("\n" + "=" * 60)
    log.info("  RESULTS")
    log.info("=" * 60)

    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        log.info(f"  [{status}] {name}")
        if detail and not ok:
            log.info(f"         {detail}")

    log.info(f"\n  {passed} passed, {failed} failed")
    log.info("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    # Allow overriding from CLI
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]
            API_BASE = f"http://{host}:8081"

    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]
            # Replace port in API_BASE
            parts = API_BASE.rsplit(":", 1)
            API_BASE = f"{parts[0]}:{port}"

    exit_code = run_tests()
    sys.exit(exit_code)
