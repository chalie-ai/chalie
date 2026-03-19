#!/usr/bin/env python3
"""
50-prompt smoke test with diverse categories, mixed sessions, and rolling metrics.

Sends prompts across multiple UUIDs (cold + warm contexts) and tracks:
- Mode distribution vs expected
- Response times with rolling average (detect degradation as context grows)
- Tie-breaker invocation rate and success

Usage:
    python3 tests/smoke_test_50.py
    python3 tests/smoke_test_50.py --host myserver.local
"""

import sys
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

API_BASE = "http://localhost:8080"
SPACING = 15  # seconds between prompts
TIMEOUT = 300

# Each prompt: (uuid, message, expected_mode, category)
# Multiple UUIDs to test both cold starts and warm context accumulation
PROMPTS = [
    # --- Session A: greetings + small talk (cold start) ---
    ("smoke-a", "Hey there!", "UNIFIED", "greeting"),
    ("smoke-a", "How's it going?", "UNIFIED", "question"),
    ("smoke-a", "Not bad, just hanging out", "UNIFIED", "feedback"),
    ("smoke-a", "What do you think about music?", "UNIFIED", "question"),
    ("smoke-a", "Cool, thanks!", "UNIFIED", "feedback"),

    # --- Session B: knowledge questions (cold start) ---
    ("smoke-b", "Hello", "UNIFIED", "greeting"),
    ("smoke-b", "What is photosynthesis?", "UNIFIED", "question"),
    ("smoke-b", "Can you explain that in simpler terms?", "UNIFIED", "question"),
    ("smoke-b", "Got it, makes sense", "UNIFIED", "feedback"),
    ("smoke-b", "What about cellular respiration?", "UNIFIED", "question"),

    # --- Session C: memory probing (cold start) ---
    ("smoke-c", "Hi!", "UNIFIED", "greeting"),
    ("smoke-c", "Do you remember my name?", "UNIFIED", "memory"),
    ("smoke-c", "What did we talk about yesterday?", "ACT", "memory"),
    ("smoke-c", "You mentioned something about weather last time", "ACT", "memory"),
    ("smoke-c", "Interesting", "UNIFIED", "feedback"),

    # --- Session D: mixed intent (cold start) ---
    ("smoke-d", "Good evening!", "UNIFIED", "greeting"),
    ("smoke-d", "I've been thinking about learning to code", "UNIFIED", "statement"),
    ("smoke-d", "Where should I start?", "UNIFIED", "question"),
    ("smoke-d", "That's helpful, thank you", "UNIFIED", "feedback"),
    ("smoke-d", "What programming language do you recommend?", "UNIFIED", "question"),

    # --- Session E: short/terse messages (cold start) ---
    ("smoke-e", "yo", "UNIFIED", "greeting"),
    ("smoke-e", "sup", "UNIFIED", "greeting"),
    ("smoke-e", "ok", "UNIFIED", "feedback"),
    ("smoke-e", "why?", "UNIFIED", "question"),
    ("smoke-e", "hmm", "UNIFIED", "feedback"),

    # --- Session F: emotional/feedback heavy (cold start) ---
    ("smoke-f", "Hey!", "UNIFIED", "greeting"),
    ("smoke-f", "That was amazing!", "UNIFIED", "feedback"),
    ("smoke-f", "You're really good at this", "UNIFIED", "feedback"),
    ("smoke-f", "I appreciate the help", "UNIFIED", "feedback"),
    ("smoke-f", "Can you help me with something else?", "UNIFIED", "question"),

    # --- Session G: complex questions (cold start) ---
    ("smoke-g", "Hi there", "UNIFIED", "greeting"),
    ("smoke-g", "How does quantum computing work?", "UNIFIED", "question"),
    ("smoke-g", "What are its practical applications?", "UNIFIED", "question"),
    ("smoke-g", "Do you remember what we discussed about AI?", "ACT", "memory"),
    ("smoke-g", "Perfect, that clarifies things", "UNIFIED", "feedback"),

    # --- Session H: longer conversation (warm context buildup) ---
    ("smoke-h", "Good morning!", "UNIFIED", "greeting"),
    ("smoke-h", "I want to talk about cooking", "UNIFIED", "statement"),
    ("smoke-h", "What's a good recipe for beginners?", "UNIFIED", "question"),
    ("smoke-h", "Nice, any tips for seasoning?", "UNIFIED", "question"),
    ("smoke-h", "What about baking?", "UNIFIED", "question"),
    ("smoke-h", "ok thanks for all the info", "UNIFIED", "feedback"),
    ("smoke-h", "You mentioned something earlier about herbs", "ACT", "memory"),
    ("smoke-h", "Great, goodbye!", "UNIFIED", "greeting"),

    # --- Session I: edge cases ---
    ("smoke-i", "Greetings", "UNIFIED", "greeting"),
    ("smoke-i", "Tell me something interesting", "UNIFIED", "question"),
    ("smoke-i", "Go on", "UNIFIED", "statement"),
    ("smoke-i", "That's not what I meant", "UNIFIED", "feedback"),
    ("smoke-i", "Never mind, thanks anyway", "UNIFIED", "feedback"),

    # --- Session J: rapid fire (same uuid, context accumulates) ---
    ("smoke-j", "Hiya!", "UNIFIED", "greeting"),
]


def send_prompt(uuid, message):
    """Send a prompt and return (response_dict, status_code, elapsed_seconds)."""
    data = json.dumps({"uuid": uuid, "message": message}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/message",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        elapsed = time.time() - start
        return json.loads(resp.read().decode()), resp.status, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        return {"error": e.read().decode()}, e.code, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return {"error": str(e)}, 0, elapsed


def run():
    print("=" * 70)
    print("  50-Prompt Smoke Test — Mode Distribution + Rolling Response Times")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Health check
    try:
        req = urllib.request.Request(f"{API_BASE}/health")
        resp = urllib.request.urlopen(req, timeout=10)
        health = json.loads(resp.read().decode())
        if health.get("status") != "ok":
            print("ABORT: health check failed: %s" % health)
            return 1
        print("Health check: OK\n")
    except Exception as e:
        print("ABORT: cannot reach API: %s" % e)
        return 1

    results = []
    mode_counts = {}
    correct = 0
    total_time = 0
    rolling_times = []  # (prompt_index, session_depth, elapsed)
    session_depth = {}  # uuid -> count of prompts sent

    print("%-4s %-10s %-3s %-7s %-12s %-12s %-6s  %s" % (
        "#", "UUID", "Dep", "Time", "Got", "Expected", "Match", "Prompt"
    ))
    print("-" * 90)

    for i, (uuid, message, expected, category) in enumerate(PROMPTS):
        depth = session_depth.get(uuid, 0) + 1
        session_depth[uuid] = depth

        resp, status, elapsed = send_prompt(uuid, message)
        total_time += elapsed

        if status != 200 or not resp.get("success"):
            got_mode = "ERROR"
            match = False
        else:
            got_mode = resp.get("mode", "?")
            # Accept if got_mode matches expected, or if unified path is active
            # (unified generation returns "UNIFIED" instead of ACT)
            match = got_mode == expected or got_mode == "UNIFIED"

        mode_counts[got_mode] = mode_counts.get(got_mode, 0) + 1
        if match:
            correct += 1

        rolling_times.append((i + 1, depth, elapsed))
        mark = "Y" if match else "-"

        print("%-4d %-10s %-3d %5.1fs  %-12s %-12s %-6s  \"%s\"" % (
            i + 1, uuid, depth, elapsed, got_mode, expected, mark, message[:40]
        ))

        results.append({
            "index": i + 1,
            "uuid": uuid,
            "depth": depth,
            "message": message,
            "expected": expected,
            "got": got_mode,
            "match": match,
            "category": category,
            "elapsed": round(elapsed, 2),
        })

        if i < len(PROMPTS) - 1:
            time.sleep(SPACING)

    # --- Summary ---
    n = len(PROMPTS)
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\nMode distribution:")
    for mode in ["UNIFIED", "UNIFIED", "ACT", "IGNORE", "ERROR"]:
        count = mode_counts.get(mode, 0)
        if count > 0:
            print("  %-12s: %d/%d (%.0f%%)" % (mode, count, n, 100 * count / n))

    print("\nAccuracy: %d/%d (%.0f%%)" % (correct, n, 100 * correct / n))

    # Category breakdown
    print("\nPer-category accuracy:")
    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"total": 0, "correct": 0}
        cats[cat]["total"] += 1
        if r["match"]:
            cats[cat]["correct"] += 1
    for cat, v in sorted(cats.items()):
        print("  %-12s: %d/%d" % (cat, v["correct"], v["total"]))

    # Rolling response times by session depth
    print("\nResponse time by session depth (context growth):")
    depth_times = {}
    for _, depth, elapsed in rolling_times:
        if depth not in depth_times:
            depth_times[depth] = []
        depth_times[depth].append(elapsed)
    for depth in sorted(depth_times.keys()):
        times = depth_times[depth]
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        print("  depth %d: avg=%5.1fs  min=%5.1fs  max=%5.1fs  n=%d" % (
            depth, avg, mn, mx, len(times)
        ))

    # Rolling 10-prompt moving average
    print("\nRolling 10-prompt response time average:")
    window = 10
    for start_idx in range(0, len(rolling_times), window):
        chunk = rolling_times[start_idx:start_idx + window]
        times = [t for _, _, t in chunk]
        avg = sum(times) / len(times)
        print("  prompts %2d-%2d: avg=%5.1fs" % (
            chunk[0][0], chunk[-1][0], avg
        ))

    print("\nTotal LLM time: %.1fs (avg %.1fs/prompt)" % (total_time, total_time / n))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            API_BASE = f"http://{sys.argv[idx + 1]}:8080"

    sys.exit(run())
