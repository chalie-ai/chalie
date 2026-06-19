"""PROBE: turn-scoped state accumulation across 2+ tool-bearing chain steps.

Drives the real recursive _continue() chain via SubconsciousWorker._step_pattern_match.
Step 1 (MP1) saves pattern_x. Step 2 (MP2) saves pattern_y. Step 3 (MP3) ends.
The PatternDecayHook fires on MP3. If turn-scoped state accumulated correctly across
the chain, BOTH freshly-saved patterns stay at confidence 7.0 (touched -> not decayed),
and the counters reflect the full total (2 save_pattern calls).
"""
import json

import pytest

from tests.test_pattern_match_processor import (
    _inject_fake_client,
    _make_llm_response,
    _seed_transcripts,
    _tool_call,
    _fetch_pattern,
)

pytestmark = pytest.mark.unit


def test_two_tool_bearing_steps_accumulate_touched_and_counters(db, store):
    _seed_transcripts(db, 60)

    tc_x = _tool_call(
        "save_pattern", name="pattern_x", frequency="weekday",
        summary="x habit", evidence_transcript_ids=[1, 2],
    )
    tc_y = _tool_call(
        "save_pattern", name="pattern_y", frequency="weekday",
        summary="y habit", evidence_transcript_ids=[3, 4],
    )

    call_count = {"n": 0}

    def _fake_send(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_llm_response(tool_calls=[tc_x])   # MP1
        if call_count["n"] == 2:
            return _make_llm_response(tool_calls=[tc_y])   # MP2
        return _make_llm_response(tool_calls=None)         # MP3 ends

    with _inject_fake_client(_fake_send):
        from services.subconscious_worker import SubconsciousWorker
        worker = SubconsciousWorker(tick_sec=10, idle_window_sec=60)
        worker._step_pattern_match()

    print(f"\n[PROBE] total send calls: {call_count['n']}")

    px = _fetch_pattern(db, "pattern_x")
    py = _fetch_pattern(db, "pattern_y")
    assert px is not None and py is not None, "both patterns must persist"

    # The CRITICAL assertion: a pattern saved in step 1 (MP1) must NOT be decayed
    # by the post-turn hook running on MP3. Fresh insert confidence is 7.0; decay
    # would knock the UNtouched one to 6.995.
    assert px["value"]["confidence"] == pytest.approx(7.0), (
        f"pattern_x (saved in step 1) was decayed -> touched set did NOT "
        f"accumulate across the chain. confidence={px['value']['confidence']}"
    )
    assert py["value"]["confidence"] == pytest.approx(7.0), (
        f"pattern_y (saved in step 2) confidence={py['value']['confidence']}"
    )
