"""Unit tests for vision_probe JSON extraction from messy model replies.

Scope note: these cover ONLY the parse-robustness of _extract_json — the one part
of the probe with non-trivial, regression-prone logic (see the S5857 fenced-JSON
regex fix). The probe's actual job — verifying a real model SEES vision-test.png —
is validated in real use, not here: a model-free assertion against the scorer just
restates the scoring constants back to themselves (tautological) and cannot catch
the failure modes that matter (answer-key drift, a real model silently ignoring the
image). See the TKT-715 decision log.
"""
import json

import pytest

pytestmark = pytest.mark.unit

_PERFECT = json.dumps({
    "number_of_shapes": 3,
    "shapes": [
        {"shape": "rectangle", "color": "red"},
        {"shape": "circle", "color": "yellow"},
        {"shape": "hexagon", "color": "green"},
    ],
    "text": "Chalie can read!",
})


def test_json_in_code_fence_is_parsed():
    # Guards the S5857 fenced-JSON regex: the payload has nested braces (one object
    # per shape), so the extractor must capture the WHOLE object, not stop at the
    # first '}'. A reluctant or [^}]* regex would truncate it and score 0.0.
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"Here you go:\n```json\n{_PERFECT}\n```") == pytest.approx(1.0)


def test_prose_then_json_is_parsed():
    # Models often narrate before the JSON; find('{')..rfind('}') must still recover it.
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"I can see three shapes. {_PERFECT}") == pytest.approx(1.0)


def test_garbage_scores_zero():
    from services.vision_probe import score_probe_response
    assert score_probe_response("no json here") == pytest.approx(0.0)
    assert score_probe_response("") == pytest.approx(0.0)
