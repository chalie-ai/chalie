"""Unit tests for vision_probe scoring + probe_provider (no network)."""
import json
from unittest.mock import patch

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


def test_perfect_scores_1() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(_PERFECT) == pytest.approx(1.0)


def test_count_plus_two_shapes_plus_text_passes() -> None:
    # 0.30 + 0.15 + 0.15 + 0.25 = 0.85  (≥ 0.80)
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "circle", "color": "yellow"},
        ],
        "text": "Chalie can read!",
    })
    assert score_probe_response(body) == pytest.approx(0.85)


def test_count_plus_three_shapes_no_text_fails() -> None:
    # 0.30 + 0.45 = 0.75  (< 0.80)
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "circle", "color": "yellow"},
            {"shape": "hexagon", "color": "green"},
        ],
        "text": "",
    })
    assert score_probe_response(body) == pytest.approx(0.75)


def test_three_shapes_text_wrong_count_fails() -> None:
    # 0.45 + 0.25 = 0.70  (< 0.80)
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 5,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "circle", "color": "yellow"},
            {"shape": "hexagon", "color": "green"},
        ],
        "text": "Chalie can read!",
    })
    assert score_probe_response(body) == pytest.approx(0.70)


def test_case_insensitive_and_trimmed_text() -> None:
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "Rectangle", "color": "RED"},
            {"shape": "circle", "color": "Yellow"},
            {"shape": "HEXAGON", "color": "green"},
        ],
        "text": "  chalie can read!  ",
    })
    assert score_probe_response(body) == pytest.approx(1.0)


def test_json_in_code_fence_is_parsed() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"Here you go:\n```json\n{_PERFECT}\n```") == pytest.approx(1.0)


def test_prose_then_json_is_parsed() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"I can see three shapes. {_PERFECT}") == pytest.approx(1.0)


def test_garbage_scores_zero() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response("no json here") == pytest.approx(0.0)
    assert score_probe_response("") == pytest.approx(0.0)


def test_duplicate_correct_shape_counts_once() -> None:
    # repeated (rectangle,red) must not double-count
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "rectangle", "color": "red"},
        ],
        "text": "Chalie can read!",
    })
    assert score_probe_response(body) == pytest.approx(0.70)  # 0.30 + 0.15 + 0.25


def test_probe_provider_true_on_passing_reply() -> None:
    from services import vision_probe
    with patch("services.vision_service.send_image_with_config", return_value=_PERFECT):
        assert vision_probe.probe_provider(
            {"platform": "ollama", "model": "llava"}) is True


def test_probe_provider_false_on_low_score() -> None:
    from services import vision_probe
    with patch("services.vision_service.send_image_with_config",
               return_value='{"number_of_shapes": 0, "shapes": [], "text": ""}'):
        assert vision_probe.probe_provider(
            {"platform": "ollama", "model": "llava"}) is False


def test_probe_provider_false_on_none_reply() -> None:
    from services import vision_probe
    with patch("services.vision_service.send_image_with_config", return_value=None):
        assert vision_probe.probe_provider(
            {"platform": "ollama", "model": "llava"}) is False
