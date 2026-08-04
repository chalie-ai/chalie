"""Unit tests for vision_probe scoring + probe_provider (no network)."""
import json
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# Old-style exact reply (still scores 1.0).
_PREFERRED = json.dumps({
    "number_of_shapes": 3,
    "shapes": [
        {"shape": "rectangle", "color": "red"},
        {"shape": "circle", "color": "yellow"},
        {"shape": "hexagon", "color": "green"},
    ],
    "text": "Chalie can read!",
})

# Reply mirroring the real-world folded-color failure that triggered the rewrite.
_FOLDED_COLOR = json.dumps({
    "number_of_shapes": 3,
    "shapes": [
        {"shape": "red rod", "color": "red"},
        {"shape": "yellow circle", "color": "yellow"},
        {"shape": "green hexagon", "color": "green"},
    ],
    "text": "Chalie can read!",
})


def test_perfect_exact_form_scores_one() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(_PREFERRED) == pytest.approx(1.0)


def test_folded_color_failure_scores_one_and_passes() -> None:
    """Replicate the reproduced failure: model folds colour into the shape.
    Previously this got 0.55; with synonym+token matching it gets 1.0 and the
    probe passes."""
    from services.vision_probe import score_probe_response
    assert score_probe_response(_FOLDED_COLOR) == pytest.approx(1.0)


def test_count_plus_two_shapes_plus_text_passes() -> None:
    # One shape missed entirely: 0.30 count + 2*0.15 attributes + 0.25 text = 0.85.
    # A model that saw the image but under-reported one shape still passes.
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
    """Every shape right but the text unread is not a passing vision provider.

    0.30 + 0.45 + 0.00 = 0.75, under the threshold. Asserted against
    PASS_THRESHOLD and not just the number: reading the rendered text is the
    one thing this probe tests that a blind guess from the prompt cannot fake,
    so no future reweighting may quietly promote this reply to a pass.
    """
    from services.vision_probe import PASS_THRESHOLD, score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "circle", "color": "yellow"},
            {"shape": "hexagon", "color": "green"},
        ],
        "text": "",
    })
    score = score_probe_response(body)
    assert score == pytest.approx(0.75)
    assert score < PASS_THRESHOLD


def test_three_shapes_correct_text_wrong_count_fails() -> None:
    """Miscounting the shapes fails even with every other slot earned.

    0.00 + 0.45 + 0.25 = 0.70. Also pinned against PASS_THRESHOLD: a model that
    reports five shapes when three are drawn did not see the image accurately,
    however well it described the three it did name.
    """
    from services.vision_probe import PASS_THRESHOLD, score_probe_response
    body = json.dumps({
        "number_of_shapes": 5,
        "shapes": [
            {"shape": "rectangle", "color": "red"},
            {"shape": "circle", "color": "yellow"},
            {"shape": "hexagon", "color": "green"},
        ],
        "text": "Chalie can read!",
    })
    score = score_probe_response(body)
    assert score == pytest.approx(0.70)
    assert score < PASS_THRESHOLD


def test_case_insensitive_and_trimmed_text() -> None:
    # Synonym/normaliser handles casing: "Rectangle"/"RED" + leading/trailing whitespace.
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


def test_quoted_uppercase_text_norm_equal() -> None:
    # Capitalised shape/value keywords + surrounding quotes around the text.
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "Rectangle", "color": "RED"},
            {"shape": "Circle", "color": "YELLOW"},
            {"shape": "Hexagon", "color": "GREEN"},
        ],
        "text": '"Chalie can read!"',
    })
    assert score_probe_response(body) == pytest.approx(1.0)


def test_json_in_code_fence_is_parsed() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"Here you go:\n```json\n{_PREFERRED}\n```") == pytest.approx(1.0)


def test_prose_then_json_is_parsed() -> None:
    from services.vision_probe import score_probe_response
    assert score_probe_response(f"I can see three shapes. {_PREFERRED}") == pytest.approx(1.0)


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
    # Greedy consumption: the first (rect,red) is claimed by the rectangle and
    # scores both halves; the second is left unmatched because circle/yellow and
    # hexagon/green find nothing in it. 0.30 count + 0.15 attr + 0.25 text = 0.70.
    assert score_probe_response(body) == pytest.approx(0.70)


def test_zero_shape_colour_matches_correct_text_scores_count_and_text() -> None:
    # Three 'blue triangles': neither word appears in any synonym set, so no
    # attribute scores at all -> 0.30 count + 0.00 attr + 0.25 text = 0.55.
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "blue triangle", "color": "blue"},
            {"shape": "blue triangle", "color": "blue"},
            {"shape": "blue triangle", "color": "blue"},
        ],
        "text": "Chalie can read!",
    })
    assert score_probe_response(body) == pytest.approx(0.55)


def test_three_blue_triangles_wrong_text_fails() -> None:
    # Wrong text zeros the text slot too, leaving just the count slot.
    from services.vision_probe import score_probe_response
    body = json.dumps({
        "number_of_shapes": 3,
        "shapes": [
            {"shape": "blue triangle", "color": "blue"},
            {"shape": "blue triangle", "color": "blue"},
            {"shape": "blue triangle", "color": "blue"},
        ],
        "text": "nothing visible here",
    })
    assert score_probe_response(body) == pytest.approx(0.30)


def test_text_scores_regardless_of_punctuation_and_spacing() -> None:
    """The text slot compares words, not formatting.

    Both replies transcribe the image correctly; one adds a comma (which the
    normaliser turns into a second space) and one doubles a space. Scoring these
    below a clean transcription would fail a provider for its punctuation, so
    the text field runs through the same normaliser as the shape tokens.
    """
    from services.vision_probe import score_probe_response

    def scored(text: str) -> float:
        return score_probe_response(json.dumps({
            "number_of_shapes": 3,
            "shapes": [
                {"shape": "rectangle", "color": "red"},
                {"shape": "circle", "color": "yellow"},
                {"shape": "hexagon", "color": "green"},
            ],
            "text": text,
        }))

    assert scored("Chalie, can read!") == pytest.approx(1.0)
    assert scored("Chalie  can   read") == pytest.approx(1.0)
    assert scored("\n\tChalie can read!\n") == pytest.approx(1.0)


def test_probe_provider_true_on_passing_reply() -> None:
    from services import vision_probe
    with patch("services.vision_service.send_image_with_config", return_value=_PREFERRED):
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
