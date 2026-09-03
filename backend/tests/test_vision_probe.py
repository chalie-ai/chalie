"""Unit tests for the vision probe: answer normalisation, the pass rule and the
fail-closed paths. No network — the provider send is replaced."""
from unittest.mock import patch

import pytest

from services import vision_probe
from services.vision_probe import PASS_MINIMUM, PROBE_QUESTIONS, normalise_answer, score_answers

pytestmark = pytest.mark.unit

_SEND = "services.vision_service.send_image_with_config"
_PROVIDER = {"platform": "ollama", "model": "llava", "name": "probe-under-test"}

#: An accepted answer to every question, in question order.
_ALL_CORRECT = ["No", "Yes", "4", "Count the blue as red", "3"]


def test_formatting_around_a_bare_answer_is_ignored() -> None:
    """Quotes, a full stop, markdown emphasis, casing and stray whitespace are
    how models decorate a one-word answer; none of them says anything about
    whether the image was seen, so all of them normalise away."""
    assert normalise_answer(' "No." ') == "no"
    assert normalise_answer("**Four**\n") == "four"
    assert normalise_answer('"Count the blue as red."') == "count the blue as red"
    assert score_answers([' "No." ', "YES", "4.", '"Count the blue as red"', "**3**"]) == [True] * 5


def test_prose_around_an_answer_is_wrong() -> None:
    """Every prompt demands a bare answer, and the comparison is whole-answer
    equality on purpose: a substring search would score a sentence that merely
    contains the right word, which is not the instruction-following the probe
    is asking for. Relaxing this is a decision, not a fix."""
    replies = ["No, yellow is not present.", *_ALL_CORRECT[1:]]
    assert score_answers(replies) == [False, True, True, True, True]


def test_missing_replies_are_wrong_answers_not_errors() -> None:
    assert score_answers([None, "", "4", "Count the blue as red", "3"]) == [False, False, True, True, True]


def test_reply_count_must_match_question_count() -> None:
    with pytest.raises(ValueError):
        score_answers(["No"])


def test_minimum_correct_answers_pass_and_every_question_is_asked_with_the_image() -> None:
    """The pass rule is PASS_MINIMUM of five, and each question goes out as its
    own request carrying the probe image, so no answer can lean on an earlier
    one and a text-only reply cannot answer for the picture."""
    replies = ["No", "Yes", "4", "nothing to read", "7"]
    assert sum(score_answers(replies)) == PASS_MINIMUM
    with patch(_SEND, side_effect=replies) as send:
        assert vision_probe.probe_provider(_PROVIDER) is True
    assert [call.args[2] for call in send.call_args_list] == [prompt for prompt, _ in PROBE_QUESTIONS]
    images = {call.args[1] for call in send.call_args_list}
    assert len(images) == 1
    assert next(iter(images))[:8] == b"\x89PNG\r\n\x1a\n"


def test_one_short_of_the_minimum_fails() -> None:
    replies = ["No", "Yes", "5", "nothing to read", "7"]
    assert sum(score_answers(replies)) == PASS_MINIMUM - 1
    with patch(_SEND, side_effect=replies):
        assert vision_probe.probe_provider(_PROVIDER) is False


def test_send_failure_is_a_fail_not_an_exception() -> None:
    """The send helper returns None on any transport failure, so a text-only
    model that rejects the image scores 0/5 and the row is written as
    no-vision instead of the save blowing up."""
    with patch(_SEND, return_value=None):
        assert vision_probe.probe_provider(_PROVIDER) is False


def test_exception_inside_the_probe_is_a_fail() -> None:
    with patch(_SEND, side_effect=RuntimeError("boom")):
        assert vision_probe.probe_provider(_PROVIDER) is False
