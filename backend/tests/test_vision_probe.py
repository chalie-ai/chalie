"""Unit tests for the vision probe: answer normalisation, the pass rule and the
fail-closed paths. No network — the provider send is replaced."""
import logging
import threading
import time
from typing import Callable, Dict, List, Optional
from unittest.mock import patch

import pytest

from services import vision_probe
from services.vision_probe import PASS_MINIMUM, PROBE_QUESTIONS, normalise_answer, score_answers

pytestmark = pytest.mark.unit

_SEND = "services.vision_service.send_image_with_config"
_PROVIDER: Dict[str, object] = {"platform": "ollama", "model": "llava", "name": "probe-under-test"}

#: An accepted answer to every question, in question order.
_ALL_CORRECT: List[str] = ["No", "Yes", "4", "Count the blue as red", "3"]

#: The prompt sent with the probe image for each question, in question order.
_PROMPTS: List[str] = [prompt for prompt, _ in PROBE_QUESTIONS]


def _send_reply(replies: List[str]) -> Callable[[Dict[str, object], bytes, str], Optional[str]]:
    """A send stand-in that finds each reply by the prompt it was sent with.

    The probe now fires its questions concurrently, so a positional
    side_effect list could hand a reply to the wrong question — a verdict
    only means what the test intends if every reply reaches the question it
    was written for."""
    by_prompt = dict(zip(_PROMPTS, replies))

    def fake(config: Dict[str, object], image: bytes, prompt: str,
             **kwargs: object) -> Optional[str]:
        return by_prompt[prompt]

    return fake


def test_probe_prompts_are_distinct() -> None:
    """The concurrent fakes below route each reply by the prompt it was sent
    with, which only works while no two questions share a prompt."""
    assert len(set(_PROMPTS)) == len(PROBE_QUESTIONS)


def test_formatting_around_a_bare_answer_is_ignored() -> None:
    """Quotes, a full stop, markdown emphasis, casing and stray whitespace are
    how models decorate a one-word answer; none of them says anything about
    whether the image was seen, so all of them normalise away. Trailing zeros
    make a decimal a figure, not decoration: 4.0 is the number 4 and 10.00
    is 10, while a genuine fraction like 4.5 stays a different answer."""
    assert normalise_answer(' "No." ') == "no"
    assert normalise_answer("**Four**\n") == "four"
    assert normalise_answer('"Count the blue as red."') == "count the blue as red"
    assert normalise_answer("4.0") == "4"
    assert normalise_answer("10.00") == "10"
    assert normalise_answer("4.5") != "4"
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
    one and a text-only reply cannot answer for the picture. The prompts are
    compared as a set: the questions fire concurrently, so their order on the
    wire is the scheduler's, not the test's to assert."""
    replies = ["No", "Yes", "4", "nothing to read", "7"]
    assert sum(score_answers(replies)) == PASS_MINIMUM
    with patch(_SEND, side_effect=_send_reply(replies)) as send:
        assert vision_probe.probe_provider(_PROVIDER) is True
    assert {call.args[2] for call in send.call_args_list} == set(_PROMPTS)
    images = {call.args[1] for call in send.call_args_list}
    assert len(images) == 1
    assert next(iter(images))[:8] == b"\x89PNG\r\n\x1a\n"


def test_one_short_of_the_minimum_fails() -> None:
    replies = ["No", "Yes", "5", "nothing to read", "7"]
    assert sum(score_answers(replies)) == PASS_MINIMUM - 1
    with patch(_SEND, side_effect=_send_reply(replies)):
        assert vision_probe.probe_provider(_PROVIDER) is False


def test_questions_are_asked_concurrently() -> None:
    """Every send must reach the barrier before any may return: a sequential
    probe would time it out and fail, so a pass with one send per question
    proves all five were in flight together."""
    barrier = threading.Barrier(len(PROBE_QUESTIONS), timeout=30)
    by_prompt = dict(zip(_PROMPTS, _ALL_CORRECT))

    def fake(config: Dict[str, object], image: bytes, prompt: str,
             **kwargs: object) -> Optional[str]:
        barrier.wait()
        return by_prompt[prompt]

    with patch(_SEND, side_effect=fake) as send:
        assert vision_probe.probe_provider(_PROVIDER) is True
    assert send.call_count == len(PROBE_QUESTIONS)


def test_each_answer_is_logged_as_it_arrives(caplog: pytest.LogCaptureFixture) -> None:
    """The per-answer line appears the moment its own reply arrives, not once
    the slowest reply is in: an interrupted probe must leave evidence of what
    the model answered up to the interruption, and the held reply here is the
    one that never arrives on time."""
    caplog.set_level(logging.INFO, logger="services.vision_probe")
    by_prompt = dict(zip(_PROMPTS, _ALL_CORRECT))
    held_prompt = _PROMPTS[0]
    gate = threading.Event()

    def fake(config: Dict[str, object], image: bytes, prompt: str,
             **kwargs: object) -> Optional[str]:
        if prompt == held_prompt:
            gate.wait(timeout=10)
        return by_prompt[prompt]

    def run_probe() -> None:
        vision_probe.probe_provider(_PROVIDER)

    worker = threading.Thread(target=run_probe)
    with patch(_SEND, side_effect=fake):
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                messages = [record.message for record in caplog.records]
                if all(
                    any(f"q{number} correct=" in message for message in messages)
                    for number in (2, 3, 4, 5)
                ):
                    break
                time.sleep(0.01)
            messages = [record.message for record in caplog.records]
            assert all(
                any(f"q{number} correct=" in message for message in messages)
                for number in (2, 3, 4, 5)
            ), "the four unheld answers must be logged before the held reply is released"
        finally:
            gate.set()
            worker.join(timeout=10)
        assert not worker.is_alive()
        messages = [record.message for record in caplog.records]
        assert any("correct=5/5 pass=True" in message for message in messages)


def test_send_failure_is_a_fail_not_an_exception() -> None:
    """The send helper returns None on any transport failure, so a text-only
    model that rejects the image scores 0/5 and the row is written as
    no-vision instead of the save blowing up."""
    with patch(_SEND, return_value=None):
        assert vision_probe.probe_provider(_PROVIDER) is False


def test_exception_inside_the_probe_is_a_fail() -> None:
    with patch(_SEND, side_effect=RuntimeError("boom")):
        assert vision_probe.probe_provider(_PROVIDER) is False


def test_one_raising_send_still_logs_the_other_answers(caplog: pytest.LogCaptureFixture) -> None:
    """A send that raises is logged against its own question and fails the
    probe closed, but the other four answers are still collected and logged:
    the raise arrives first here, and the operator must still see what the
    model answered to everything else."""
    caplog.set_level(logging.INFO, logger="services.vision_probe")
    by_prompt = dict(zip(_PROMPTS, _ALL_CORRECT))
    raising_prompt = _PROMPTS[0]

    def fake(config: Dict[str, object], image: bytes, prompt: str,
             **kwargs: object) -> Optional[str]:
        if prompt == raising_prompt:
            raise RuntimeError("boom")
        time.sleep(0.05)
        return by_prompt[prompt]

    with patch(_SEND, side_effect=fake):
        assert vision_probe.probe_provider(_PROVIDER) is False
    messages = [record.message for record in caplog.records]
    assert any("q1 raised: boom" in message for message in messages)
    for number in (2, 3, 4, 5):
        assert any(f"q{number} correct=True" in message for message in messages)
    assert not any("pass=" in message for message in messages)
