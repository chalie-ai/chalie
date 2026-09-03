"""Vision probe — five questions about one image decide whether a provider's
model can actually see.

Runs synchronously when a provider is saved (on create, and on update whenever
the platform, model, host or api key change); its verdict becomes
``supports_vision``. The image (``backend/vision/vision-test.png``) shows two
red circles, one green circle and one blue square above the black caption
"Count the blue as red". Each question is sent as its own image+text request,
so no answer can lean on an earlier one, and each demands a bare answer — one
word, one digit, or the caption verbatim — so scoring is whole-answer equality
after normalisation, never a search through prose.

The five questions probe distinct abilities, and a blind model that guesses
cannot pass them together: a colour that is absent (a reflexive "yes" fails),
a colour that is present, counting the shapes, reading the caption, and
reading it well enough to apply it to the shapes. A provider passes with at
least ``PASS_MINIMUM`` correct answers; fewer means the model is not seeing the
picture, however fluently it talks about it.

Fail-closed: an empty reply, or one that is not an accepted answer, is wrong;
any exception inside the probe is a fail. The caller writes the verdict
straight into the provider row.
"""

import logging
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from services.log_utils import safe

logger = logging.getLogger(__name__)

#: Correct answers, out of five, a provider needs to be marked vision-capable.
PASS_MINIMUM = 3


def normalise_answer(reply: str) -> str:
    """The comparison form of an answer: lower case, every non-alphanumeric
    character turned into a space, whitespace collapsed and stripped.

    ``No.``, ``"no"`` and ``**No**`` all become ``no``; the caption wrapped in
    quotes or ended with a full stop still equals the caption. Words are never
    dropped or reordered, so prose around an answer still fails it.
    """
    return ' '.join(''.join(ch if ch.isalnum() else ' ' for ch in reply.lower()).split())


def _accepted(*answers: str) -> FrozenSet[str]:
    """Accepted answers in their comparison form, so the table below reads as
    plain words and can never drift from the normaliser."""
    return frozenset(normalise_answer(answer) for answer in answers)


_BARE_ANSWER = 'No reasoning, thinking or prose allowed.'

#: ``(prompt sent with the image, accepted answers)`` — asked in this order.
PROBE_QUESTIONS: Tuple[Tuple[str, FrozenSet[str]], ...] = (
    (
        f'Is the colour yellow present in this image? Respond with ONLY Yes or No. {_BARE_ANSWER}',
        _accepted('No', 'false', 'not present'),
    ),
    (
        f'Is the colour black present in this image? Respond with ONLY Yes or No. {_BARE_ANSWER}',
        _accepted('Yes', 'true', 'correct', 'present'),
    ),
    (
        f'How many shapes are in this image? Respond with ONLY 1 numeric figure between 1 & 10. {_BARE_ANSWER}',
        _accepted('4', 'four'),
    ),
    (
        'What text is present in this image? Respond with ONLY the text on screen VERBATIM, '
        'no prose or reasoning allowed.',
        _accepted('Count the blue as red'),
    ),
    (
        'If you followed the instruction shown in the image, how many red shapes would there be? '
        f'Respond with ONLY 1 numeric figure between 1 & 10. {_BARE_ANSWER}',
        _accepted('3', 'three'),
    ),
)


def score_answers(replies: Sequence[Optional[str]]) -> List[bool]:
    """One verdict per question, in question order: whether the reply's
    comparison form is an accepted answer. A missing or empty reply is wrong."""
    if len(replies) != len(PROBE_QUESTIONS):
        raise ValueError(f'expected {len(PROBE_QUESTIONS)} replies, got {len(replies)}')
    return [
        normalise_answer(reply or '') in accepted
        for reply, (_, accepted) in zip(replies, PROBE_QUESTIONS)
    ]


def probe_provider(provider: Dict[str, object]) -> bool:
    """Ask the provider every probe question about the probe image; True when
    at least ``PASS_MINIMUM`` answers are correct.

    Every answer is logged so an operator can see exactly which question a
    provider failed. Any exception is a fail, never a raise: the caller is a
    provider save, and an unprobeable provider is a provider without vision.
    """
    try:
        from services.file_mapper_service import FileMapperService
        from services import vision_service

        asset_path = FileMapperService.get_backend_path('vision', 'vision-test.png')
        with open(asset_path, 'rb') as fh:
            image_bytes = fh.read()
        config = vision_service.build_vision_config(provider)
        replies = [
            vision_service.send_image_with_config(config, image_bytes, prompt, mime_type='image/png')
            for prompt, _ in PROBE_QUESTIONS
        ]
        verdicts = score_answers(replies)
        name = safe(provider.get('name'))
        for number, (reply, correct) in enumerate(zip(replies, verdicts), start=1):
            logger.info(
                "[VisionProbe] name=%s q%d correct=%s answer=%r",
                name, number, correct, safe((reply or '')[:80]),
            )
        correct_count = sum(verdicts)
        passed = correct_count >= PASS_MINIMUM
        logger.info(
            "[VisionProbe] name=%s platform=%s model=%s correct=%d/%d pass=%s",
            name, safe(provider.get('platform')), safe(provider.get('model')),
            correct_count, len(PROBE_QUESTIONS), passed,
        )
        return passed
    except Exception as exc:
        logger.warning("[VisionProbe] probe failed: %s", exc)
        return False
