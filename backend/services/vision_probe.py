"""Vision probe — five questions about one image decide whether a provider's
model can actually see.

Runs synchronously when a provider is saved (on create, and on update whenever
the platform, model, host or api key change); its verdict becomes
``supports_vision``. The image (``backend/vision/vision-test.png``) shows two
red circles, one green circle and one blue square above the black caption
"Count the blue as red". Each question is sent as its own image+text request,
so no answer can lean on an earlier one; the five go out concurrently, so the
probe costs one round-trip instead of five. Each demands a bare answer — one
word, one digit, or the caption verbatim — so scoring is whole-answer equality
after normalisation, never a search through prose.

The five questions probe distinct abilities, and a blind model that guesses
cannot pass them together: a colour that is absent (a reflexive "yes" fails),
a colour that is present, counting the shapes, reading the caption, and
reading it well enough to apply it to the shapes. A provider passes with at
least ``PASS_MINIMUM`` correct answers; fewer means the model is not seeing the
picture, however fluently it talks about it.

Fail-closed: an empty reply, or one that is not an accepted answer, is wrong;
any exception inside the probe is a fail. Each answer is logged the moment it
arrives, so an interrupted probe still leaves evidence of what the model
answered. The caller writes the verdict straight into the provider row.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from services.log_utils import safe

logger = logging.getLogger(__name__)

#: Correct answers, out of five, a provider needs to be marked vision-capable.
PASS_MINIMUM = 3


def normalise_answer(reply: str) -> str:
    """The comparison form of an answer: lower case, integer-valued decimals
    folded to their integer, every non-alphanumeric character turned into a
    space, whitespace collapsed and stripped.

    ``No.``, ``"no"`` and ``**No**`` all become ``no``; the caption wrapped in
    quotes or ended with a full stop still equals the caption. ``4.0`` and
    ``10.00`` are numeric figures, so they fold to ``4`` and ``10``; a genuine
    fraction like ``4.5`` is a different answer and is left as is, so it still
    fails a whole-answer comparison. Words are never dropped or reordered, so
    prose around an answer still fails it.
    """
    text = re.sub(r'\b(\d+)\.0+\b', r'\1', reply.lower())
    return ' '.join(''.join(ch if ch.isalnum() else ' ' for ch in text).split())


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


def _is_correct(index: int, reply: Optional[str]) -> bool:
    """Whether one reply passes its question: the reply's comparison form is in
    the accepted set of the question at ``index``. A missing or empty reply is
    a wrong answer, not an error — the probe scores what came back, however
    thin, and only the transport layer decides what is a failure."""
    _, accepted = PROBE_QUESTIONS[index]
    return normalise_answer(reply or '') in accepted


def score_answers(replies: Sequence[Optional[str]]) -> List[bool]:
    """One verdict per question, in question order: whether the reply's
    comparison form is an accepted answer. A missing or empty reply is wrong."""
    if len(replies) != len(PROBE_QUESTIONS):
        raise ValueError(f'expected {len(PROBE_QUESTIONS)} replies, got {len(replies)}')
    return [_is_correct(index, reply) for index, reply in enumerate(replies)]


def probe_provider(provider: Dict[str, object]) -> bool:
    """Ask the provider every probe question about the probe image; True when
    at least ``PASS_MINIMUM`` answers are correct.

    The five questions go out on a function-scoped pool, so the probe costs
    one round-trip instead of five, and every answer is logged the moment it
    arrives: an operator sees exactly which question a provider failed, and an
    interrupted probe still leaves evidence of what the model answered. Any
    exception is a fail, never a raise: the caller is a provider save, and an
    unprobeable provider is a provider without vision. A send that raises is
    logged against its question and the remaining answers are still collected
    and logged before the fail, so one bad question never hides the others.
    """
    try:
        from services.file_mapper_service import FileMapperService
        from services import vision_service

        asset_path = FileMapperService.get_backend_path('vision', 'vision-test.png')
        with open(asset_path, 'rb') as fh:
            image_bytes = fh.read()
        config = vision_service.build_vision_config(provider)
        name = safe(provider.get('name'))
        replies: List[Optional[str]] = [None] * len(PROBE_QUESTIONS)
        raised = False
        with ThreadPoolExecutor(max_workers=len(PROBE_QUESTIONS)) as pool:
            futures = {
                pool.submit(
                    vision_service.send_image_with_config,
                    config, image_bytes, prompt, mime_type='image/png',
                ): index
                for index, (prompt, _) in enumerate(PROBE_QUESTIONS)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    reply = future.result()
                except Exception as exc:
                    raised = True
                    logger.warning("[VisionProbe] name=%s q%d raised: %s", name, index + 1, safe(str(exc)))
                    continue
                replies[index] = reply
                logger.info(
                    "[VisionProbe] name=%s q%d correct=%s answer=%r",
                    name, index + 1, _is_correct(index, reply), safe((reply or '')[:80]),
                )
        if raised:
            logger.warning("[VisionProbe] name=%s probe failed: a question raised", name)
            return False
        verdicts = score_answers(replies)
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
