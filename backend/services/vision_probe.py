"""Vision probe — verify a provider actually understands images.

Sends a known test image (3 shapes + text) with an exact-JSON prompt, scores the
structured reply against the answer key, and returns True iff score >= 0.80.

Scoring scheme (weights sum to 1.0):
    * 0.30  count — exactly 3 shapes reported
    * 0.45  six attribute checks (0.075 each): for each of the three expected
            (shape, colour) tuples we greedily pick one not-yet-consumed reported
            entry whose combined token bag contains the expected colour, then
            award the colour half and — if that same entry's bag also holds the
            expected shape word — the shape half. Each reported entry answers
            for at most one expected shape. If no entry matches the colour we
            fall back to matching on shape alone and award only that half.
    * 0.25  text  — equality of the normalised forms

The three slot totals are the probe's original balance and are deliberately
unchanged: the pass threshold sits at 0.80, so a reply that misses the count or
the text still fails on 0.70 / 0.75. What *did* change is how each slot is
earned — matching is case-insensitive and synonym-aware (see _COLOUR_SYNONYMS /
_SHAPE_SYNONYMS), colour and shape score independently, and both fields of an
entry contribute to one token bag. A model folding colour into the shape name
({"shape": "red rod"}) therefore scores the rectangle in full, which is a
correct observation phrased differently — not a vision failure. Leniency belongs
in what counts as the right answer, never in how much of the answer is required.

Normaliser for every comparison (shape tokens, colour tokens, and the text
field alike): lowercase, non-alphanumeric characters become spaces, whitespace
collapsed and stripped. One helper does it — :func:`_normalise_tokens` — so the
text field cannot drift from the tokens.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.80

# Sent verbatim with backend/vision/vision-test.png attached.
PROBE_PROMPT = """Analyse the image attached and return back this EXACT json with filled in details based on the image you see;

{
"number_of_shapes": <<enter_a_whole_number_here>>,
"shapes": [
    <<specify the shape name and color you see, 1 object per shape as per the example below>>
    {"shape": <<shape_name>>, "color": <<color>>}
],
"text": <<is there text in the image and if so what does it read? Paste it EXACTLY without prose>>
}"""

# Answer key for backend/vision/vision-test.png.
_EXPECTED_COUNT = 3
_EXPECTED_SHAPES = [
    ('rectangle', 'red'),
    ('circle', 'yellow'),
    ('hexagon', 'green'),
]

# Slot weights, summing to 1.0 across the count, six attribute halves, and the
# text. Named rather than inlined because PASS_THRESHOLD is only meaningful
# relative to them: at 0.80 a reply must earn the count, the text, and nearly
# every attribute — miss the count and the ceiling is 0.70, miss the text 0.75.
_COUNT_WEIGHT = 0.30
_ATTRIBUTE_HALF_WEIGHT = 0.075  # × 3 shapes × 2 halves (colour, shape) = 0.45
_TEXT_WEIGHT = 0.25
# Colour and shape synonym sets used by the tokenizer/matcher in
# score_probe_response. Each set maps the canonical key to all accepted
# surface forms. The normaliser strips punctuation and splits on whitespace, so
# compound model outputs like "red rod" yield token bags containing both
# "red" and "rod".
_COLOUR_SYNONYMS: Dict[str, Set[str]] = {
    'red': {'red', 'crimson', 'scarlet', 'maroon', 'vermilion'},
    'yellow': {'yellow', 'gold', 'golden', 'amber', 'mustard'},
    'green': {'green', 'lime', 'emerald', 'olive'},
}
_SHAPE_SYNONYMS: Dict[str, Set[str]] = {
    'rectangle': {
        'rectangle', 'rect', 'rectangular', 'square', 'bar', 'rod',
        'box', 'block', 'oblong', 'quadrilateral', 'stick', 'strip',
    },
    'circle': {
        'circle', 'circular', 'dot', 'disc', 'disk', 'round',
        'ellipse', 'oval', 'sphere', 'ball',
    },
    'hexagon': {'hexagon', 'hex', 'hexagonal', 'polygon'},
}

#: Lookup miss for either table above — a colour or shape the probe does not ask
#: about, which therefore matches nothing.
_NO_SYNONYMS: Set[str] = set()

# The words the image spells out. Written here in normalised form — lowercase
# and unpunctuated — because the comparison runs on normalised tokens; the image
# itself renders "Chalie can read!" and the exclamation mark is not evidence of
# anything the probe is testing.
_EXPECTED_TEXT = 'chalie can read'


def _normalise_tokens(value: str) -> List[str]:
    """Split on whitespace after lowering case and replacing non-alphanumerics
    with spaces — used for both shape/colour tokens and the text field."""
    return ''.join(ch if ch.isalnum() else ' ' for ch in value.lower()).split()


#: The text answer key as the normaliser will render it, so the comparison never
#: depends on this constant having been written punctuation-free by hand.
_EXPECTED_TEXT_TOKENS = _normalise_tokens(_EXPECTED_TEXT)


class _TokenBag:
    """Frozen set of normalised tokens derived from two string fields, treated
    as one combined bag for matching purposes. Useful when models fold colour
    into the shape name (e.g. ``{"shape": "red rod"}``).

    Tokens are always lowercased so that a model reply of ``{"shape": "Red
    Rect", "color": "BLUE"}`` still matches the lowercase synonym sets.
    """

    __slots__ = ('token_set',)

    def __init__(self, color: str, shape: str) -> None:
        # One combined bag, not two: the model might put the colour in the shape
        # field, so which field a token arrived in carries no information.
        self.token_set = set(_normalise_tokens(color)) | set(_normalise_tokens(shape))

    def matches_colour(self, colour: str) -> bool:
        # Keys are canonical (lowercase); also compare against normalised tokens
        # so mixed-case inputs still resolve.
        key: str = colour.lower().strip()
        return bool(self.token_set & _COLOUR_SYNONYMS.get(key, _NO_SYNONYMS))

    def matches_shape(self, shape: str) -> bool:
        key: str = shape.lower().strip()
        return bool(self.token_set & _SHAPE_SYNONYMS.get(key, _NO_SYNONYMS))


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    if not text:
        return None
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        candidate = text[start:end + 1] if (start != -1 and end > start) else None
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def score_probe_response(text: str) -> float:
    data = _extract_json(text)
    if not data:
        return 0.0

    score = 0.0

    # ---- count ------------------------------------------------------------
    # Accepted as a number or as its digits: models answer this field both ways,
    # and "3" is the same observation as 3. Anything else (a list, a null, a
    # word) simply does not score — it is not evidence the model counted.
    count = data.get('number_of_shapes')
    if isinstance(count, (int, float, str)) and not isinstance(count, bool):
        try:
            if int(count) == _EXPECTED_COUNT:
                score += _COUNT_WEIGHT
        except ValueError:
            pass

    # ---- build bags -------------------------------------------------------
    bags: List[_TokenBag] = []
    shapes = data.get('shapes')
    if isinstance(shapes, list):
        for item in shapes:
            if not isinstance(item, dict):
                continue
            color = str(item.get('color', '') or '').strip()
            shape = str(item.get('shape', '') or '').strip()
            if color or shape:
                bags.append(_TokenBag(color, shape))

    # ---- attribute matching -----------------------------------------------
    # Colour and shape score independently, so a reply naming one correctly is
    # worth more than a reply naming neither — the probe is asking whether the
    # model saw the image, not whether it phrased the answer our way.
    used_indices: Set[int] = set()
    for expected_shape, expected_colour in _EXPECTED_SHAPES:
        chosen_idx: Optional[int] = None

        # Prefer the first unconsumed entry matching the expected colour;
        # colour is the more discriminating of the two, since every shape
        # synonym set overlaps ordinary description ("round", "box").
        for i, bag in enumerate(bags):
            if i not in used_indices and bag.matches_colour(expected_colour):
                chosen_idx = i
                break

        if chosen_idx is None:
            for i, bag in enumerate(bags):
                if i not in used_indices and bag.matches_shape(expected_shape):
                    chosen_idx = i
                    break

        if chosen_idx is None:
            continue

        # One reported entry answers for at most one expected shape, so three
        # copies of the same correct answer cannot score as three shapes seen.
        used_indices.add(chosen_idx)
        bag = bags[chosen_idx]
        if bag.matches_colour(expected_colour):
            score += _ATTRIBUTE_HALF_WEIGHT
        if bag.matches_shape(expected_shape):
            score += _ATTRIBUTE_HALF_WEIGHT

    # ---- text -------------------------------------------------------------
    # Compared through the shared normaliser, so punctuation and spacing in the
    # model's transcription are irrelevant while the words must be exact.
    reported_text = _normalise_tokens(str(data.get('text', '') or ''))
    if reported_text == _EXPECTED_TEXT_TOKENS:
        score += _TEXT_WEIGHT

    return round(score, 4)


def probe_provider(provider: Dict[str, object]) -> bool:
    try:
        from services.file_mapper_service import FileMapperService
        from services import vision_service
        asset_path = FileMapperService.get_backend_path('vision', 'vision-test.png')
        with open(asset_path, 'rb') as fh:
            image_bytes = fh.read()
        config = vision_service.build_vision_config(provider)
        reply = vision_service.send_image_with_config(
            config, image_bytes, PROBE_PROMPT, mime_type='image/png',
        )
        if not reply:
            return False
        score = score_probe_response(reply)
        passed = score >= PASS_THRESHOLD
        logger.info(
            "[VisionProbe] name=%s platform=%s model=%s score=%.2f pass=%s",
            provider.get('name'), provider.get('platform'),
            provider.get('model'), score, passed,
        )
        return passed
    except Exception as exc:
        logger.warning("[VisionProbe] probe failed: %s", exc)
        return False
