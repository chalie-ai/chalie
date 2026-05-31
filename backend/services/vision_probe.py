"""Vision probe — verify a provider actually understands images.

Sends a known test image (3 shapes + text) with an exact-JSON prompt, scores the
structured reply against the answer key, and returns True iff score ≥ 0.80. This
defeats silent-ignore false positives (models that accept an image field but
never look at it). Fully defensive — any failure returns False / 0.0.
"""

import json
import logging
import re
from typing import Dict, Any, Optional

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

# Answer key for backend/vision/vision-test.png
_EXPECTED_COUNT = 3
_EXPECTED_SHAPES = {('rectangle', 'red'), ('circle', 'yellow'), ('hexagon', 'green')}
# Deliberately lowercase: score_probe_response compares the model's reply via
# .strip().lower(), so this constant MUST be lowercase. The image itself reads
# "Chalie can read!" — do not "correct" the capitalisation here or the text
# score silently drops to 0 and every provider fails the 0.80 threshold.
_EXPECTED_TEXT = 'chalie can read!'


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the first JSON object in the model reply."""
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
    """Score a probe reply against the answer key. Max 1.0.

    +0.30 number_of_shapes == 3
    +0.15 per correct (shape name AND color, case-insensitive), max 3 → 0.45
    +0.25 text == 'Chalie can read!' (case-insensitive, trimmed)
    """
    data = _extract_json(text)
    if not data:
        return 0.0

    score = 0.0

    try:
        if int(data.get('number_of_shapes')) == _EXPECTED_COUNT:
            score += 0.30
    except (ValueError, TypeError):
        pass

    seen = set()
    shapes = data.get('shapes')
    if isinstance(shapes, list):
        for item in shapes:
            if not isinstance(item, dict):
                continue
            pair = (
                str(item.get('shape', '')).strip().lower(),
                str(item.get('color', '')).strip().lower(),
            )
            if pair in _EXPECTED_SHAPES and pair not in seen:
                seen.add(pair)
                score += 0.15

    text_val = data.get('text')
    if isinstance(text_val, str) and text_val.strip().lower() == _EXPECTED_TEXT:
        score += 0.25

    return round(score, 4)


def probe_provider(provider: Dict[str, Any]) -> bool:
    """Return True iff *provider* scores ≥ 0.80 on the vision probe.

    *provider* is a dict with platform/model/api_key/host (plaintext api_key).
    Any failure (missing asset, network, parse, low score) returns False.
    """
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
