"""Voice clean-up — one transient LLM pass that corrects a speech-to-text transcript.

The pass runs on the selected default model through the ordinary message
processor (thinking off, nothing stored, nothing broadcast). A guard rejects any
answer that is not plausibly a correction of the input — the model answering the
transcript instead of fixing it, rambling, or saying nothing — and the raw
transcript is returned instead, so clean-up can only ever improve on it.
"""

from __future__ import annotations

import logging
import re

from configs.channels import VoiceCleanupConfig
from controllers.message_processor import MessageProcessor
from services.llm_service import _strip_think_blocks

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_MIN_OVERLAP = 0.7
_MAX_GROWTH = 1.3
_MAX_EXTRA_TOKENS = 2


def clean_transcript(raw: str) -> str:
    """Return ``raw`` corrected by the selected model, or ``raw`` itself when it
    is blank, the call fails, or the answer does not look like a correction."""
    if not raw.strip():
        return raw
    try:
        answer = MessageProcessor.process(
            VoiceCleanupConfig(), raw_input=f"<transcript>\n{raw}\n</transcript>"
        ).result()
        cleaned = _strip_think_blocks(answer)[0].strip()
    except Exception:
        logger.exception("[Voice] clean-up failed, returning raw transcript")
        return raw
    return cleaned if _is_correction(raw, cleaned) else raw


def _is_correction(raw: str, cleaned: str) -> bool:
    """True when ``cleaned`` is plausibly ``raw`` with a few words fixed."""
    source = _TOKEN_RE.findall(raw.lower())
    answer = _TOKEN_RE.findall(cleaned.lower())
    if not answer:
        logger.info("[Voice] clean-up rejected: empty answer")
        return False
    known = set(source)
    if sum(1 for t in answer if t in known) / len(answer) < _MIN_OVERLAP:
        logger.info("[Voice] clean-up rejected: low overlap with the transcript")
        return False
    if len(answer) > _MAX_GROWTH * len(source) + _MAX_EXTRA_TOKENS:
        logger.info("[Voice] clean-up rejected: answer too long")
        return False
    return True
