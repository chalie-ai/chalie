"""Thread-gist think-block stripping — the persisted-label leak fix.

Reasoning models can leak chain-of-thought into the gist delegate's text:
either a well-formed ``<think>...</think>`` block, or — as observed live — an
UNCLOSED ``<think>`` block with the label glued straight onto the reasoning
(no closing tag, no delimiter), which a closed-pair regex cannot touch. The
production seam is ``generate_gist`` in
``services/thread_gist_message_processor.py``: it strips at the single point a
raw completion becomes a storable label, and answers ``None`` for one that
strips to empty so no caller has anything to persist. This file drives that
exact production function — the real delegate MP, PromptService and DB, with
only the LLM transport stubbed — and reads stored rows back through the real
``bulk_get`` batch-read the thread feed uses.
"""

import sqlite3
from unittest.mock import patch

import pytest

from configs.channels.thread_gist import ThreadGistConfig
from controllers.message_processor import MessageProcessor
from models.thread_gist import ThreadGist
from services.thread_gist_message_processor import generate_gist
from tests.helpers import LabelProvider, seed_selected_provider

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _generate(db: sqlite3.Connection, turn_id: int, raw_completion: str) -> str | None:
    """Run the real gist delegate over a fixed completion the model 'returned'."""
    seed_selected_provider(db)
    with patch("services.provider_service.build_client", return_value=LabelProvider(raw_completion)):
        return generate_gist(_CHANNEL, turn_id, "Yo")


def _labels(turn_ids: list[int]) -> dict[int, str]:
    return MessageProcessor(ThreadGistConfig()).gist_service.bulk_get(_CHANNEL, turn_ids)


def test_gist_with_think_block_is_persisted_clean(db: sqlite3.Connection) -> None:
    """A gist carrying a well-formed <think>...</think> block must be stored
    with the reasoning noise removed — never leaked into the user-facing label."""
    label = _generate(db, 101, "<think>reasoning noise</think>Weekend Malta trip planning")
    assert label == "Weekend Malta trip planning"

    ThreadGist(channel=_CHANNEL, turn_id=101, gist=label).upsert()
    labels = _labels([101])

    assert labels[101] == "Weekend Malta trip planning"
    assert "<think>" not in labels[101]
    assert "</think>" not in labels[101]


def test_gist_without_think_block_is_stored_unchanged(db: sqlite3.Connection) -> None:
    """A normal gist with no think tags must survive the strip untouched — the
    should-not-fire path, guarding against over-stripping ordinary labels."""
    label = _generate(db, 102, "Mac Mini Research")
    assert label == "Mac Mini Research"

    ThreadGist(channel=_CHANNEL, turn_id=102, gist=label).upsert()

    assert _labels([102])[102] == "Mac Mini Research"


def test_unclosed_think_block_is_never_stored(db: sqlite3.Connection) -> None:
    """The live defect shape: no closing tag, label glued onto the reasoning
    with no delimiter. Nothing after the opener is mechanically separable, so
    the whole gist is dropped — a missing label beats persisting raw
    chain-of-thought as a user-facing pill. The caller is handed nothing to
    store, but ``""`` rather than None: the delegate did answer, so a caller
    that retries on no-answer must not retry this."""
    raw = (
        '<think>\nThe user sent just "Yo" - a casual greeting. I need to create '
        "a terse 3-5 word topical label for this message.Casual Greeting Message"
    )

    assert _generate(db, 103, raw) == ""
    assert 103 not in _labels([103])


def test_a_whitespace_only_completion_is_dropped(db: sqlite3.Connection) -> None:
    """``_strip_think_blocks`` returns think-free text verbatim, so a reply of
    pure whitespace would pass a bare truthiness check and be handed on as a
    label. It must be dropped like any other unusable completion."""
    assert _generate(db, 104, "   \n  ") is None
    assert 104 not in _labels([104])
