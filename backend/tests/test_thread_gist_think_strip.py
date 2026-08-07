"""Thread-gist think-block stripping — the persisted-label leak fix.

Reasoning models can leak chain-of-thought into the gist delegate's text:
either a well-formed ``<think>...</think>`` block, or — as observed live — an
UNCLOSED ``<think>`` block with the label glued straight onto the reasoning
(no closing tag, no delimiter), which a closed-pair regex cannot touch. The
production seam is ``_persist_gist`` in
``services/thread_gist_message_processor.py``: it strips at the single point
the gist becomes persisted, user-facing data, and refuses to store a gist
that strips to empty. This file drives that exact production function — the
real ``GistService`` against a real DB — and reads rows back through the real
``bulk_get`` batch-read the thread feed uses.
"""

import sqlite3

import pytest

from configs.channels.thread_gist import ThreadGistConfig
from controllers.message_processor import MessageProcessor
from services.thread_gist_message_processor import ThreadGistMessageProcessor

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _labels(turn_ids: list[int]) -> dict[int, str]:
    return MessageProcessor(ThreadGistConfig()).gist_service.bulk_get(_CHANNEL, turn_ids)


def test_gist_with_think_block_is_persisted_clean(db: sqlite3.Connection) -> None:
    """A gist carrying a well-formed <think>...</think> block must be stored
    with the reasoning noise removed — never leaked into the user-facing label."""
    stored = ThreadGistMessageProcessor._persist_gist(_CHANNEL, 101, "<think>reasoning noise</think>Weekend Malta trip planning")

    labels = _labels([101])

    assert stored is True
    assert labels[101] == "Weekend Malta trip planning"
    assert "<think>" not in labels[101]
    assert "</think>" not in labels[101]


def test_gist_without_think_block_is_stored_unchanged(db: sqlite3.Connection) -> None:
    """A normal gist with no think tags must survive the strip untouched — the
    should-not-fire path, guarding against over-stripping ordinary labels."""
    stored = ThreadGistMessageProcessor._persist_gist(_CHANNEL, 102, "Mac Mini Research")

    labels = _labels([102])

    assert stored is True
    assert labels[102] == "Mac Mini Research"


def test_unclosed_think_block_is_never_stored(db: sqlite3.Connection) -> None:
    """The live defect shape: no closing tag, label glued onto the reasoning
    with no delimiter. Nothing after the opener is mechanically separable, so
    the whole gist is dropped — a missing label beats persisting raw
    chain-of-thought as a user-facing pill."""
    raw = (
        '<think>\nThe user sent just "Yo" - a casual greeting. I need to create '
        "a terse 3-5 word topical label for this message.Casual Greeting Message"
    )
    stored = ThreadGistMessageProcessor._persist_gist(_CHANNEL, 103, raw)

    labels = _labels([103])

    assert stored is False
    assert 103 not in labels
