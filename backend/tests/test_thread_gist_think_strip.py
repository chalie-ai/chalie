"""Thread-gist think-block stripping — the persisted-label leak fix.

Think-stripping historically only happened inside the OpenAI provider client,
so a non-OpenAI delegate provider (or a model that emits an unclosed think
block) would otherwise leak raw ``<think>...</think>`` chain-of-thought text
into the ``thread_gist`` row — a user-facing collapsed-thread pill. The fix
(``services/thread_gist_message_processor.py:83``) strips at the single point
the gist becomes persisted data: ``inert.gist_service.upsert(channel, turn_id,
_strip_think_blocks(gist))``. This file drives that exact call — the real
``GistService`` against a real DB — and reads the row back through the real
``bulk_get`` batch-read the thread feed uses.
"""

import sqlite3

import pytest

from configs.channels.thread_gist import ThreadGistConfig
from controllers.message_processor import MessageProcessor
from services.llm_service import _strip_think_blocks

pytestmark = pytest.mark.unit

_CHANNEL = "user"


def _persist_gist(turn_id: int, raw_gist: str) -> None:
    """The exact production call site: bare inert construction (I2, no DB/WS
    side effects at construction), then the real strip-then-upsert pair."""
    inert = MessageProcessor(ThreadGistConfig())
    inert.gist_service.upsert(_CHANNEL, turn_id, _strip_think_blocks(raw_gist))


def test_gist_with_think_block_is_persisted_clean(db: sqlite3.Connection) -> None:
    """A gist carrying a well-formed <think>...</think> block must be stored
    with the reasoning noise removed — never leaked into the user-facing label."""
    _persist_gist(101, "<think>reasoning noise</think>Weekend Malta trip planning")

    labels = MessageProcessor(ThreadGistConfig()).gist_service.bulk_get(_CHANNEL, [101])

    assert labels[101] == "Weekend Malta trip planning"
    assert "<think>" not in labels[101]
    assert "</think>" not in labels[101]


def test_gist_without_think_block_is_stored_unchanged(db: sqlite3.Connection) -> None:
    """A normal gist with no think tags must survive the strip untouched — the
    should-not-fire path, guarding against over-stripping ordinary labels."""
    _persist_gist(102, "Mac Mini Research")

    labels = MessageProcessor(ThreadGistConfig()).gist_service.bulk_get(_CHANNEL, [102])

    assert labels[102] == "Mac Mini Research"
