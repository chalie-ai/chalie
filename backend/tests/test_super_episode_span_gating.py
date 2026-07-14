# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — the super-episode encoder distils gists at higher levels and
only re-hydrates raw transcript spans at the leaf (level-1) consolidation.

The distillation hierarchy must contract at every step: L0 episodes -> L1
super-ep (grounded in the raw turns the gists were summarised from), then L1 ->
L2 -> ... encoded from the *child gists alone*. Re-hydrating raw spans above the
leaf re-expands the entire subtree into the encoder prompt (the OOM regression),
so ``consolidate.py`` passes ``spans=""`` for level >= 2 and the prompt builder
must then omit the raw-spans section entirely — the model sees gists only.

Driven on the real assembly path with real SQLite (``db``), a real
``SuperEpisodeConfig`` and the production ``PromptService`` (``mp.prompt_service``,
the same service the turn's request assembly reads from) — zero mocks. We build a
real MessageProcessor per case and assert on the returned user-prompt string,
i.e. precisely what the encoder is fed.
"""

import sqlite3
from typing import cast

import pytest

from configs.channels import SuperEpisodeConfig, _spans_for_level
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_SPANS_MARKER = "Raw transcript spans covering these episodes:"

# Two source episode gists — the distillation input that must be present at
# every level (a super-ep is always the summary of its children's gists).
_SOURCES: list[object] = [
    {"id": 101, "gist": "Planned the Q3 launch timeline with the design team."},
    {"id": 102, "gist": "Resolved the payments migration blocker with finance."},
]

# A raw transcript span blob as the leaf round would fetch it.
_RAW_SPANS = "[2959] (2026-07-01) user: what's left before launch?\n[2960] (2026-07-01) assistant: the payments migration."


def _seed_transcript(db: sqlite3.Connection, content: str) -> int:
    """Insert one real transcript row exactly as production does; return its id."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", content),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def test_spans_for_level_leaf_fetches_real_rows(db: sqlite3.Connection) -> None:
    # The level-1 gate (the exact line consolidate.py calls) must fetch the real
    # transcript rows for the given ids and return their content.
    id_a = _seed_transcript(db, "leaf turn about the launch timeline")
    id_b = _seed_transcript(db, "leaf turn about the payments blocker")

    spans = _spans_for_level({id_a, id_b}, level=1)

    assert "leaf turn about the launch timeline" in spans
    assert "leaf turn about the payments blocker" in spans


def test_spans_for_level_higher_skips_fetch_despite_real_rows(db: sqlite3.Connection) -> None:
    # Same non-empty id set, same rows present in the DB — but at level >= 2 the
    # gate must return "" WITHOUT re-hydrating them. This is the regression
    # guard: reverting the gate to an unconditional fetch (the ~30GB OOM line)
    # would return the row content here and fail. The empty result is the level
    # decision, NOT an empty id set.
    id_a = _seed_transcript(db, "era turn that must not be re-hydrated")
    id_b = _seed_transcript(db, "another era turn that must not be re-hydrated")

    assert _spans_for_level({id_a, id_b}, level=2) == ""
    assert _spans_for_level({id_a, id_b}, level=3) == ""


def _encoder_prompt(spans: str) -> str:
    """Build a real MessageProcessor for a SuperEpisodeConfig carrying *spans*
    and return the user-prompt the encoder turn would send — the production
    ``PromptService.user_prompt`` output for the super-episode channel."""
    config = SuperEpisodeConfig("user", _SOURCES, spans)
    mp = MessageProcessor(config, raw_input="")
    return mp.prompt_service.user_prompt()


def test_leaf_level_includes_raw_spans(db: sqlite3.Connection) -> None:
    # Level 1 (leaf -> super): consolidate.py fetches and passes the raw spans,
    # grounding the synthesis in the turns the gists were summarised from.
    prompt = _encoder_prompt(_RAW_SPANS)

    assert _SPANS_MARKER in prompt, "leaf-level encoder prompt dropped its raw transcript spans"
    assert _RAW_SPANS in prompt
    # The source gists are always present — they are the thing being consolidated.
    assert "Planned the Q3 launch timeline" in prompt
    assert "Resolved the payments migration blocker" in prompt


def test_higher_level_distils_gists_without_raw_spans(db: sqlite3.Connection) -> None:
    # Level >= 2 (era and up): consolidate.py passes spans="" so nothing is
    # re-hydrated. The prompt must carry the child gists and NO raw-spans section
    # — this is the contraction that keeps the encoder under its token cap.
    prompt = _encoder_prompt("")

    assert _SPANS_MARKER not in prompt, "higher-level encoder re-hydrated raw spans — distillation inverted"
    # Gists still drive the synthesis, so the level stays productive.
    assert "Planned the Q3 launch timeline" in prompt
    assert "Resolved the payments migration blocker" in prompt
