# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test — the super-episode encoder distils child gists only and never
re-hydrates raw transcript turns into the encoder prompt, at any level.

The distillation hierarchy must contract at every step: L0 episodes -> L1
super-ep -> L2 -> ... each level summarises the one below it from its child
gists ALONE. Re-hydrating the raw turns behind those gists re-expands the whole
subtree into the encoder prompt (the ~30GB OOM regression), so the assembly
feeds the model the source gists and nothing else — there is no ``level`` knob
and no raw-spans section, even when the source episodes carry real
``transcript_ids`` pointing at real rows.

Driven on the real assembly path with real SQLite (``db``), a real
``SuperEpisodeConfig`` and the production ``PromptService`` (``mp.prompt_service``,
the same service the turn's request assembly reads from) — zero mocks. We build a
real MessageProcessor and assert on the returned user-prompt string, i.e.
precisely what the encoder is fed.
"""

import sqlite3
from typing import cast

import pytest

from configs.channels import SuperEpisodeConfig, _collect_transcript_ids
from controllers.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

# Any raw-turn re-hydration used this heading; it must never appear again.
_SPANS_MARKER = "Raw transcript spans"

# Verbatim content of a real transcript row. Its absence from the prompt is the
# regression guard, so it must not be a substring of any source gist below.
_RAW_TURN_CONTENT = "what is left before launch, and the payments migration status"

# Two source episode gists — the distillation input. A super-ep is always the
# summary of its children's gists, at every level.
_SOURCES: list[object] = [
    {"id": 101, "gist": "Planned the Q3 launch timeline with the design team."},
    {"id": 102, "gist": "Resolved the payments migration blocker with finance."},
]


def _seed_transcript(db: sqlite3.Connection, content: str) -> int:
    """Insert one real transcript row exactly as production does; return its id."""
    cur = db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
        ("user", "user", content),
    )
    db.commit()
    return cast(int, cur.lastrowid)


def _encoder_prompt(sources: list[object]) -> str:
    """Build a real MessageProcessor for a ``SuperEpisodeConfig`` over *sources*
    and return the user-prompt the encoder turn would send — the production
    ``PromptService.user_prompt`` output for the super-episode channel."""
    config = SuperEpisodeConfig("user", sources)
    mp = MessageProcessor(config, raw_input="")
    return mp.prompt_service.user_prompt()


def test_prompt_carries_child_gists_and_no_spans_section(db: sqlite3.Connection) -> None:
    # The source gists are always present — they are the thing being consolidated
    # — and the prompt never carries a raw-transcript-spans section.
    prompt = _encoder_prompt(_SOURCES)

    assert "Planned the Q3 launch timeline" in prompt
    assert "Resolved the payments migration blocker" in prompt
    assert _SPANS_MARKER not in prompt, "encoder prompt re-introduced a raw-spans section"


def test_no_rehydration_even_with_real_transcript_ids(db: sqlite3.Connection) -> None:
    # Regression guard for the ~30GB OOM: source episodes that carry real
    # ``transcript_ids`` pointing at a real row must STILL not pull that raw turn
    # into the encoder prompt. The row exists in the DB, so its absence proves
    # the assembly never fetched it — distillation feeds gists only, every level.
    tid = _seed_transcript(db, _RAW_TURN_CONTENT)
    sources: list[object] = [
        {
            "id": 201,
            "gist": "Summarised the launch-readiness thread.",
            "transcript_ids": [tid],
            "transcript_id_start": tid,
            "transcript_id_end": tid,
        },
        {
            "id": 202,
            "gist": "Summarised the finance sign-off thread.",
            "transcript_ids": [tid],
            "transcript_id_start": tid,
            "transcript_id_end": tid,
        },
    ]

    prompt = _encoder_prompt(sources)

    # Gists still drive the synthesis, so the level stays productive...
    assert "Summarised the launch-readiness thread." in prompt
    assert "Summarised the finance sign-off thread." in prompt
    # ...but the raw turn behind them is never re-hydrated.
    assert _SPANS_MARKER not in prompt
    assert _RAW_TURN_CONTENT not in prompt, (
        "encoder re-hydrated a raw transcript turn — distillation inverted, OOM regression"
    )


def test_provenance_is_the_sparse_union_not_a_filled_range() -> None:
    # Lineage stamping (transcript_ids / transcript_id_start / transcript_id_end
    # on the parent super-episode) is the per-episode UNION of the ids each child
    # actually records — never the old min(start)..max(end) fill, which claimed
    # rows no child covered. Two children with a gap between their windows must
    # yield the two windows and nothing in the gap.
    children: list[object] = [
        {"id": 1, "transcript_ids": [100, 101, 102]},
        {"id": 2, "transcript_ids": [200, 201]},
    ]

    ids = _collect_transcript_ids(children)

    assert ids == {100, 101, 102, 200, 201}
    assert 150 not in ids, "range-fill regressed — gap rows no child covered were re-introduced"
    # start/end stamped from the union stay correct (this is what feeds the
    # extraction watermark, so it must not silently narrow or widen).
    assert min(ids) == 100
    assert max(ids) == 201
