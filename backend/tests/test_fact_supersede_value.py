# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature test: value-addressed coexist supersede.

Coexist-rule concept keys (``relationships``, ``family``, ``name``) append
values with no revision path: a corrected fact stored under the same canonical
key kept the stale value live forever ("Married to Tom" observed live next to
"Married to Thomas (goes by Tom)"). :meth:`FactRow.supersede_value`
plus the memory tool's optional ``replaces`` store param give the model a
deterministic demote+insert — no thresholds, no similarity heuristics.

Part 1 exercises the FactRow matching ladder pure-DB: the canonical key is
given directly, so no LUT or key-embedding sits in the loop. Part 2 drives the
real production path via ``DispatchService.dispatch("memory")`` with zero
mocks — real concept LUT, real embeddings — on the canonical coexist key
``relationships``, the exact key of the observed failure (precedent:
``test_memory_recall_guardrail_and_seed_radius.py``)."""

import sqlite3

import pytest

from configs.channels import UserConfig
from controllers.message_processor import MessageProcessor
from models.fact import FactRow

pytestmark = pytest.mark.unit

_KEY = "relationships"


def _build_user_mp(text: str) -> MessageProcessor:
    """Real MessageProcessor with the synchronous half of begin() replayed —
    same idiom as test_memory_recall_guardrail_and_seed_radius.py."""
    mp = MessageProcessor(UserConfig(), raw_input=text)
    mp.active_tools = list(mp.config.always_available or [])
    with mp.db.transaction():
        mp.turn_id = mp.transcript_service.allocate_turn()
        mp.uid = mp.transcript_service.append_input(mp.raw_input)
        mp.current_transcript_id = mp.uid
    return mp


def _row_state(db: sqlite3.Connection, row_id: "int | str | None") -> tuple[int, float, str | None]:
    row = db.execute(
        "SELECT active, retrieval_weight, valid_to FROM data_graph WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert row is not None
    return row[0], row[1], row[2]


# ── Part 1: the FactRow matching ladder, pure DB ────────────────────────────────


def test_exact_match_demotes_old_and_inserts_new(db: sqlite3.Connection) -> None:
    old_row, _, _ = FactRow.store_coexist(_KEY, "Married to Tom")
    FactRow.store_coexist(_KEY, "Son named Max")
    _, rw_before, _ = _row_state(db, old_row.id)

    result = FactRow.supersede_value(_KEY, "Married to Tom", "Married to Thomas (goes by Tom)")

    assert result is not None
    new_row, status, matched_old = result
    assert status == "superseded"
    assert matched_old == "Married to Tom"
    # Old row demoted: active=0, retrieval_weight halved, valid_to stamped.
    active, rw_after, valid_to = _row_state(db, old_row.id)
    assert active == 0
    assert rw_after == pytest.approx(rw_before * 0.5)
    assert valid_to is not None
    # Live set: the new value replaced the old; the unrelated value survived.
    live = FactRow.active_values(_KEY)
    assert sorted(live) == sorted(["Married to Thomas (goes by Tom)", "Son named Max"])
    assert new_row.value == "Married to Thomas (goes by Tom)"


def test_case_folded_match(db: sqlite3.Connection) -> None:
    assert db is not None
    FactRow.store_coexist(_KEY, "Married to Tom")

    result = FactRow.supersede_value(_KEY, "  MARRIED TO TOM ", "Married to Thomas")

    assert result is not None
    assert result[1] == "superseded"
    assert FactRow.active_values(_KEY) == ["Married to Thomas"]


def test_unique_substring_match(db: sqlite3.Connection) -> None:
    assert db is not None
    FactRow.store_coexist(_KEY, "Married to Tom")
    FactRow.store_coexist(_KEY, "Son named Max")

    # "Tom" appears in exactly one live value — rung (b) resolves it.
    result = FactRow.supersede_value(_KEY, "Tom", "Married to Thomas")

    assert result is not None
    _, status, matched_old = result
    assert status == "superseded"
    assert matched_old == "Married to Tom"
    assert sorted(FactRow.active_values(_KEY)) == sorted(["Married to Thomas", "Son named Max"])


def test_ambiguous_substring_returns_none(db: sqlite3.Connection) -> None:
    assert db is not None
    FactRow.store_coexist(_KEY, "Married to Tom")
    FactRow.store_coexist(_KEY, "Tommy is my cat")

    # "tom" is a substring of BOTH live values — ambiguous, no write.
    result = FactRow.supersede_value(_KEY, "tom", "Married to Thomas")

    assert result is None
    assert sorted(FactRow.active_values(_KEY)) == sorted(["Married to Tom", "Tommy is my cat"])


def test_absent_value_returns_none(db: sqlite3.Connection) -> None:
    assert db is not None
    FactRow.store_coexist(_KEY, "Married to Tom")

    result = FactRow.supersede_value(_KEY, "Best friend is Alice", "Best friend is Alicia")

    assert result is None
    assert FactRow.active_values(_KEY) == ["Married to Tom"]


def test_same_value_reinforces_instead_of_churning(db: sqlite3.Connection) -> None:
    row, _, _ = FactRow.store_coexist(_KEY, "Married to Tom")

    # Old and new resolve to the same live row — reinforce, never demote+reinsert.
    result = FactRow.supersede_value(_KEY, "Married to Tom", "married to tom")

    assert result is not None
    reinforced, status, matched_old = result
    assert status == "reinforced"
    assert matched_old == "Married to Tom"
    assert reinforced.id == row.id
    active, _, _ = _row_state(db, row.id)
    assert active == 1
    assert FactRow.active_values(_KEY) == ["Married to Tom"]



