"""Feature tests for :class:`services.user_synthesis.UserSynthesis`.

Real-DataGraph round-trips over the conftest ``db`` fixture (isolated SQLite,
zero mocks). Covers the single-key contract every caller depends on:

* the channel-response ingest (``persist_user_summary``) writes the one
  ``user_summary`` row ``get`` reads back — and never a ``user_summary_long``
  row (the removed dual-key contract must stay gone);
* malformed or incomplete responses are dropped wholesale, never raised;
* ``upsert`` overwrites in place;
* an absent synthesis reads back as ``""`` (the falsy skip-gate every caller uses).
"""

import json
import sqlite3

import pytest

from models.machine_state import MachineStateRow
from services.user_synthesis import UserSynthesis

pytestmark = pytest.mark.unit

_RESPONSE = json.dumps({"summary": "A senior architect who loves espresso."})


class TestPersistAndGet:

    def test_persist_writes_the_single_row_get_reads_back(self, db: sqlite3.Connection) -> None:
        """persist_user_summary ingests {"summary": ...} into the 'user_summary'
        row; the long row of the removed dual-key contract is never written."""
        UserSynthesis.persist_user_summary(_RESPONSE)

        assert UserSynthesis.get() == "A senior architect who loves espresso."
        row = MachineStateRow.active_by_key("user_summary")
        assert row is not None and row.value == "A senior architect who loves espresso."
        assert MachineStateRow.active_by_key("user_summary_long") is None

    def test_code_fenced_response_is_unwrapped(self, db: sqlite3.Connection) -> None:
        """A ```json-fenced response still parses — the ingest strips fences."""
        UserSynthesis.persist_user_summary(f"```json\n{_RESPONSE}\n```")

        assert UserSynthesis.get() == "A senior architect who loves espresso."

    def test_absent_synthesis_reads_empty(self, db: sqlite3.Connection) -> None:
        """No row → "" (the falsy skip-gate callers rely on)."""
        assert UserSynthesis.get() == ""


class TestUpsert:

    def test_upsert_targets_the_user_summary_key(self, db: sqlite3.Connection) -> None:
        """upsert writes the 'user_summary' machine-state row."""
        UserSynthesis.upsert("Short portrait.")

        row = MachineStateRow.active_by_key("user_summary")
        assert row is not None and row.value == "Short portrait."

    def test_upsert_overwrites_prior_value(self, db: sqlite3.Connection) -> None:
        """A second upsert supersedes the first — get returns the latest."""
        UserSynthesis.upsert("First.")
        UserSynthesis.upsert("Second.")

        assert UserSynthesis.get() == "Second."


class TestPersistGuards:

    def test_malformed_json_is_skipped_not_raised(self, db: sqlite3.Connection) -> None:
        """A non-JSON channel response is logged and dropped — nothing stored,
        nothing raised (a bad turn must never poison the system prompt)."""
        UserSynthesis.persist_user_summary("not json at all")

        assert UserSynthesis.get() == ""

    def test_missing_summary_key_writes_nothing(self, db: sqlite3.Connection) -> None:
        """A response without 'summary' is rejected wholesale."""
        UserSynthesis.persist_user_summary(json.dumps({"other": "value"}))

        assert UserSynthesis.get() == ""

    def test_empty_summary_writes_nothing(self, db: sqlite3.Connection) -> None:
        """An empty 'summary' value is rejected, not stored."""
        UserSynthesis.persist_user_summary(json.dumps({"summary": ""}))

        assert UserSynthesis.get() == ""
