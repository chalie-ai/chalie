"""Feature tests for :class:`services.user_synthesis.UserSynthesis`.

Real-DataGraph round-trips over the conftest ``db`` fixture (isolated SQLite,
zero mocks). Covers the single-key contract every caller depends on:

* ``upsert`` writes the one ``user_summary`` row ``get`` reads back;
* ``upsert`` overwrites in place;
* an absent synthesis reads back as ``""`` (the falsy skip-gate every caller uses).
"""

import sqlite3

import pytest

from models.machine_state import MachineStateRow
from services.user_synthesis import UserSynthesis

pytestmark = pytest.mark.unit


class TestGet:

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
