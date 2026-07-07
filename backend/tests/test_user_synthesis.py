"""Feature tests for :class:`services.user_synthesis.UserSynthesis`.

Real-DataGraph round-trips over the conftest ``db`` fixture (isolated SQLite,
zero mocks). Covers the behaviour every former caller depended on:

* the channel-response ingest (``persist_user_summary``) writes both rows, and
  ``get`` reads each back at its own shorthand;
* the long read falls back to the short synthesis when no long row exists — the
  contract DMN and the prompt spine relied on;
* ``upsert`` routes to the right key per ``shorthand``;
* an absent synthesis reads back as ``""`` (the falsy skip-gate every caller uses).
"""

import json
import sqlite3

import pytest

from models.system import SystemRow
from services.user_synthesis import UserSynthesis

pytestmark = pytest.mark.unit

_RESPONSE = json.dumps({"short": "Loves espresso.", "long": "A senior architect who loves espresso and jazz."})


class TestPersistAndGet:

    def test_persist_writes_both_rows_readable_at_each_shorthand(self, db: sqlite3.Connection) -> None:
        """persist_user_summary ingests the {short, long} response; get returns each."""
        UserSynthesis.persist_user_summary(_RESPONSE)

        assert UserSynthesis.get(shorthand=True) == "Loves espresso."
        assert UserSynthesis.get(shorthand=False) == "A senior architect who loves espresso and jazz."

    def test_default_get_is_long(self, db: sqlite3.Connection) -> None:
        """get() with no arg returns the long synthesis (shorthand defaults False)."""
        UserSynthesis.persist_user_summary(_RESPONSE)

        assert UserSynthesis.get() == "A senior architect who loves espresso and jazz."

    def test_long_read_falls_back_to_short(self, db: sqlite3.Connection) -> None:
        """When only the short row exists, the long read serves the short one —
        the fallback DMN and _dmn_prompt depend on."""
        UserSynthesis.upsert("Loves espresso.", shorthand=True)

        assert UserSynthesis.get(shorthand=False) == "Loves espresso."

    def test_short_read_does_not_fall_back_to_long(self, db: sqlite3.Connection) -> None:
        """The short read is exact — it never serves the long row (matches the
        pre-consolidation user_summary() semantics)."""
        UserSynthesis.upsert("The long one only.", shorthand=False)

        assert UserSynthesis.get(shorthand=True) == ""

    def test_absent_synthesis_reads_empty(self, db: sqlite3.Connection) -> None:
        """No rows → "" at either shorthand (the falsy skip-gate callers rely on)."""
        assert UserSynthesis.get(shorthand=True) == ""
        assert UserSynthesis.get(shorthand=False) == ""


class TestUpsertRouting:

    def test_upsert_default_targets_short_key(self, db: sqlite3.Connection) -> None:
        """upsert defaults to shorthand=True → the 'user_summary' row."""
        UserSynthesis.upsert("Short portrait.")

        short = SystemRow.active_by_key("user_summary")
        assert short is not None and short.value == "Short portrait."
        assert SystemRow.active_by_key("user_summary_long") is None

    def test_upsert_long_targets_long_key(self, db: sqlite3.Connection) -> None:
        """upsert(shorthand=False) → the 'user_summary_long' row."""
        UserSynthesis.upsert("Long portrait.", shorthand=False)

        long_ = SystemRow.active_by_key("user_summary_long")
        assert long_ is not None and long_.value == "Long portrait."
        assert SystemRow.active_by_key("user_summary") is None

    def test_upsert_overwrites_prior_value(self, db: sqlite3.Connection) -> None:
        """A second upsert supersedes the first — get returns the latest."""
        UserSynthesis.upsert("First.", shorthand=True)
        UserSynthesis.upsert("Second.", shorthand=True)

        assert UserSynthesis.get(shorthand=True) == "Second."


class TestPersistGuards:

    def test_malformed_json_is_skipped_not_raised(self, db: sqlite3.Connection) -> None:
        """A non-JSON channel response is logged and dropped — never raised, and
        nothing is written."""
        UserSynthesis.persist_user_summary("not json at all")

        assert UserSynthesis.get(shorthand=True) == ""
        assert UserSynthesis.get(shorthand=False) == ""

    def test_missing_field_writes_nothing(self, db: sqlite3.Connection) -> None:
        """A response missing 'long' is rejected wholesale — neither row is written."""
        UserSynthesis.persist_user_summary(json.dumps({"short": "only short"}))

        assert UserSynthesis.get(shorthand=True) == ""
