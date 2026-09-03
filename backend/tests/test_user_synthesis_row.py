"""Regression guard for :class:`models.user_synthesis.UserSynthesisRow`.

Real-SQLite round-trips over the conftest ``db`` fixture (isolated database,
zero mocks). Covers the append-only versioned contract:

* two successive :meth:`write` calls yield versions 1 then 2, and
  :meth:`latest` returns the second row;
* :meth:`latest_content` returns the second body after two writes;
* on an empty table :meth:`latest` is ``None`` and :meth:`latest_content` is ``""``.
"""

import sqlite3

import pytest

from models.user_synthesis import UserSynthesisRow

pytestmark = pytest.mark.unit


class TestWriteAndLatest:

    def test_two_writes_yield_versions_1_and_2(self, db: sqlite3.Connection) -> None:
        """Two :meth:`write` calls produce rows whose ``version`` columns are
        1 then 2 in insertion order."""
        UserSynthesisRow.write("first body")
        UserSynthesisRow.write("second body")

        rows = UserSynthesisRow.order_by("id ASC").get()
        assert len(rows) == 2
        assert rows[0].version == 1
        assert rows[0].content == "first body"
        assert rows[1].version == 2
        assert rows[1].content == "second body"

    def test_latest_is_second_row_after_two_writes(self, db: sqlite3.Connection) -> None:
        """After two writes :meth:`latest` returns the newest (highest id) row."""
        UserSynthesisRow.write("first body")
        UserSynthesisRow.write("second body")

        row = UserSynthesisRow.latest()
        assert row is not None
        assert row.content == "second body"

    def test_latest_content_returns_second_body(self, db: sqlite3.Connection) -> None:
        """After two writes :meth:`latest_content` returns the second body."""
        UserSynthesisRow.write("first body")
        UserSynthesisRow.write("second body")

        assert UserSynthesisRow.latest_content() == "second body"


class TestEmptyTable:

    def test_latest_returns_none_when_empty(self, db: sqlite3.Connection) -> None:
        """No rows at all → :meth:`latest` is ``None``."""
        assert UserSynthesisRow.latest() is None

    def test_latest_content_returns_empty_string_when_empty(self, db: sqlite3.Connection) -> None:
        """No rows at all → :meth:`latest_content` is ``""``."""
        assert UserSynthesisRow.latest_content() == ""
