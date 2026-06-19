# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""user_messages_total counts — dashboard derives on demand from transcript COUNT."""

import sqlite3
from typing import cast

import pytest

from services.memory_store import MemoryStore

pytestmark = pytest.mark.unit


# ── user_messages_total deleted as stored counter; dashboard uses COUNT ────────


def test_stored_counter_does_not_affect_user_messages(db: sqlite3.Connection, store: MemoryStore) -> None:
    from services.metrics_service import MetricsService

    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES "
        "('user', 'user', 'one')"
    )
    db.commit()

    m = MetricsService()
    m.record_counter('user_messages_total', 999)  # stale / ignored
    dash = m.get_dashboard_data()
    assert cast("dict[str, object]", dash['counters'])['user_messages_total'] == 1


# ── one user turn = exactly one user/user transcript row ───────────────────────


def test_count_exactness_n_turns(db: sqlite3.Connection, store: MemoryStore) -> None:
    from services.metrics_service import MetricsService

    for i in range(5):
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES "
            "('user', 'user', ?)",
            (f'turn-{i}',),
        )
    # Rows on other channels/roles must not inflate the count.
    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES "
        "('user', 'assistant', 'a'), ('dmn', 'user', 'b'), "
        "('external-agent:bot', 'external_agent', 'c')"
    )
    db.commit()

    dash = MetricsService().get_dashboard_data()
    assert cast("dict[str, object]", dash['counters'])['user_messages_total'] == 5
