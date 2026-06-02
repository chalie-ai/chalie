# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""user_messages_total — read-time transcript COUNT (spec §9).

``user_messages_total`` is deleted as a stored counter; the dashboard derives
it on demand from the transcript table:

    SELECT COUNT(*) FROM transcript WHERE role='user' AND channel='user'

These two tests exercise that pure read-path against a real SQLite database (no
mocked collaborators): a stale stored counter is ignored, and the COUNT is exact
and scoped to user/user rows. The per-send gateway wiring (token accumulation,
request/turn counters) is covered end-to-end by nightly scenario
052-tool-code-eval (message-event ``metrics`` object).
"""

import pytest

pytestmark = pytest.mark.unit


# ── user_messages_total deleted as stored counter; dashboard uses COUNT ────────


def test_stored_counter_does_not_affect_user_messages(db, store):
    """Recording a stale user_messages_total counter does NOT change the
    dashboard value — the dashboard reads the transcript COUNT only."""
    from services.metrics_service import MetricsService

    db.execute(
        "INSERT INTO transcript (channel, role, content) VALUES "
        "('user', 'user', 'one')"
    )
    db.commit()

    m = MetricsService()
    m.record_counter('user_messages_total', 999)  # stale / ignored
    dash = m.get_dashboard_data()
    assert dash['counters']['user_messages_total'] == 1


# ── one user turn = exactly one user/user transcript row ───────────────────────


def test_count_exactness_n_turns(db, store):
    """N user turns → COUNT N over (role='user' AND channel='user')."""
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
    assert dash['counters']['user_messages_total'] == 5
