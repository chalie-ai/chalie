"""Test data factories — produce realistic row tuples matching actual DB column orders.

All factories return tuples unless noted otherwise. Override any field via keyword argument.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import cast

from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig


# ─── ProcessorConfig test stub ───────────────────────────────────────
# A minimal concrete ProcessorConfig for tests that need a config instance
# (policy-channel derivation, tool-finder embeddings). Prompt assembly lives in
# PromptService, so the stub carries no prompt-builder surface.

class StubProcessorConfig(ProcessorConfig):
    def __init__(
        self,
        *,
        channel: str,
        role: str,
        policy_channel: PolicyChannel,
        always_available: list[str],
        skip_transcript: bool,
        skip_input_row: bool,
        suppress_history: bool,
        broadcast_to: str | None,
        memory_seed: bool,
    ) -> None:
        super().__init__(
            channel=channel,
            role=role,
            policy_channel=policy_channel,
            always_available=always_available,
            skip_transcript=skip_transcript,
            skip_input_row=skip_input_row,
            suppress_history=suppress_history,
            broadcast_to=broadcast_to,
            memory_seed=memory_seed,
        )


def make_stub_config(
    *,
    always_available: list[str] | None = None,
    channel: str = "user",
    role: str = "user",
    policy_channel: PolicyChannel | None = None,
) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=policy_channel or PolicyChannel.CHAT,
        always_available=list(always_available or []),
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        broadcast_to=None,
        memory_seed=False,
    )


# ─── scheduled_items ─────────────────────────────────────────────────
# Column order matches schema.sql (the real 5-field crontab engine):
#   message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
#   enabled, channel, created_by_session, created_at
# ``id`` is INTEGER PRIMARY KEY AUTOINCREMENT — never supplied on insert, always
# read back via cursor.lastrowid (it also doubles as the schedule's turn_id on
# the 'schedule' channel). All five cron_* columns are TEXT NOT NULL DEFAULT '*'
# crontab expressions (services.cron_schedule.validate_cron / matches); '*' =
# every. start_at is a UTC activation floor.
#
# This is the SINGLE shared seed path for a real scheduled_items row — every
# test that needs a pre-existing row (rather than driving one through the
# ability/REST create path) calls this, so the INSERT's column list can never
# drift out of step with schema.sql in more than one place (the exact bug
# class a hand-rolled per-file INSERT reintroduces).

def insert_scheduled_item(
    db: sqlite3.Connection,
    message: str = "Test reminder",
    start_at: str | None = None,
    cron_minute: str = "*",
    cron_hour: str = "*",
    cron_dom: str = "*",
    cron_month: str = "*",
    cron_dow: str = "*",
    enabled: int = 1,
    channel: str | None = "general",
    created_by_session: str | None = None,
    created_at: str | None = None,
) -> int:
    """Insert a real ``scheduled_items`` row and return its auto-assigned id."""
    start_at = start_at or datetime.now(timezone.utc).isoformat()
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    cur = db.execute(
        """
        INSERT INTO scheduled_items
          (message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
           enabled, channel, created_by_session, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message, start_at, cron_minute, cron_hour, cron_dom, cron_month, cron_dow,
            enabled, channel, created_by_session, created_at,
        ),
    )
    db.commit()
    return cast(int, cur.lastrowid)


# ─── providers ───────────────────────────────────────────────────────
# Column order matches: SELECT id, name, platform, model, host, api_key,
#   dimensions, timeout, supports_vision

def make_provider_row(
    provider_id: int = 1,
    name: str = "test-provider",
    platform: str = "ollama",
    model: str = "gemma4:31b",
    host: str = "http://localhost:11434",
    api_key: str | None = None,
    dimensions: int = 768,
    timeout: int = 30,
    supports_vision: int = 0,
) -> tuple[object, ...]:
    return (
        provider_id, name, platform, model, host,
        api_key, dimensions, timeout, supports_vision,
    )
