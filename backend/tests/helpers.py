"""Test data factories — produce realistic row tuples matching actual DB column orders.

All factories return tuples unless noted otherwise. Override any field via keyword argument.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Callable

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor
    from services.post_turn_hook import PostTurnHook

    class _WithBuilders(ProcessorConfig):
        _b_up: Callable[["MessageProcessor"], str]
        _b_ud: Callable[["MessageProcessor"], str]
        _b_sp: Callable[["MessageProcessor"], str]


# ─── ProcessorConfig test stub ───────────────────────────────────────
# ProcessorConfig's three prompt builders are abstractmethods, so the base
# cannot be instantiated directly.  This concrete stub takes the builders as
# callables (signature (mp) -> str — the pre-refactor field API) and delegates
# to them, letting test helpers inject custom prompt bodies exactly as before.

class StubProcessorConfig(ProcessorConfig):
    def __init__(
        self,
        *,
        build_user_prompt: Callable[["MessageProcessor"], str] | None = None,
        build_user_definition: Callable[["MessageProcessor"], str] | None = None,
        build_system_prompt: Callable[["MessageProcessor"], str] | None = None,
        channel: str,
        role: str,
        policy_channel: ProcessorConfig.PolicyChannel,
        always_available: list[str],
        skip_transcript: bool,
        skip_input_row: bool,
        suppress_history: bool,
        broadcast_to: str | None,
        memory_seed: bool,
        post_turn_hooks: tuple["PostTurnHook", ...] = (),
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
            post_turn_hooks=post_turn_hooks,
        )
        object.__setattr__(self, "_b_up", build_user_prompt or (lambda _mp: ""))
        object.__setattr__(self, "_b_ud", build_user_definition or (lambda _mp: ""))
        object.__setattr__(self, "_b_sp", build_system_prompt or (lambda _mp: ""))

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        from typing import cast
        return cast("_WithBuilders", self)._b_up(mp)

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        from typing import cast
        return cast("_WithBuilders", self)._b_ud(mp)

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        from typing import cast
        return cast("_WithBuilders", self)._b_sp(mp)


def make_stub_config(
    *,
    always_available: list[str] | None = None,
    channel: str = "user",
    role: str = "user",
    policy_channel: ProcessorConfig.PolicyChannel | None = None,
) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=policy_channel or ProcessorConfig.PolicyChannel.CHAT,
        always_available=list(always_available or []),
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        broadcast_to=None,
        memory_seed=False,
    )


# ─── scheduled_items ─────────────────────────────────────────────────
# Column order matches: SELECT id, item_type, message, due_at, recurrence,
#   topic, created_by_session, group_id, is_prompt

def make_scheduled_item(
    item_id: str = "sched-001",
    item_type: str = "reminder",
    message: str = "Test reminder",
    due_at: str | None = None,
    recurrence: str | None = None,
    topic: str | None = None,
    created_by_session: str | None = None,
    group_id: str | None = None,
    is_prompt: bool = False,
) -> tuple[object, ...]:
    due_at = due_at or (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    return (
        item_id, item_type, message, due_at, recurrence,
        topic, created_by_session, group_id,
        is_prompt,
    )


# ─── episodes ────────────────────────────────────────────────────────
# Used by episodic_service._hybrid_retrieve() which returns dicts,
# but the raw query returns tuples.  This factory returns a dict matching
# the service's output format (since the service converts internally).

_EPISODE_DEFAULTS: dict[str, object] = {
    "id": "ep-001",
    "gist": "Weather conversation about Malta",
    "salience": 5,
    "channel": "weather",
    "created_at": None,
    "last_accessed_at": None,
    "transcript_ids": "[]",
    "transcript_id_start": None,
    "transcript_id_end": None,
    "consolidated_from": "[]",
    "consolidated_into": None,
    "retrieval_weight": 1.0,
}


def make_episode_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {**_EPISODE_DEFAULTS, **overrides}
    if row["created_at"] is None:
        row["created_at"] = datetime.now(timezone.utc)
    return row


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
    dimensions: int = 256,
    timeout: int = 30,
    supports_vision: int = 0,
) -> tuple[object, ...]:
    return (
        provider_id, name, platform, model, host,
        api_key, dimensions, timeout, supports_vision,
    )
