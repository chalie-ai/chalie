"""Test data factories — produce realistic row tuples matching actual DB column orders.

All factories return tuples unless noted otherwise. Override any field via keyword argument.
"""

from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Optional

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from typing import Unpack, TypedDict
    from services.post_turn_hook import PostTurnHook

    class _PCFields(TypedDict, total=False):
        channel: str
        role: str
        policy_channel: ProcessorConfig.PolicyChannel
        always_available: list[str]
        discoverable: list[str]
        blocked: frozenset[str]
        max_iterations: Optional[int]
        skip_transcript: bool
        skip_input_row: bool
        suppress_history: bool
        broadcast_to: Optional[str]
        memory_seed: bool
        post_turn_hooks: "tuple[PostTurnHook, ...]"


# ─── ProcessorConfig test stub ───────────────────────────────────────
# ProcessorConfig's three prompt builders are abstractmethods, so the base
# cannot be instantiated directly.  This concrete stub takes the builders as
# callables (signature (mp) -> str — the pre-refactor field API) and delegates
# to them, letting test helpers inject custom prompt bodies exactly as before.

class StubProcessorConfig(ProcessorConfig):
    _b_up: Callable[[object], str]
    _b_ud: Callable[[object], str]
    _b_sp: Callable[[object], str]

    def __init__(
        self,
        *,
        build_user_prompt: Optional[Callable[[object], str]] = None,
        build_user_definition: Optional[Callable[[object], str]] = None,
        build_system_prompt: Optional[Callable[[object], str]] = None,
        **kwargs: "Unpack[_PCFields]",
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_b_up", build_user_prompt or (lambda _mp: ""))
        object.__setattr__(self, "_b_ud", build_user_definition or (lambda _mp: ""))
        object.__setattr__(self, "_b_sp", build_system_prompt or (lambda _mp: ""))

    def get_user_prompt(self, mp: object) -> str:
        return self._b_up(mp)

    def get_user_definition(self, mp: object) -> str:
        return self._b_ud(mp)

    def get_system_prompt(self, mp: object) -> str:
        return self._b_sp(mp)


def make_stub_config(
    *,
    discoverable: Optional[list[str]] = None,
    blocked: frozenset[str] = frozenset(),
    always_available: Optional[list[str]] = None,
    channel: str = "user",
    role: str = "user",
    policy_channel: Optional[ProcessorConfig.PolicyChannel] = None,
) -> StubProcessorConfig:
    return StubProcessorConfig(
        channel=channel,
        role=role,
        policy_channel=policy_channel or ProcessorConfig.PolicyChannel.CHAT,
        always_available=list(always_available or []),
        discoverable=list(discoverable or []),
        blocked=frozenset(blocked),
        max_iterations=None,
        skip_transcript=False,
        skip_input_row=False,
        suppress_history=False,
        broadcast_to=None,
        memory_seed=False,
    )


# ─── scheduled_items ─────────────────────────────────────────────────
# Column order matches: SELECT id, item_type, message, due_at, recurrence,
#   window_start, window_end, topic, created_by_session, group_id, is_prompt

def make_scheduled_item(
    item_id: str = "sched-001",
    item_type: str = "reminder",
    message: str = "Test reminder",
    due_at: Optional[str] = None,
    recurrence: Optional[str] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    topic: Optional[str] = None,
    created_by_session: Optional[str] = None,
    group_id: Optional[str] = None,
    is_prompt: bool = False,
) -> tuple[object, ...]:
    due_at = due_at or (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    return (
        item_id, item_type, message, due_at, recurrence,
        window_start, window_end, topic, created_by_session, group_id,
        is_prompt,
    )


# ─── episodes ────────────────────────────────────────────────────────
# Used by episodic_service._hybrid_retrieve() which returns dicts,
# but the raw query returns tuples.  This factory returns a dict matching
# the service's output format (since the service converts internally).

_EPISODE_DEFAULTS = {
    "id": "ep-001",
    "gist": "Weather conversation about Malta",
    "salience": 5,
    "channel": "weather",
    "created_at": None,
    "last_accessed_at": None,
    "emotional_valence": None,
    "emotional_arousal": None,
    "transcript_ids": "[]",
    "transcript_id_start": None,
    "transcript_id_end": None,
    "consolidated_from": "[]",
    "consolidated_into": None,
    "storage_strength": 1.0,
    "retrieval_weight": 1.0,
}


def make_episode_row(**overrides: object) -> dict[str, object]:
    row = {**_EPISODE_DEFAULTS, **overrides}
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
    api_key: Optional[str] = None,
    dimensions: int = 256,
    timeout: int = 30,
    supports_vision: int = 0,
) -> tuple[object, ...]:
    return (
        provider_id, name, platform, model, host,
        api_key, dimensions, timeout, supports_vision,
    )


