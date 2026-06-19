"""Test data factories — produce realistic row tuples matching actual DB column orders.

All factories return tuples unless noted otherwise. Override any field via keyword argument.
"""

from datetime import datetime, timezone, timedelta

from services.processor_config import ProcessorConfig


# ─── ProcessorConfig test stub ───────────────────────────────────────
# ProcessorConfig's three prompt builders are abstractmethods, so the base
# cannot be instantiated directly.  This concrete stub takes the builders as
# callables (signature (mp) -> str — the pre-refactor field API) and delegates
# to them, letting test helpers inject custom prompt bodies exactly as before.

class StubProcessorConfig(ProcessorConfig):
    def __init__(
        self,
        *,
        build_user_prompt=None,
        build_user_definition=None,
        build_system_prompt=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_b_up", build_user_prompt or (lambda _mp: ""))
        object.__setattr__(self, "_b_ud", build_user_definition or (lambda _mp: ""))
        object.__setattr__(self, "_b_sp", build_system_prompt or (lambda _mp: ""))

    def get_user_prompt(self, mp) -> str:
        return self._b_up(mp)

    def get_user_definition(self, mp) -> str:
        return self._b_ud(mp)

    def get_system_prompt(self, mp) -> str:
        return self._b_sp(mp)


def make_stub_config(
    *,
    always_available=None,
    channel="user",
    role="user",
    policy_channel=None,
):
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
#   window_start, window_end, topic, created_by_session, group_id, is_prompt

def make_scheduled_item(
    item_id="sched-001",
    item_type="reminder",
    message="Test reminder",
    due_at=None,
    recurrence=None,
    window_start=None,
    window_end=None,
    topic=None,
    created_by_session=None,
    group_id=None,
    is_prompt=False,
):
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


def make_episode_row(**overrides):
    row = {**_EPISODE_DEFAULTS, **overrides}
    if row["created_at"] is None:
        row["created_at"] = datetime.now(timezone.utc)
    return row


# ─── providers ───────────────────────────────────────────────────────
# Column order matches: SELECT id, name, platform, model, host, api_key,
#   dimensions, timeout, supports_vision

def make_provider_row(
    provider_id=1,
    name="test-provider",
    platform="ollama",
    model="gemma4:31b",
    host="http://localhost:11434",
    api_key=None,
    dimensions=256,
    timeout=30,
    supports_vision=0,
):
    return (
        provider_id, name, platform, model, host,
        api_key, dimensions, timeout, supports_vision,
    )


