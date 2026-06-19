"""Feature tests for the unified provider API contract.

Covers the NEW behaviours introduced by the refactor:
  - Providers().send(dto) raises RequestOverCapError on pre-flight over-cap.
  - Providers().measure(dto) returns a usable token estimate.
  - MessageProcessor._build_send_dto() derives the correct ProviderType for
    CHAT vs VISION configs, carrying the provider role on the DTO's type field.
  - send_image_with_config builds a VISION DTO and returns None (not raises)
    when the provider is unreachable (defensive probe path).

All tests drive the real production stack — real DB, real provider clients,
real DTO construction — zero mocks.
"""

import base64

import pytest

from services.provider_api import (
    ProviderApiRequest,
    ProviderType,
    ThinkingLevel,
    RequestOverCapError,
)

pytestmark = pytest.mark.unit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _png_1x1() -> bytes:
    return base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


def _seed_ollama_with_window(db, max_tokens):
    """OllamaClient.estimate_request_tokens works without a live model (pure heuristic /
    json serialisation), so over-cap tests never hit the network.
    """
    cur = db.execute(
        "INSERT INTO providers (name, platform, model, host, max_tokens) "
        "VALUES ('cap-test', 'ollama', 'cap-model', 'http://localhost:11434', ?)",
        (max_tokens,),
    )
    pid = cur.lastrowid
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('selected_provider_id', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(pid),),
    )
    db.commit()
    return pid


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_send_raises_request_over_cap_for_oversized_dto(db):
    """Replaces the old OVER_CAP sentinel: callers now catch RequestOverCapError instead
    of comparing `response is OVER_CAP`. The pre-flight check fires before any network call.

    Providers.send() derives the window from client.get_context_limit() (falls back to 8192
    for an offline Ollama). The DB-declared max_tokens feeds Providers.get_context_limit()
    (the facade) but not the send path directly - we assert structural invariants on the
    exception, not a specific window value.
    """
    from services.providers import Providers
    from services.provider_cache_service import ProviderCacheService

    _seed_ollama_with_window(db, 20000)
    ProviderCacheService.invalidate()

    # Build a DTO large enough to exceed even a small window:
    # ~400 words × 1.3 ≈ 520 tokens/msg × 30 messages ≈ 15.6k tokens, which
    # exceeds any sane window cap (8192 default → cap ≈ 192).
    big = " ".join(f"w{j}" for j in range(400))
    big_messages = [
        {"role": "user", "content": f"msg{i} {big}"}
        for i in range(30)
    ]
    dto = ProviderApiRequest(
        system="You are Chalie.",
        messages=big_messages,
        type=ProviderType.CHAT,
        tools=None,
        thinking_mode=ThinkingLevel.LOW,
    )

    try:
        with pytest.raises(RequestOverCapError) as exc_info:
            Providers().send(dto)

        err = exc_info.value
        # Structural invariants: the error must carry window, measured, cap
        # with the relationship measured >= cap.
        assert err.window > 0
        assert err.measured > 0
        assert err.cap > 0
        assert err.measured >= err.cap, (
            f"measured ({err.measured}) must be >= cap ({err.cap})"
        )
    finally:
        ProviderCacheService.invalidate()


def test_measure_returns_positive_integer_for_real_dto(db):
    """Used by ChatHistoryCompactor._fit_compaction_input to decide how much history to drop."""
    from services.providers import Providers
    from services.provider_cache_service import ProviderCacheService

    _seed_ollama_with_window(db, 200000)
    ProviderCacheService.invalidate()

    dto = ProviderApiRequest(
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        type=ProviderType.CHAT,
        tools=None,
        thinking_mode=ThinkingLevel.LOW,
    )

    try:
        measured = Providers().measure(dto)
        assert isinstance(measured, int)
        assert measured > 0, "measure() must return a positive token estimate"
        # A 7-word system + 5-word message should be well under 1000 tokens.
        assert measured < 1000, f"measure() returned suspiciously large value: {measured}"
    finally:
        ProviderCacheService.invalidate()


def test_build_send_dto_type_chat_for_user_config(db):
    """The config's role is encoded in the DTO's type field, derived from the config's
    provider flags rather than read directly at the send call site.
    """
    from services.message_processor import MessageProcessor
    from configs.channels.user import UserConfig
    from services.provider_cache_service import ProviderCacheService

    _seed_ollama_with_window(db, 200000)
    ProviderCacheService.invalidate()

    try:
        mp = object.__new__(MessageProcessor)
        MessageProcessor.__init__(mp, "tell me a joke", None)
        mp.config = UserConfig()
        mp.thinking_level = "low"

        dto = mp._build_send_dto()

        assert dto.type == ProviderType.CHAT, (
            "UserConfig must produce a CHAT-type DTO; got: %s" % dto.type
        )
        assert isinstance(dto.system, str) and dto.system
        assert isinstance(dto.messages, list) and len(dto.messages) > 0
        assert dto.messages[0]["role"] == "user"
    finally:
        ProviderCacheService.invalidate()


def test_build_send_dto_type_vision_for_vision_config(db, tmp_path):
    """The DTO type drives provider resolution in Providers.send(dto), derived from
    the config's uses_vision_provider flag.
    """
    from services.message_processor import MessageProcessor
    from configs.channels.vision import VisionConfig
    from services.processor_config import ProcessorConfig

    img = tmp_path / "test.png"
    img.write_bytes(_png_1x1())

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(
        mp, "what is this image?",
        {"image_path": str(img), "mime_type": "image/png"},
    )
    mp.config = VisionConfig(ProcessorConfig.PolicyChannel.CHAT)
    mp.thinking_level = "low"

    dto = mp._build_send_dto()

    assert dto.type == ProviderType.VISION, (
        "VisionConfig must produce a VISION-type DTO; got: %s" % dto.type
    )
    # The image attachment must survive into the messages array.
    assert "image" in dto.messages[0], (
        "VisionConfig DTO messages[0] must carry 'image' dict; got: %s" % dto.messages[0].keys()
    )
    assert dto.messages[0]["image"]["mime_type"] == "image/png"


def test_send_image_with_config_returns_none_for_unreachable_provider():
    """The vision probe is a best-effort path - an offline host gets a network error,
    which send_image_with_config catches and maps to None.
    """
    from services.vision_service import send_image_with_config, build_vision_config

    # A syntactically valid but unreachable Ollama config (closed port 1).
    provider_row = {
        "platform": "ollama",
        "model": "llava",
        "host": "http://127.0.0.1:1",
        "api_key": "",
    }
    config = build_vision_config(provider_row)
    result = send_image_with_config(config, _png_1x1(), "describe this", "image/png")

    # Must return None, not raise — the function is a fully defensive probe.
    assert result is None, (
        "send_image_with_config must return None on unreachable provider; "
        f"got: {result!r}"
    )
