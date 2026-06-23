"""Feature tests: the Delegate Provider — a dedicated provider for
subagent (delegate) turns (web_search, web_browse, …), independent of the main
chat provider, defaulting to "use main provider".

Proves, against the real production stack (real DB, real Providers facade, real
ProcessorConfig subclasses, real client factory — zero mocks):

  Routing (Providers._resolve)
    1. DELEGATE with nothing pinned resolves the SELECTED (main) provider — the
       "use main provider" default.
    2. DELEGATE with an explicit pin resolves the PINNED provider, NOT the
       selected one — proving the two pools are genuinely distinct.
    3. DELEGATE with neither a pin nor a selected provider raises loudly — never
       a silent fallback.

  DB layer (ProviderDbService)
    4. Status round-trip: default → 'auto' (main), explicit pin → 'explicit',
       clear → 'auto' (main again, NOT the just-pinned provider). There is no
       'Disabled'/'none' clear state for delegate, unlike vision.

  DTO derivation (MessageProcessor._build_send_dto)
    5. WebSearchConfig produces a DELEGATE-type DTO.
    6. WebBrowseConfig produces a DELEGATE-type DTO.
    7. VisionConfig still produces a VISION-type DTO — vision precedence holds
       even though `delegate:vision` is itself a delegate channel.

The provider clients are built with unreachable hosts (127.0.0.1:<closed port>);
the factory constructs a client without touching the network, and we assert on
client.model / client.host to tell the providers apart — no LLM call is made.
"""

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from services.database_service import get_shared_db_service
from services.provider_api import ProviderType
from services.provider_db_service import ProviderDbService
from services.providers import Providers

if TYPE_CHECKING:
    from services.llm_clients.ollama import OllamaClient
    from services.message_processor import MessageProcessor

pytestmark = pytest.mark.unit

_CHAT = None  # set lazily inside DTO tests to avoid an import at collection time


def _svc() -> ProviderDbService:
    return ProviderDbService(get_shared_db_service())


def _mk(svc: ProviderDbService, name: str, model: str, host: str) -> int:
    """Host is an unreachable loopback port to skip the on-create vision probe."""
    return cast(int, cast(dict[str, object], svc.create_provider(
        {"name": name, "platform": "ollama", "model": model, "host": host, "api_key": ""}
    ))["id"])


# ── Routing: Providers._resolve(DELEGATE) ─────────────────────────────────────


def test_resolve_delegate_falls_back_to_selected_when_unpinned(db: sqlite3.Connection) -> None:
    svc = _svc()
    main = _mk(svc, "main-text", "qwen3", "http://127.0.0.1:2")
    svc.set_selected_provider(main)

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    delegate_client = cast("OllamaClient", Providers()._resolve(ProviderType.DELEGATE))

    assert delegate_client.model == "qwen3"
    assert delegate_client.host == "http://127.0.0.1:2"


def test_resolve_delegate_returns_pinned_provider_not_selected(db: sqlite3.Connection) -> None:
    svc = _svc()
    main = _mk(svc, "main-text", "qwen3", "http://127.0.0.1:2")
    svc.set_selected_provider(main)

    worker = _mk(svc, "delegate-worker", "llama3", "http://127.0.0.1:3")
    svc.set_delegate_provider(worker)

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    chat_client = cast("OllamaClient", Providers()._resolve(ProviderType.CHAT))
    delegate_client = cast("OllamaClient", Providers()._resolve(ProviderType.DELEGATE))

    # Chat resolves the selected (main) provider.
    assert chat_client.model == "qwen3"
    assert chat_client.host == "http://127.0.0.1:2"

    # Delegate resolves the pinned worker — different model AND host.
    assert delegate_client.model == "llama3"
    assert delegate_client.host == "http://127.0.0.1:3"


def test_resolve_delegate_raises_when_no_provider_configured(db: sqlite3.Connection) -> None:
    svc = _svc()
    svc.set_delegate_provider(None)
    svc.set_selected_provider(0)  # 0 → get_provider_by_id miss → no selected

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    assert svc.get_delegate_provider() is None

    with pytest.raises(RuntimeError, match="delegate"):
        Providers()._resolve(ProviderType.DELEGATE)


def test_send_routes_delegate_turn_through_delegate_client(db: sqlite3.Connection) -> None:
    """Over-cap guard inside send() fires with DELEGATE provider's model ('llama3'), not
    chat provider's — proving send() resolves the correct pool."""
    from configs.channels.web_search import WebSearchConfig
    from services.provider_api import RequestOverCapError
    from services.provider_cache_service import ProviderCacheService

    svc = _svc()
    main = _mk(svc, "main-text", "qwen3", "http://127.0.0.1:2")
    svc.set_selected_provider(main)
    worker = _mk(svc, "delegate-worker", "llama3", "http://127.0.0.1:3")
    svc.set_delegate_provider(worker)
    ProviderCacheService.invalidate()

    # A real delegate DTO, oversized so the pre-flight cap fires deterministically
    # (offline ollama window 8192 → cap ≈ 192 tokens).
    big_query = " ".join(f"w{j}" for j in range(2000))
    mp = _delegate_mp(WebSearchConfig, big_query)
    dto = mp._build_send_dto()
    assert dto.type == ProviderType.DELEGATE

    try:
        with pytest.raises(RequestOverCapError) as exc_info:
            Providers().send(dto)
        assert exc_info.value.model == "llama3", (
            "send() must resolve the DELEGATE provider (worker); got model %r"
            % exc_info.value.model
        )
    finally:
        ProviderCacheService.invalidate()


# ── DB layer: ProviderDbService delegate status round-trip ─────────────────────


def test_delegate_status_default_explicit_then_clear_back_to_main(db: sqlite3.Connection) -> None:
    """get_delegate_provider_status() walks the full admin lifecycle:

      • default (nothing pinned) → source 'auto', provider == main
      • explicit pin             → source 'explicit', provider == worker
      • clear (None)             → source 'auto', provider == main (NOT worker)

    The clear path is the key invariant: delegate has no 'Disabled' state, so
    clearing falls back to the main provider — never the last-pinned one.
    """
    svc = _svc()
    main = _mk(svc, "main-text", "qwen3", "http://127.0.0.1:2")
    svc.set_selected_provider(main)
    worker = _mk(svc, "delegate-worker", "llama3", "http://127.0.0.1:3")

    # Default — auto-resolves to the selected main provider.
    status = svc.get_delegate_provider_status()
    assert status["source"] == "auto"
    assert cast(dict[str, object], status["provider"])["id"] == main

    # Explicit pin.
    svc.set_delegate_provider(worker)
    status = svc.get_delegate_provider_status()
    assert status["source"] == "explicit"
    assert cast(dict[str, object], status["provider"])["id"] == worker

    # Clear → falls back to main, NOT the worker just pinned.
    svc.set_delegate_provider(None)
    status = svc.get_delegate_provider_status()
    assert status["source"] == "auto"
    assert cast(dict[str, object], status["provider"])["id"] == main


# ── DTO derivation: MessageProcessor._build_send_dto ───────────────────────────


def _delegate_mp(config_cls: type, raw_input: str) -> "MessageProcessor":
    """Build a real MessageProcessor bound to a delegate config, ready for
    _build_send_dto().

    Mirrors the DTO-type derivation pattern in test_provider_api_contract.py:
    object.__new__ + __init__ + config + thinking_level, then _build_send_dto().
    The delegate configs' get_user_prompt calls render_trail(mp), which is fully
    defensive (returns "" when no act-trail/uid exists), so no _setup() is needed
    to derive the DTO's provider type.
    """
    from services.message_processor import MessageProcessor
    from services.processor_config import ProcessorConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, raw_input, {})
    mp.config = config_cls(ProcessorConfig.PolicyChannel.CHAT)
    mp.thinking_level = "low"
    return mp


def test_build_send_dto_type_delegate_for_web_search_config(db: sqlite3.Connection) -> None:
    """WebSearchConfig → _build_send_dto() yields a DELEGATE-type DTO, so the
    web_search subagent runs on the delegate provider."""
    from configs.channels.web_search import WebSearchConfig

    mp = _delegate_mp(WebSearchConfig, "what is the capital of Malta?")
    dto = mp._build_send_dto()

    assert dto.type == ProviderType.DELEGATE, (
        "WebSearchConfig must produce a DELEGATE-type DTO; got: %s" % dto.type
    )


def test_build_send_dto_type_delegate_for_web_browse_config(db: sqlite3.Connection) -> None:
    """WebBrowseConfig → _build_send_dto() yields a DELEGATE-type DTO, so the
    web_browse subagent runs on the delegate provider."""
    from configs.channels.web_browse import WebBrowseConfig

    mp = _delegate_mp(WebBrowseConfig, "find the cheapest flight to Malta")
    dto = mp._build_send_dto()

    assert dto.type == ProviderType.DELEGATE, (
        "WebBrowseConfig must produce a DELEGATE-type DTO; got: %s" % dto.type
    )


def test_build_send_dto_vision_precedence_over_delegate(db: sqlite3.Connection) -> None:
    """VisionConfig (`delegate:vision`) still produces a VISION-type DTO — vision
    provider precedence holds even though it is technically a delegate channel.
    Guards the vision > delegate > chat ordering in _build_send_dto()."""
    from configs.channels.vision import VisionConfig
    from services.message_processor import MessageProcessor
    from services.processor_config import ProcessorConfig

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, "describe this", {})
    mp.config = VisionConfig(ProcessorConfig.PolicyChannel.CHAT)
    mp.thinking_level = "low"

    dto = mp._build_send_dto()

    assert dto.type == ProviderType.VISION, (
        "VisionConfig must keep VISION precedence; got: %s" % dto.type
    )
