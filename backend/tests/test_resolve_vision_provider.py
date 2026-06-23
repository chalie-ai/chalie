"""Proves: a Providers()._resolve(ProviderType.VISION) resolves the DB-configured
vision provider — NOT the globally selected provider."""

import sqlite3
from typing import cast

import pytest

from services.database_service import get_shared_db_service
from services.llm_clients.ollama import OllamaClient
from services.provider_api import ProviderType
from services.provider_db_service import ProviderDbService
from services.providers import Providers

pytestmark = pytest.mark.unit


def test_resolve_chat_returns_global_provider(db: sqlite3.Connection) -> None:
    svc = ProviderDbService(get_shared_db_service())

    global_provider = svc.create_provider(
        {
            "name": "global-text",
            "platform": "ollama",
            "model": "qwen3",
            "host": "http://127.0.0.1:2",
            "api_key": "",
        }
    )
    svc.set_selected_provider(cast(int, cast("dict[str, object]", global_provider)["id"]))

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    client = Providers()._resolve(ProviderType.CHAT)

    # The client is an OllamaClient (ProviderClient subclass) — assert on its fields directly.
    assert cast(OllamaClient, client).model == "qwen3"
    assert cast(OllamaClient, client).host == "http://127.0.0.1:2"


def test_resolve_vision_returns_vision_provider_not_global(db: sqlite3.Connection) -> None:
    svc = ProviderDbService(get_shared_db_service())

    global_provider = svc.create_provider(
        {
            "name": "global-text",
            "platform": "ollama",
            "model": "qwen3",
            "host": "http://127.0.0.1:2",
            "api_key": "",
        }
    )
    svc.set_selected_provider(cast(int, cast("dict[str, object]", global_provider)["id"]))

    vision_provider = svc.create_provider(
        {
            "name": "probe-vision",
            "platform": "ollama",
            "model": "llava",
            "host": "http://127.0.0.1:1",
            "api_key": "",
        }
    )
    vid = cast(int, cast("dict[str, object]", vision_provider)["id"])
    db.execute("UPDATE providers SET supports_vision = 1 WHERE id = ?", (vid,))
    db.commit()
    svc.set_vision_provider(vid)
    assert svc.get_vision_provider() is not None

    from services.provider_cache_service import ProviderCacheService
    ProviderCacheService.invalidate()

    chat_client = cast(OllamaClient, Providers()._resolve(ProviderType.CHAT))
    vision_client = cast(OllamaClient, Providers()._resolve(ProviderType.VISION))

    # Chat resolves the global provider.
    assert chat_client.model == "qwen3"
    assert chat_client.host == "http://127.0.0.1:2"

    # Vision resolves the DB vision provider — completely different model/host.
    assert vision_client.model == "llava"
    assert vision_client.host == "http://127.0.0.1:1"


def test_resolve_vision_raises_when_no_vision_provider_configured(db: sqlite3.Connection) -> None:
    svc = ProviderDbService(get_shared_db_service())
    svc.set_vision_provider(None)
    assert svc.get_vision_provider() is None

    with pytest.raises(RuntimeError, match="no vision provider"):
        Providers()._resolve(ProviderType.VISION)
