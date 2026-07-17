import sqlite3
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from services.provider_db_service import ProviderDbService

pytestmark = pytest.mark.unit


def _svc(db: sqlite3.Connection) -> "ProviderDbService":
    from services.provider_db_service import ProviderDbService
    return ProviderDbService()


def test_create_sets_supports_vision_when_probe_passes(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=True) as pp:
        p = svc.create_provider(
            {"name": "v", "platform": "ollama", "model": "llava"}
        )
    assert pp.called
    assert cast("dict[str, object]", p)["supports_vision"] is True


def test_create_clears_supports_vision_when_probe_fails(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=False):
        p = svc.create_provider(
            {"name": "n", "platform": "ollama", "model": "plain"}
        )
    assert cast("dict[str, object]", p)["supports_vision"] is False


def test_create_keyless_key_requiring_provider_skips_probe(db: sqlite3.Connection) -> None:
    # Key-requiring platforms (openai/anthropic/gemini/openai_compatible) with no
    # api_key cannot be probed — probe is skipped to avoid guaranteed-to-fail network call.
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=True) as pp:
        p = svc.create_provider(
            {"name": "keyless-openai", "platform": "openai", "model": "gpt-4o"}
        )
    assert not pp.called
    assert cast("dict[str, object]", p)["supports_vision"] is False


def _make(db: sqlite3.Connection, svc: "ProviderDbService", name: str = "p", model: str = "m", vision: int = 0) -> "dict[str, object]":
    with patch("services.vision_probe.probe_provider", return_value=bool(vision)):
        return cast("dict[str, object]", svc.create_provider(
            {"name": name, "platform": "ollama", "model": model}))


def test_update_name_only_does_not_probe(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    p = _make(db, svc, name="orig", vision=1)
    with patch("services.vision_probe.probe_provider", return_value=False) as pp:
        svc.update_provider(cast(int, p["id"]), {"name": "renamed"})
    assert not pp.called
    # supports_vision untouched (still True from create)
    assert cast("dict[str, object]", svc.get_provider_by_id(cast(int, p["id"])))["supports_vision"] is True


def test_update_model_reprobes(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    p = _make(db, svc, model="old", vision=1)
    with patch("services.vision_probe.probe_provider", return_value=False) as pp:
        svc.update_provider(cast(int, p["id"]), {"model": "new"})
    assert pp.called
    assert cast("dict[str, object]", svc.get_provider_by_id(cast(int, p["id"])))["supports_vision"] is False


def test_delete_is_permanent_and_frees_the_name(db: sqlite3.Connection) -> None:
    svc = _svc(db)
    _make(db, svc, name="anchor-main", vision=0)  # auto-selected as main
    p = _make(db, svc, name="reusable", vision=0)
    assert svc.delete_provider(cast(int, p["id"])) is True
    # row is gone
    assert svc.get_provider_by_id(cast(int, p["id"])) is None
    assert all(row["id"] != p["id"] for row in svc.get_all_providers())
    # the freed name can be reused without a UNIQUE collision
    p2 = _make(db, svc, name="reusable", vision=0)
    assert p2["id"] != p["id"]
