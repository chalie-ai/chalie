"""Unit tests for probe-on-save wiring in ProviderDbService (probe mocked)."""
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _svc(db):
    import services.database_service as _db_mod
    from services.provider_db_service import ProviderDbService
    return ProviderDbService(_db_mod._shared_db_service)


def test_create_sets_supports_vision_when_probe_passes(db):
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=True) as pp:
        p = svc.create_provider(
            {"name": "v", "platform": "ollama", "model": "llava"}
        )
    assert pp.called
    assert p["supports_vision"] is True


def test_create_clears_supports_vision_when_probe_fails(db):
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=False):
        p = svc.create_provider(
            {"name": "n", "platform": "ollama", "model": "plain"}
        )
    assert p["supports_vision"] is False


def test_infer_vision_support_is_gone():
    import services.provider_db_service as mod
    assert not hasattr(mod, "_infer_vision_support")


def test_create_keyless_key_requiring_provider_skips_probe(db):
    """A key-requiring platform (openai/anthropic/gemini/openai_compatible) with
    no api_key cannot be probed: the probe is skipped (no guaranteed-to-fail
    network call) and supports_vision defaults to 0. Mirrors the update guard."""
    svc = _svc(db)
    with patch("services.vision_probe.probe_provider", return_value=True) as pp:
        p = svc.create_provider(
            {"name": "keyless-openai", "platform": "openai", "model": "gpt-4o"}
        )
    assert not pp.called
    assert p["supports_vision"] is False


def _make(db, svc, name="p", model="m", vision=0):
    with patch("services.vision_probe.probe_provider", return_value=bool(vision)):
        return svc.create_provider(
            {"name": name, "platform": "ollama", "model": model})


def test_update_name_only_does_not_probe(db):
    svc = _svc(db)
    p = _make(db, svc, name="orig", vision=1)
    with patch("services.vision_probe.probe_provider", return_value=False) as pp:
        svc.update_provider(p["id"], {"name": "renamed"})
    assert not pp.called
    # supports_vision untouched (still True from create)
    assert svc.get_provider_by_id(p["id"])["supports_vision"] is True


def test_update_model_reprobes(db):
    svc = _svc(db)
    p = _make(db, svc, model="old", vision=1)
    with patch("services.vision_probe.probe_provider", return_value=False) as pp:
        svc.update_provider(p["id"], {"model": "new"})
    assert pp.called
    assert svc.get_provider_by_id(p["id"])["supports_vision"] is False


def test_delete_is_permanent_and_frees_the_name(db):
    """Delete physically removes the row — no soft-delete flag survives — so the
    name is immediately reusable (regression: soft-deleted rows held the UNIQUE
    name and caused 'A provider with that name already exists' on re-create).

    TKT-960 spec change: a provider assigned as the main/vision/delegate provider
    can no longer be deleted (the deletion guard). The FIRST provider created is
    auto-selected as main, so an anchor main is created first to keep the
    provider under test role-free and freely deletable — preserving this test's
    original intent (permanent delete frees the unique name).
    """
    svc = _svc(db)
    _make(db, svc, name="anchor-main", vision=0)  # auto-selected as main
    p = _make(db, svc, name="reusable", vision=0)
    assert svc.delete_provider(p["id"]) is True
    # row is gone
    assert svc.get_provider_by_id(p["id"]) is None
    assert all(row["id"] != p["id"] for row in svc.get_all_providers())
    # the freed name can be reused without a UNIQUE collision
    p2 = _make(db, svc, name="reusable", vision=0)
    assert p2["id"] != p["id"]
