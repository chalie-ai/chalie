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
