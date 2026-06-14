"""Feature tests (TKT-960): the provider deletion guard.

A provider currently assigned as the main (selected), vision, or delegate
provider cannot be deleted — removing it would leave a dangling reference the
resolver can no longer satisfy. The admin must clear or reassign that role first.

Proves, against the real production stack (real DB, real Flask route, real
ProviderDbService, the shared create_provider factory — zero mocks):

  Through the real HTTP route (DELETE /providers/<id>)
    1. The active main provider cannot be deleted → HTTP 409 + the exact guard
       message, and the row survives.
    2. A provider pinned as the delegate cannot be deleted → HTTP 409, row survives.
    3. Once the delegate pin is cleared, that now role-free provider is deletable
       → HTTP 200, row gone.
    4. A provider holding no role is deletable → HTTP 200, row gone.

  Through the service entry point the route calls directly
    5. A provider pinned as the vision provider cannot be deleted → ValueError
       carrying the exact guard message, row survives.

The guard message is asserted byte-for-byte against PROVIDER_IN_USE_MSG so the
service constant and the api/providers allowlist can never silently drift apart.
"""

import pytest

from services.database_service import get_shared_db_service
from services.provider_db_service import PROVIDER_IN_USE_MSG, ProviderDbService

pytestmark = pytest.mark.unit


def _svc() -> ProviderDbService:
    return ProviderDbService(get_shared_db_service())


def _mk(svc, name, model, host):
    """Create a provider via the production factory; return its id."""
    return svc.create_provider(
        {"name": name, "platform": "ollama", "model": model, "host": host, "api_key": ""}
    )["id"]


def _provider_ids(client):
    return [p["id"] for p in client.get("/providers").get_json()["providers"]]


# ── Through the real HTTP route ───────────────────────────────────────────────


def test_cannot_delete_main_provider_returns_409(authed_client):
    """The first provider an admin adds is auto-selected as main; deleting it via
    the Providers tab is refused with HTTP 409 and the guard message, and the
    provider is still listed afterwards."""
    client, _, _ = authed_client

    created = client.post(
        "/providers",
        json={"name": "Main", "platform": "ollama", "model": "qwen3",
              "host": "http://127.0.0.1:2"},
    )
    assert created.status_code == 201, created.get_data(as_text=True)
    main_id = created.get_json()["provider"]["id"]

    resp = client.delete(f"/providers/{main_id}")
    assert resp.status_code == 409, resp.get_data(as_text=True)
    assert resp.get_json()["error"] == PROVIDER_IN_USE_MSG

    # Downstream: the row survived the refused delete.
    assert main_id in _provider_ids(client)


def test_cannot_delete_delegate_provider_returns_409(authed_client):
    """A provider pinned as the delegate (but NOT the main) cannot be deleted."""
    client, _, _ = authed_client

    # First provider → auto-selected as main.
    client.post("/providers", json={"name": "Main", "platform": "ollama",
                                     "model": "qwen3", "host": "http://127.0.0.1:2"})
    worker = client.post(
        "/providers",
        json={"name": "Worker", "platform": "ollama", "model": "llama3",
              "host": "http://127.0.0.1:3"},
    ).get_json()["provider"]["id"]

    pin = client.put("/providers/delegate", json={"provider_id": worker})
    assert pin.status_code == 200, pin.get_data(as_text=True)

    resp = client.delete(f"/providers/{worker}")
    assert resp.status_code == 409, resp.get_data(as_text=True)
    assert resp.get_json()["error"] == PROVIDER_IN_USE_MSG
    assert worker in _provider_ids(client)


def test_provider_deletable_after_delegate_pin_cleared(authed_client):
    """Clearing the delegate pin releases the role; the provider is then deletable."""
    client, _, _ = authed_client

    client.post("/providers", json={"name": "Main", "platform": "ollama",
                                    "model": "qwen3", "host": "http://127.0.0.1:2"})
    worker = client.post(
        "/providers",
        json={"name": "Worker", "platform": "ollama", "model": "llama3",
              "host": "http://127.0.0.1:3"},
    ).get_json()["provider"]["id"]
    client.put("/providers/delegate", json={"provider_id": worker})

    # Blocked while pinned.
    assert client.delete(f"/providers/{worker}").status_code == 409

    # Clear the pin (falls back to the main provider), then delete succeeds.
    cleared = client.put("/providers/delegate", json={"provider_id": None})
    assert cleared.status_code == 200, cleared.get_data(as_text=True)

    resp = client.delete(f"/providers/{worker}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert worker not in _provider_ids(client)


def test_can_delete_unassigned_provider(authed_client):
    """A provider holding no role (not main, vision, or delegate) is deletable."""
    client, _, _ = authed_client

    client.post("/providers", json={"name": "Main", "platform": "ollama",
                                    "model": "qwen3", "host": "http://127.0.0.1:2"})
    spare = client.post(
        "/providers",
        json={"name": "Spare", "platform": "ollama", "model": "phi3",
              "host": "http://127.0.0.1:5"},
    ).get_json()["provider"]["id"]

    resp = client.delete(f"/providers/{spare}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert spare not in _provider_ids(client)


# ── Through the service entry point the route calls directly ───────────────────


def test_cannot_delete_vision_provider(db):
    """A provider pinned as the vision provider cannot be deleted.

    delete_provider is the exact method the DELETE route invokes; it raises
    ValueError carrying the guard message (the route maps that to HTTP 409). The
    vision role is isolated here: a separate provider is the selected main, so
    only the vision pin protects the provider under test.
    """
    svc = _svc()
    main = _mk(svc, "main-text", "qwen3", "http://127.0.0.1:2")
    svc.set_selected_provider(main)
    vis = _mk(svc, "vision-pin", "llava", "http://127.0.0.1:4")
    svc.set_vision_provider(vis)

    assert svc._provider_roles(vis) == ["vision"]

    with pytest.raises(ValueError) as exc_info:
        svc.delete_provider(vis)
    assert str(exc_info.value) == PROVIDER_IN_USE_MSG

    # Downstream: the row survived the refused delete.
    assert svc.get_provider_by_id(vis) is not None
