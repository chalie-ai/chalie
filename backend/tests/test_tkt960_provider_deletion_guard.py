"""Feature tests: provider deletion guard via routes + service layer.

A provider assigned as main, vision, or delegate cannot be deleted because it
would leave a dangling reference the resolver can no longer satisfy; the admin
must clear or reassign that role first.

All tests use real production stack (real DB, real Flask route, real
ProviderDbService — zero mocks). The guard message is asserted against
PROVIDER_IN_USE_MSG so the service constant and api/providers allowlist never
drift apart.
"""

import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from services.provider_db_service import PROVIDER_IN_USE_MSG, ProviderDbService

if TYPE_CHECKING:
    from flask.testing import FlaskClient

pytestmark = pytest.mark.unit


def _svc() -> ProviderDbService:
    return ProviderDbService()


def _mk(svc: ProviderDbService, name: str, model: str, host: str) -> int:
    return cast(int, cast("dict[str, object]", svc.create_provider(
        {"name": name, "platform": "ollama", "model": model, "host": host, "api_key": ""}
    ))["id"])


def _unwrap_listing(body: "dict[str, object]") -> "list[dict[str, object]]":
    """Assert the success listing envelope shape and return the result array."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("list[dict[str, object]]", body["result"])


def _unwrap_single(body: "dict[str, object]") -> "dict[str, object]":
    """Assert the success single-resource envelope shape and return the result dict."""
    assert body.get("success") is True
    assert "error" not in body
    return cast("dict[str, object]", body["result"])


def _provider_ids(client: "FlaskClient") -> list[object]:
    body = cast("dict[str, object]", client.get("/api/providers/all").get_json())
    return [p["id"] for p in _unwrap_listing(body)]


# ── Through the real HTTP route ───────────────────────────────────────────────


def test_cannot_delete_main_provider_returns_409(authed_client: "tuple[FlaskClient, object, object]") -> None:
    client, _, _ = authed_client

    created = client.post(
        "/api/providers/-1",
        json={"name": "Main", "platform": "ollama", "model": "qwen3",
              "host": "http://127.0.0.1:2"},
    )
    assert created.status_code == 201, created.get_data(as_text=True)
    main_id = _unwrap_single(cast("dict[str, object]", created.get_json()))["id"]

    resp = client.delete(f"/api/providers/{main_id}")
    assert resp.status_code == 409, resp.get_data(as_text=True)
    assert resp.get_json()["error"] == PROVIDER_IN_USE_MSG

    # Downstream: the row survived the refused delete.
    assert main_id in _provider_ids(client)


def test_cannot_delete_delegate_provider_returns_409(authed_client: "tuple[FlaskClient, object, object]") -> None:
    client, _, _ = authed_client

    # First provider → auto-selected as main.
    client.post("/api/providers/-1", json={"name": "Main", "platform": "ollama",
                                     "model": "qwen3", "host": "http://127.0.0.1:2"})
    worker = _unwrap_single(cast("dict[str, object]", client.post(
        "/api/providers/-1",
        json={"name": "Worker", "platform": "ollama", "model": "llama3",
              "host": "http://127.0.0.1:3"},
    ).get_json()))["id"]

    pin = client.post("/api/providers/delegate", json={"provider_id": worker})
    assert pin.status_code == 200, pin.get_data(as_text=True)

    resp = client.delete(f"/api/providers/{worker}")
    assert resp.status_code == 409, resp.get_data(as_text=True)
    assert resp.get_json()["error"] == PROVIDER_IN_USE_MSG
    assert worker in _provider_ids(client)


def test_provider_deletable_after_delegate_pin_cleared(authed_client: "tuple[FlaskClient, object, object]") -> None:
    client, _, _ = authed_client

    client.post("/api/providers/-1", json={"name": "Main", "platform": "ollama",
                                    "model": "qwen3", "host": "http://127.0.0.1:2"})
    worker = _unwrap_single(cast("dict[str, object]", client.post(
        "/api/providers/-1",
        json={"name": "Worker", "platform": "ollama", "model": "llama3",
              "host": "http://127.0.0.1:3"},
    ).get_json()))["id"]
    client.post("/api/providers/delegate", json={"provider_id": worker})

    # Blocked while pinned.
    assert client.delete(f"/api/providers/{worker}").status_code == 409

    # Clear the pin (falls back to the main provider), then delete succeeds.
    cleared = client.post("/api/providers/delegate", json={"provider_id": None})
    assert cleared.status_code == 200, cleared.get_data(as_text=True)

    resp = client.delete(f"/api/providers/{worker}")
    assert resp.status_code == 204, resp.get_data(as_text=True)
    assert worker not in _provider_ids(client)


def test_can_delete_unassigned_provider(authed_client: "tuple[FlaskClient, object, object]") -> None:
    client, _, _ = authed_client

    client.post("/api/providers/-1", json={"name": "Main", "platform": "ollama",
                                    "model": "qwen3", "host": "http://127.0.0.1:2"})
    spare = _unwrap_single(cast("dict[str, object]", client.post(
        "/api/providers/-1",
        json={"name": "Spare", "platform": "ollama", "model": "phi3",
              "host": "http://127.0.0.1:5"},
    ).get_json()))["id"]

    resp = client.delete(f"/api/providers/{spare}")
    assert resp.status_code == 204, resp.get_data(as_text=True)
    assert spare not in _provider_ids(client)


# ── Through the service entry point the route calls directly ───────────────────


def test_cannot_delete_vision_provider(db: sqlite3.Connection) -> None:
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
