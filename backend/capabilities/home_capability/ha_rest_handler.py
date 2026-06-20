"""Stateless REST client for the Home Assistant API.

Every function accepts connection parameters (url, token, verify_ssl) rather
than storing them -- the caller (HomeCapability) owns credential lifecycle.
"""

import logging
from typing import cast

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _get(url: str, path: str, token: str, verify_ssl: bool) -> requests.Response:
    resp = requests.get(
        f"{url.rstrip('/')}{path}",
        headers=_headers(token),
        verify=verify_ssl,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def _post(
    url: str,
    path: str,
    token: str,
    verify_ssl: bool,
    body: dict[str, object] | None = None,
) -> requests.Response:
    resp = requests.post(
        f"{url.rstrip('/')}{path}",
        headers=_headers(token),
        json=body or {},
        verify=verify_ssl,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def probe(url: str, token: str, verify_ssl: bool) -> dict[str, object]:
    return cast("dict[str, object]", _get(url, "/api/", token, verify_ssl).json())


def list_devices(
    url: str,
    token: str,
    verify_ssl: bool,
    domain: str | None = None,
    area: str | None = None,
    limit: int = 50,
) -> dict[str, object]:
    states = _get(url, "/api/states", token, verify_ssl).json()
    if domain:
        states = [s for s in states if s["entity_id"].startswith(f"{domain}.")]

    if area:
        try:
            areas = _get(url, "/api/config/area_registry/list", token, verify_ssl).json()
            area_ids = {a["area_id"] for a in areas if area.lower() in a.get("name", "").lower()}
            if area_ids:
                states = [
                    s for s in states
                    if s.get("attributes", {}).get("area_id") in area_ids
                ]
        except requests.HTTPError:
            logger.debug("Area registry not available -- skipping area filter")

    total = len(states)
    devices = [
        {
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", s["entity_id"]),
            "domain": s["entity_id"].split(".")[0],
        }
        for s in states[:limit]
    ]
    return {"devices": devices, "count": len(devices), "total": total}


def get_state(url: str, token: str, verify_ssl: bool, entity_id: str) -> dict[str, object]:
    data = _get(url, f"/api/states/{entity_id}", token, verify_ssl).json()

    return {
        "entity_id": data["entity_id"],
        "state": data["state"],
        "attributes": data.get("attributes", {}),
        "last_changed": data.get("last_changed"),
        "last_updated": data.get("last_updated"),
    }


def control(
    url: str,
    token: str,
    verify_ssl: bool,
    entity_id: str,
    service: str,
    service_data: dict[str, object] | None = None,
) -> dict[str, object]:
    domain = entity_id.split(".")[0]

    body: dict[str, object] = {"entity_id": entity_id}
    if service_data:
        body.update(service_data)
    _post(url, f"/api/services/{domain}/{service}", token, verify_ssl, body)
    return {"status": "ok", "entity_id": entity_id, "service": f"{domain}.{service}"}


def list_automations(url: str, token: str, verify_ssl: bool) -> dict[str, object]:
    states = _get(url, "/api/states", token, verify_ssl).json()

    autos = [
        {
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s.get("attributes", {}).get("friendly_name", s["entity_id"]),
            "last_triggered": s.get("attributes", {}).get("last_triggered"),
        }
        for s in states
        if s["entity_id"].startswith("automation.")
    ]
    return {"automations": autos, "count": len(autos)}


def trigger_automation(url: str, token: str, verify_ssl: bool, automation_id: str) -> dict[str, object]:
    _post(
        url,
        "/api/services/automation/trigger",
        token,
        verify_ssl,
        {"entity_id": automation_id},
    )
    return {"status": "ok", "automation_id": automation_id, "action": "triggered"}
