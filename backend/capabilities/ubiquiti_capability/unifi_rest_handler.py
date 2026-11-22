
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, cast

import requests

from capabilities.ubiquiti_capability import authenticator
from exceptions import AuthenticatorCodeRefused, AuthenticatorKeyMissing, TooManySignInAttempts

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_TIMEOUT = 15
# What a console answers a UI.com account that has an authenticator app, until a code is sent with it.
_MFA_REQUIRED = 499

_session_lock = threading.Lock()
# A session belongs to the exact sign-in that made it, so a changed password or setup key signs in again.
_session_cache: dict[tuple[str, str, str, str | None], dict[str, object]] = {}


# ── Auth helpers ─────────────────────────────────────────────────────────────


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def _mfa_cookie(resp: requests.Response) -> str | None:
    """The half-made sign-in a 499 reply hands back, as the ``name=value`` cookie that finishes it."""
    try:
        data = resp.json().get("data")
    except (ValueError, AttributeError):
        return None
    raw = data.get("mfaCookie") if isinstance(data, dict) else None
    cookie = raw.split(";")[0].strip() if isinstance(raw, str) else ""
    return cookie if "=" in cookie else None


def _sign_in_refusal(status: int, totp_key: str | None, code_sent: bool) -> type[Exception] | None:
    """The refusal a failed UI.com sign-in is raised as, when its reason is known."""
    if status == _MFA_REQUIRED and not totp_key:
        return AuthenticatorKeyMissing
    if status == _MFA_REQUIRED and code_sent:
        return AuthenticatorCodeRefused
    if status == 429 and totp_key:
        return TooManySignInAttempts
    return None


def _get_session(
    url: str, username: str, password: str, totp_key: str | None, verify_ssl: bool,
) -> tuple[requests.Session, str]:
    key = (url, username, password, totp_key)
    with _session_lock:
        cached = _session_cache.get(key)
        if cached:
            return cast(requests.Session, cached["session"]), cast(str, cached["csrf"])

    session = requests.Session()
    login_url = f"{url.rstrip('/')}/api/auth/login"
    body: dict[str, str | bool] = {"username": username, "password": password, "rememberMe": True}

    resp = session.post(login_url, json=body, verify=verify_ssl, timeout=_TIMEOUT)
    code_sent = False
    if resp.status_code == _MFA_REQUIRED and totp_key and (cookie := _mfa_cookie(resp)):
        # The cookie goes as a header: a cookie jar can drop one set for a console addressed by IP.
        resp = session.post(
            login_url, json={**body, "token": authenticator.code(totp_key, time.time())},
            headers={"Cookie": cookie}, verify=verify_ssl, timeout=_TIMEOUT,
        )
        code_sent = True
    if not resp.ok:
        # A 499 reply carries the half-made sign-in, so only its status is logged; so is any reply to a code.
        shown = "" if code_sent or resp.status_code == _MFA_REQUIRED else resp.text[:300]
        logger.warning(
            "[unifi] login %s returned %d: %s", login_url, resp.status_code,
            shown.replace("\n", " ").replace("\r", " "),
        )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        refusal = _sign_in_refusal(resp.status_code, totp_key, code_sent)
        if refusal is None:
            raise
        # The same words as the request error, which the model may read: status and address only.
        raise refusal(str(exc)) from exc
    csrf = resp.headers.get("x-csrf-token", "")
    with _session_lock:
        _session_cache[key] = {"session": session, "csrf": csrf}
    return session, csrf


def _invalidate_session(url: str, username: str, password: str, totp_key: str | None) -> None:
    with _session_lock:
        _session_cache.pop((url, username, password, totp_key), None)


# ── Core request helper ──────────────────────────────────────────────────────


def _request(
    method: str,
    url: str,
    path: str,
    api_key: str | None,
    username: str | None,
    password: str | None,
    totp_key: str | None,
    verify_ssl: bool,
    body: dict[str, object] | None = None,
) -> requests.Response:
    full_url = f"{url.rstrip('/')}{path}"
    kwargs: dict[str, object] = {"timeout": _TIMEOUT}
    if body is not None or method in ("post", "put"):
        kwargs["json"] = body or {}

    if api_key:
        fn = cast("Callable[..., requests.Response]", getattr(requests, method))
        resp = fn(full_url, headers=_auth_headers(api_key), verify=verify_ssl, **kwargs)
        resp.raise_for_status()
        return resp

    uname = username or ""
    session, csrf = _get_session(url, uname, password or "", totp_key, verify_ssl)

    def _session_call(sess: requests.Session, tok: str) -> requests.Response:
        headers = {"Content-Type": "application/json", "X-CSRF-Token": tok}
        return cast("Callable[..., requests.Response]", getattr(sess, method))(full_url, headers=headers, verify=verify_ssl, **kwargs)

    resp = _session_call(session, csrf)

    if resp.status_code == 401:
        _invalidate_session(url, uname, password or "", totp_key)
        session, csrf = _get_session(url, uname, password or "", totp_key, verify_ssl)
        resp = _session_call(session, csrf)

    resp.raise_for_status()
    return resp


# ── Path helpers ─────────────────────────────────────────────────────────────


def _site_path(site: str, endpoint: str) -> str:
    return f"/proxy/network/api/s/{site}/{endpoint}"


def _v2_path(site: str, endpoint: str) -> str:
    return f"/v2/api/site/{site}/{endpoint}"


# ── Public API ───────────────────────────────────────────────────────────────


def probe(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    try:
        resp = _request("get", url, "/api/", api_key, username, password, totp_key, verify_ssl)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ValueError(
                f"Controller at {url} returned 404 on /api/ -- legacy UniFi controllers "
                "are not supported. UniFi OS (UDM/UCK-G2-Plus or newer) is required."
            ) from exc
        raise
    return cast(dict[str, object], resp.json())


def list_devices(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    raw = cast(list[object], _request("get", url, _site_path(site, "stat/device"), api_key, username, password, totp_key, verify_ssl).json().get("data", []))
    devices = [
        {
            "mac": cast(dict[str, object], d).get("mac", ""),
            "name": cast(dict[str, object], d).get("name") or cast(dict[str, object], d).get("hostname", ""),
            "model": cast(dict[str, object], d).get("model", ""),
            "type": cast(dict[str, object], d).get("type", ""),
            "ip": cast(dict[str, object], d).get("ip", ""),
            "state": "online" if cast(dict[str, object], d).get("state", 0) == 1 else "offline",
            "uptime": cast(dict[str, object], d).get("uptime", 0),
            "version": cast(dict[str, object], d).get("version", ""),
        }
        for d in raw
    ]
    return {"devices": devices, "count": len(devices)}


def list_clients(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
    active_only: bool = True,
) -> dict[str, object]:
    endpoint = "stat/sta" if active_only else "rest/user"
    raw = cast(list[object], _request("get", url, _site_path(site, endpoint), api_key, username, password, totp_key, verify_ssl).json().get("data", []))
    clients = [
        {
            "mac": cast(dict[str, object], c).get("mac", ""),
            "name": cast(dict[str, object], c).get("name") or cast(dict[str, object], c).get("hostname", ""),
            "hostname": cast(dict[str, object], c).get("hostname", ""),
            "ip": cast(dict[str, object], c).get("ip", ""),
            "network": cast(dict[str, object], c).get("network", ""),
            "is_wired": bool(cast(dict[str, object], c).get("is_wired", False)),
            "uptime": cast(dict[str, object], c).get("uptime", 0),
        }
        for c in raw
    ]
    return {"clients": clients, "count": len(clients)}


def get_info(
    url: str,
    site: str,
    target: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    if target == "health":
        data = cast(dict[str, object], _request("get", url, _site_path(site, "stat/health"), api_key, username, password, totp_key, verify_ssl).json())
        return {"health": data.get("data", [])}

    data = cast(dict[str, object], _request("get", url, _site_path(site, "stat/device"), api_key, username, password, totp_key, verify_ssl).json())
    mac_lower = target.lower()
    for device in cast(list[object], data.get("data", [])):
        if cast(str, cast(dict[str, object], device).get("mac", "")).lower() == mac_lower:
            return cast(dict[str, object], device)
    return {"error": f"Device with MAC {target!r} not found"}


def control_client(
    url: str,
    site: str,
    mac: str,
    command: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
    **kwargs: object,
) -> dict[str, object]:
    _request("post", url, _site_path(site, "cmd/stamgr"), api_key, username, password, totp_key, verify_ssl, {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def control_device(
    url: str,
    site: str,
    mac: str,
    command: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
    **kwargs: object,
) -> dict[str, object]:
    _request("post", url, _site_path(site, "cmd/devmgr"), api_key, username, password, totp_key, verify_ssl, {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def list_wlans(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    raw = cast(list[object], _request("get", url, _site_path(site, "rest/wlanconf"), api_key, username, password, totp_key, verify_ssl).json().get("data", []))
    wlans = [
        {
            "_id": cast(dict[str, object], w).get("_id", ""),
            "name": cast(dict[str, object], w).get("name", ""),
            "enabled": bool(cast(dict[str, object], w).get("enabled", True)),
            "security": cast(dict[str, object], w).get("security", ""),
            "wpa_mode": cast(dict[str, object], w).get("wpa_mode", ""),
            "hide_ssid": bool(cast(dict[str, object], w).get("hide_ssid", False)),
        }
        for w in raw
    ]
    return {"wlans": wlans, "count": len(wlans)}


def update_wlan(
    url: str,
    site: str,
    wlan_id: str,
    updates: dict[str, object],
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    _request("put", url, _site_path(site, f"rest/wlanconf/{wlan_id}"), api_key, username, password, totp_key, verify_ssl, updates)
    return {"status": "ok", "wlan_id": wlan_id, "updated": list(updates.keys())}


def list_port_forwards(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    raw = cast(list[object], _request("get", url, _site_path(site, "rest/portforward"), api_key, username, password, totp_key, verify_ssl).json().get("data", []))
    rules = [
        {
            "_id": cast(dict[str, object], r).get("_id", ""),
            "name": cast(dict[str, object], r).get("name", ""),
            "enabled": bool(cast(dict[str, object], r).get("enabled", True)),
            "proto": cast(dict[str, object], r).get("proto", ""),
            "dst_port": cast(dict[str, object], r).get("dst_port", ""),
            "fwd": cast(dict[str, object], r).get("fwd", ""),
            "fwd_port": cast(dict[str, object], r).get("fwd_port", ""),
        }
        for r in raw
    ]
    return {"rules": rules, "count": len(rules)}


def update_port_forward(
    url: str,
    site: str,
    rule_id: str,
    updates: dict[str, object],
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    _request("put", url, _site_path(site, f"rest/portforward/{rule_id}"), api_key, username, password, totp_key, verify_ssl, updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}


def create_port_forward(
    url: str,
    site: str,
    rule: dict[str, object],
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    resp = _request("post", url, _site_path(site, "rest/portforward"), api_key, username, password, totp_key, verify_ssl, rule)
    created = cast(list[object], resp.json().get("data", [{}]))
    return {"status": "ok", "rule": created[0] if created else {}}


def list_traffic_rules(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    data = cast(dict[str, object], _request("get", url, _v2_path(site, "trafficrules"), api_key, username, password, totp_key, verify_ssl).json())
    raw = cast(list[object], data if isinstance(data, list) else data.get("data", []))
    rules = [
        {
            "_id": cast(dict[str, object], r).get("_id", ""),
            "description": cast(dict[str, object], r).get("description", ""),
            "enabled": bool(cast(dict[str, object], r).get("enabled", True)),
            "action": cast(dict[str, object], r).get("action", ""),
            "matching_target": cast(dict[str, object], r).get("matching_target", ""),
        }
        for r in raw
    ]
    return {"rules": rules, "count": len(rules)}


def update_traffic_rule(
    url: str,
    site: str,
    rule_id: str,
    updates: dict[str, object],
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    totp_key: str | None = None,
    verify_ssl: bool = True,
) -> dict[str, object]:
    _request("put", url, _v2_path(site, f"trafficrules/{rule_id}"), api_key, username, password, totp_key, verify_ssl, updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}
