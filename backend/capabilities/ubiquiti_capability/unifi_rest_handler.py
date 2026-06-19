
import logging
import threading

import requests
from requests.exceptions import SSLError

logger = logging.getLogger(__name__)

_TIMEOUT = 15

_session_lock = threading.Lock()
_session_cache: dict[tuple[str, str], dict] = {}


# ── Auth helpers ─────────────────────────────────────────────────────────────


def _auth_headers(api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def _get_session(url: str, username: str, password: str, verify_ssl: bool) -> tuple[requests.Session, str]:
    key = (url, username)
    with _session_lock:
        cached = _session_cache.get(key)
        if cached:
            return cached["session"], cached["csrf"]

    session = requests.Session()
    login_url = f"{url.rstrip('/')}/api/auth/login"
    body = {"username": username, "password": password, "rememberMe": True}

    try:
        resp = session.post(login_url, json=body, verify=verify_ssl, timeout=_TIMEOUT)
    except SSLError:
        logger.debug("SSL verification failed for %s, retrying without", url)
        resp = session.post(login_url, json=body, verify=False, timeout=_TIMEOUT)

    if not resp.ok:
        safe_body = resp.text[:300].replace("\n", " ").replace("\r", " ")
        logger.warning(
            "[unifi] login %s returned %d: %s", login_url, resp.status_code, safe_body,
        )
    resp.raise_for_status()
    csrf = resp.headers.get("x-csrf-token", "")
    with _session_lock:
        _session_cache[key] = {"session": session, "csrf": csrf}
    return session, csrf


def _invalidate_session(url: str, username: str = "") -> None:
    with _session_lock:
        _session_cache.pop((url, username), None)


# ── Core request helper ──────────────────────────────────────────────────────


def _ssl_call(fn, *args, verify_ssl: bool, url: str, **kwargs) -> requests.Response:
    try:
        return fn(*args, verify=verify_ssl, **kwargs)
    except SSLError:
        logger.debug("SSL verification failed for %s, retrying without", url)
        return fn(*args, verify=False, **kwargs)


def _request(
    method: str,
    url: str,
    path: str,
    api_key: str | None,
    username: str | None,
    password: str | None,
    verify_ssl: bool,
    body: dict | None = None,
) -> requests.Response:
    full_url = f"{url.rstrip('/')}{path}"
    kwargs: dict = {"timeout": _TIMEOUT}
    if body is not None or method in ("post", "put"):
        kwargs["json"] = body or {}

    if api_key:
        fn = getattr(requests, method)
        resp = _ssl_call(fn, full_url, headers=_auth_headers(api_key), verify_ssl=verify_ssl, url=url, **kwargs)
        resp.raise_for_status()
        return resp

    uname = username or ""
    session, csrf = _get_session(url, uname, password or "", verify_ssl)
    needs_csrf = method in ("post", "put")

    def _session_call(sess, tok):
        headers = {"Content-Type": "application/json"}
        if needs_csrf:
            headers["X-CSRF-Token"] = tok
        return _ssl_call(getattr(sess, method), full_url, headers=headers, verify_ssl=verify_ssl, url=url, **kwargs)

    resp = _session_call(session, csrf)

    if resp.status_code == 401:
        _invalidate_session(url, uname)
        session, csrf = _get_session(url, uname, password or "", verify_ssl)
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
    verify_ssl: bool = True,
) -> dict:
    try:
        resp = _request("get", url, "/api/", api_key, username, password, verify_ssl)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ValueError(
                f"Controller at {url} returned 404 on /api/ -- legacy UniFi controllers "
                "are not supported. UniFi OS (UDM/UCK-G2-Plus or newer) is required."
            ) from exc
        raise
    return resp.json()


def list_devices(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    raw = _request("get", url, _site_path(site, "stat/device"), api_key, username, password, verify_ssl).json().get("data", [])
    devices = [
        {
            "mac": d.get("mac", ""),
            "name": d.get("name") or d.get("hostname", ""),
            "model": d.get("model", ""),
            "type": d.get("type", ""),
            "ip": d.get("ip", ""),
            "state": "online" if d.get("state", 0) == 1 else "offline",
            "uptime": d.get("uptime", 0),
            "version": d.get("version", ""),
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
    verify_ssl: bool = True,
    active_only: bool = True,
) -> dict:
    endpoint = "stat/sta" if active_only else "rest/user"
    raw = _request("get", url, _site_path(site, endpoint), api_key, username, password, verify_ssl).json().get("data", [])
    clients = [
        {
            "mac": c.get("mac", ""),
            "name": c.get("name") or c.get("hostname", ""),
            "hostname": c.get("hostname", ""),
            "ip": c.get("ip", ""),
            "network": c.get("network", ""),
            "is_wired": bool(c.get("is_wired", False)),
            "uptime": c.get("uptime", 0),
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
    verify_ssl: bool = True,
) -> dict:
    if target == "health":
        data = _request("get", url, _site_path(site, "stat/health"), api_key, username, password, verify_ssl).json()
        return {"health": data.get("data", [])}

    data = _request("get", url, _site_path(site, "stat/device"), api_key, username, password, verify_ssl).json()
    mac_lower = target.lower()
    for device in data.get("data", []):
        if device.get("mac", "").lower() == mac_lower:
            return device
    return {"error": f"Device with MAC {target!r} not found"}


def control_client(
    url: str,
    site: str,
    mac: str,
    command: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
    **kwargs,
) -> dict:
    _request("post", url, _site_path(site, "cmd/stamgr"), api_key, username, password, verify_ssl, {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def control_device(
    url: str,
    site: str,
    mac: str,
    command: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
    **kwargs,
) -> dict:
    _request("post", url, _site_path(site, "cmd/devmgr"), api_key, username, password, verify_ssl, {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def list_wlans(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    raw = _request("get", url, _site_path(site, "rest/wlanconf"), api_key, username, password, verify_ssl).json().get("data", [])
    wlans = [
        {
            "_id": w.get("_id", ""),
            "name": w.get("name", ""),
            "enabled": bool(w.get("enabled", True)),
            "security": w.get("security", ""),
            "wpa_mode": w.get("wpa_mode", ""),
            "hide_ssid": bool(w.get("hide_ssid", False)),
        }
        for w in raw
    ]
    return {"wlans": wlans, "count": len(wlans)}


def update_wlan(
    url: str,
    site: str,
    wlan_id: str,
    updates: dict,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    _request("put", url, _site_path(site, f"rest/wlanconf/{wlan_id}"), api_key, username, password, verify_ssl, updates)
    return {"status": "ok", "wlan_id": wlan_id, "updated": list(updates.keys())}


def list_port_forwards(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    raw = _request("get", url, _site_path(site, "rest/portforward"), api_key, username, password, verify_ssl).json().get("data", [])
    rules = [
        {
            "_id": r.get("_id", ""),
            "name": r.get("name", ""),
            "enabled": bool(r.get("enabled", True)),
            "proto": r.get("proto", ""),
            "dst_port": r.get("dst_port", ""),
            "fwd": r.get("fwd", ""),
            "fwd_port": r.get("fwd_port", ""),
        }
        for r in raw
    ]
    return {"rules": rules, "count": len(rules)}


def update_port_forward(
    url: str,
    site: str,
    rule_id: str,
    updates: dict,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    _request("put", url, _site_path(site, f"rest/portforward/{rule_id}"), api_key, username, password, verify_ssl, updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}


def create_port_forward(
    url: str,
    site: str,
    rule: dict,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    resp = _request("post", url, _site_path(site, "rest/portforward"), api_key, username, password, verify_ssl, rule)
    created = resp.json().get("data", [{}])
    return {"status": "ok", "rule": created[0] if created else {}}


def list_traffic_rules(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    data = _request("get", url, _v2_path(site, "trafficrules"), api_key, username, password, verify_ssl).json()
    raw = data if isinstance(data, list) else data.get("data", [])
    rules = [
        {
            "_id": r.get("_id", ""),
            "description": r.get("description", ""),
            "enabled": bool(r.get("enabled", True)),
            "action": r.get("action", ""),
            "matching_target": r.get("matching_target", ""),
        }
        for r in raw
    ]
    return {"rules": rules, "count": len(rules)}


def update_traffic_rule(
    url: str,
    site: str,
    rule_id: str,
    updates: dict,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    _request("put", url, _v2_path(site, f"trafficrules/{rule_id}"), api_key, username, password, verify_ssl, updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}
