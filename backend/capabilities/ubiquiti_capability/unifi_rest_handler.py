"""Stateless REST client for the Ubiquiti UniFi Controller API (UniFi OS only).

Every function accepts connection parameters (url, site, verify_ssl, api_key,
username, password) rather than storing them -- the caller (UbiquitiCapability)
owns credential lifecycle.

Auth modes:
  - API key: pass ``api_key``; adds ``X-API-KEY`` header.
  - Session: pass ``username`` + ``password``; logs in via POST /api/auth/login,
    stores cookie + CSRF token in a module-level session cache keyed by URL.

Base path: /proxy/network/api/s/{site}/
Traffic rules path: /v2/api/site/{site}/trafficrules
"""

import logging

import requests
from requests.exceptions import SSLError

logger = logging.getLogger(__name__)

_TIMEOUT = 15

# Module-level session cache for username/password auth.
# Keyed by controller URL; values are {"session": requests.Session, "csrf": str}.
_session_cache: dict[str, dict] = {}


# ── Auth helpers ─────────────────────────────────────────────────────────────


def _auth_headers(api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    return headers


def _get_session(url: str, username: str, password: str, verify_ssl: bool) -> tuple[requests.Session, str]:
    """Return a (session, csrf_token) pair, logging in if no valid session cached."""
    cached = _session_cache.get(url)
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

    resp.raise_for_status()
    csrf = resp.headers.get("x-csrf-token", "")
    _session_cache[url] = {"session": session, "csrf": csrf}
    return session, csrf


def _invalidate_session(url: str) -> None:
    _session_cache.pop(url, None)


# ── Internal request helpers ─────────────────────────────────────────────────


def _get(
    url: str,
    path: str,
    api_key: str | None,
    username: str | None,
    password: str | None,
    verify_ssl: bool,
) -> requests.Response:
    full_url = f"{url.rstrip('/')}{path}"

    if api_key:
        try:
            resp = requests.get(full_url, headers=_auth_headers(api_key), verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = requests.get(full_url, headers=_auth_headers(api_key), verify=False, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp

    session, _ = _get_session(url, username or "", password or "", verify_ssl)
    try:
        resp = session.get(full_url, headers={"Content-Type": "application/json"}, verify=verify_ssl, timeout=_TIMEOUT)
    except SSLError:
        logger.debug("SSL verification failed for %s, retrying without", url)
        resp = session.get(full_url, headers={"Content-Type": "application/json"}, verify=False, timeout=_TIMEOUT)

    if resp.status_code == 401:
        _invalidate_session(url)
        session, _ = _get_session(url, username or "", password or "", verify_ssl)
        try:
            resp = session.get(full_url, headers={"Content-Type": "application/json"}, verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = session.get(full_url, headers={"Content-Type": "application/json"}, verify=False, timeout=_TIMEOUT)

    resp.raise_for_status()
    return resp


def _post(
    url: str,
    path: str,
    api_key: str | None,
    username: str | None,
    password: str | None,
    verify_ssl: bool,
    body: dict | None = None,
) -> requests.Response:
    full_url = f"{url.rstrip('/')}{path}"
    payload = body or {}

    if api_key:
        try:
            resp = requests.post(full_url, headers=_auth_headers(api_key), json=payload, verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = requests.post(full_url, headers=_auth_headers(api_key), json=payload, verify=False, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp

    session, csrf = _get_session(url, username or "", password or "", verify_ssl)
    headers = {"Content-Type": "application/json", "X-CSRF-Token": csrf}
    try:
        resp = session.post(full_url, headers=headers, json=payload, verify=verify_ssl, timeout=_TIMEOUT)
    except SSLError:
        logger.debug("SSL verification failed for %s, retrying without", url)
        resp = session.post(full_url, headers=headers, json=payload, verify=False, timeout=_TIMEOUT)

    if resp.status_code == 401:
        _invalidate_session(url)
        session, csrf = _get_session(url, username or "", password or "", verify_ssl)
        headers["X-CSRF-Token"] = csrf
        try:
            resp = session.post(full_url, headers=headers, json=payload, verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = session.post(full_url, headers=headers, json=payload, verify=False, timeout=_TIMEOUT)

    resp.raise_for_status()
    return resp


def _put(
    url: str,
    path: str,
    api_key: str | None,
    username: str | None,
    password: str | None,
    verify_ssl: bool,
    body: dict | None = None,
) -> requests.Response:
    full_url = f"{url.rstrip('/')}{path}"
    payload = body or {}

    if api_key:
        try:
            resp = requests.put(full_url, headers=_auth_headers(api_key), json=payload, verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = requests.put(full_url, headers=_auth_headers(api_key), json=payload, verify=False, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp

    session, csrf = _get_session(url, username or "", password or "", verify_ssl)
    headers = {"Content-Type": "application/json", "X-CSRF-Token": csrf}
    try:
        resp = session.put(full_url, headers=headers, json=payload, verify=verify_ssl, timeout=_TIMEOUT)
    except SSLError:
        logger.debug("SSL verification failed for %s, retrying without", url)
        resp = session.put(full_url, headers=headers, json=payload, verify=False, timeout=_TIMEOUT)

    if resp.status_code == 401:
        _invalidate_session(url)
        session, csrf = _get_session(url, username or "", password or "", verify_ssl)
        headers["X-CSRF-Token"] = csrf
        try:
            resp = session.put(full_url, headers=headers, json=payload, verify=verify_ssl, timeout=_TIMEOUT)
        except SSLError:
            logger.debug("SSL verification failed for %s, retrying without", url)
            resp = session.put(full_url, headers=headers, json=payload, verify=False, timeout=_TIMEOUT)

    resp.raise_for_status()
    return resp


# ── Path helpers ─────────────────────────────────────────────────────────────


def _site_path(site: str, endpoint: str) -> str:
    """Build a UniFi OS proxy path: /proxy/network/api/s/{site}/{endpoint}."""
    return f"/proxy/network/api/s/{site}/{endpoint}"


def _v2_path(site: str, endpoint: str) -> str:
    """Build a v2 API path: /v2/api/site/{site}/{endpoint}."""
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
    """Health check -- GET /api/ with auth. Returns API root response.

    Raises:
        ValueError: If the controller is not running UniFi OS (legacy 404).
        requests.HTTPError: On any non-2xx response.
    """
    resp = _get(url, "/api/", api_key, username, password, verify_ssl)
    if resp.status_code == 404:
        raise ValueError(
            f"Controller at {url} returned 404 on /api/ -- legacy UniFi controllers "
            "are not supported. UniFi OS (UDM/UCK-G2-Plus or newer) is required."
        )
    return resp.json()


def list_devices(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    """List all adopted network devices on the site.

    Returns:
        dict: ``{"devices": [...], "count": N}`` where each device has
        ``mac``, ``name``, ``model``, ``type``, ``ip``, ``state``, ``uptime``,
        ``version``. ``state`` is "online" or "offline".
    """
    data = _get(url, _site_path(site, "stat/device"), api_key, username, password, verify_ssl).json()
    raw = data.get("data", [])
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
    """List clients connected to the network.

    Args:
        active_only: When ``True``, queries ``stat/sta`` (currently associated).
            When ``False``, queries ``rest/user`` (all known clients).

    Returns:
        dict: ``{"clients": [...], "count": N}`` where each client has
        ``mac``, ``name``, ``hostname``, ``ip``, ``network``, ``is_wired``,
        ``uptime``.
    """
    endpoint = "stat/sta" if active_only else "rest/user"
    data = _get(url, _site_path(site, endpoint), api_key, username, password, verify_ssl).json()
    raw = data.get("data", [])
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
    """Get detailed info for a device (by MAC) or site health.

    Args:
        target: Either ``"health"`` to get subsystem statuses, or a device MAC
            address to get full device detail.

    Returns:
        dict: Health subsystem list when ``target == "health"``, or full device
        detail dict when ``target`` is a MAC.
    """
    if target == "health":
        data = _get(url, _site_path(site, "stat/health"), api_key, username, password, verify_ssl).json()
        return {"health": data.get("data", [])}

    data = _get(url, _site_path(site, "stat/device"), api_key, username, password, verify_ssl).json()
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
    """Send a client management command via cmd/stamgr.

    Args:
        mac: Client MAC address.
        command: One of ``"block-sta"``, ``"unblock-sta"``, ``"kick-sta"``,
            ``"authorize-guest"``.
        **kwargs: Extra command parameters, e.g. for ``authorize-guest``:
            ``minutes``, ``up``, ``down``, ``bytes``.

    Returns:
        dict: ``{"status": "ok", "command": command, "mac": mac}``
    """
    body = {"cmd": command, "mac": mac, **kwargs}
    _post(url, _site_path(site, "cmd/stamgr"), api_key, username, password, verify_ssl, body)
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
    """Send a device management command via cmd/devmgr.

    Args:
        mac: Device MAC address.
        command: One of ``"restart"``, ``"set-locate"``, ``"unset-locate"``,
            ``"power-cycle"``.
        **kwargs: Extra command parameters, e.g. for ``power-cycle``:
            ``port_idx``.

    Returns:
        dict: ``{"status": "ok", "command": command, "mac": mac}``
    """
    body = {"cmd": command, "mac": mac, **kwargs}
    _post(url, _site_path(site, "cmd/devmgr"), api_key, username, password, verify_ssl, body)
    return {"status": "ok", "command": command, "mac": mac}


def list_wlans(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    """List all WLAN configurations on the site.

    The passphrase (``x_passphrase``) is intentionally excluded from the
    output to avoid leaking credentials into LLM context.

    Returns:
        dict: ``{"wlans": [...], "count": N}`` where each WLAN has
        ``_id``, ``name``, ``enabled``, ``security``, ``wpa_mode``,
        ``hide_ssid``.
    """
    data = _get(url, _site_path(site, "rest/wlanconf"), api_key, username, password, verify_ssl).json()
    raw = data.get("data", [])
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
    """Update a WLAN configuration.

    Args:
        wlan_id: The ``_id`` of the WLAN to update.
        updates: Fields to update, e.g. ``{"enabled": False}``,
            ``{"x_passphrase": "new-password"}``, ``{"hide_ssid": True}``.

    Returns:
        dict: ``{"status": "ok", "wlan_id": wlan_id, "updated": [...field names]}``
    """
    _put(url, _site_path(site, f"rest/wlanconf/{wlan_id}"), api_key, username, password, verify_ssl, updates)
    return {"status": "ok", "wlan_id": wlan_id, "updated": list(updates.keys())}


def list_port_forwards(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    """List all port forwarding rules on the site.

    Returns:
        dict: ``{"rules": [...], "count": N}`` where each rule has
        ``_id``, ``name``, ``enabled``, ``proto``, ``dst_port``, ``fwd``,
        ``fwd_port``.
    """
    data = _get(url, _site_path(site, "rest/portforward"), api_key, username, password, verify_ssl).json()
    raw = data.get("data", [])
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
    """Update an existing port forwarding rule.

    Args:
        rule_id: The ``_id`` of the rule to update.
        updates: Fields to update, e.g. ``{"enabled": False}``.

    Returns:
        dict: ``{"status": "ok", "rule_id": rule_id, "updated": [...field names]}``
    """
    _put(url, _site_path(site, f"rest/portforward/{rule_id}"), api_key, username, password, verify_ssl, updates)
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
    """Create a new port forwarding rule.

    Args:
        rule: Rule definition dict. Required fields typically include
            ``name``, ``proto``, ``dst_port``, ``fwd``, ``fwd_port``.

    Returns:
        dict: ``{"status": "ok", "rule": <created rule dict>}``
    """
    resp = _post(url, _site_path(site, "rest/portforward"), api_key, username, password, verify_ssl, rule)
    data = resp.json()
    created = data.get("data", [{}])
    return {"status": "ok", "rule": created[0] if created else {}}


def list_traffic_rules(
    url: str,
    site: str,
    api_key: str | None = None,
    username: str | None = None,
    password: str | None = None,
    verify_ssl: bool = True,
) -> dict:
    """List traffic rules (firewall policies) on the site.

    Uses the v2 API path: ``/v2/api/site/{site}/trafficrules``.

    Returns:
        dict: ``{"rules": [...], "count": N}`` where each rule has
        ``_id``, ``description``, ``enabled``, ``action``,
        ``matching_target``.
    """
    data = _get(url, _v2_path(site, "trafficrules"), api_key, username, password, verify_ssl).json()
    # v2 API returns a list directly, not wrapped in {"data": [...]}
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
    """Update a traffic rule.

    Uses the v2 API path: ``/v2/api/site/{site}/trafficrules/{rule_id}``.
    The controller returns 201 on success.

    Args:
        rule_id: The ``_id`` of the traffic rule to update.
        updates: Fields to update, e.g. ``{"enabled": False}``.

    Returns:
        dict: ``{"status": "ok", "rule_id": rule_id, "updated": [...field names]}``
    """
    _put(url, _v2_path(site, f"trafficrules/{rule_id}"), api_key, username, password, verify_ssl, updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}
