"""REST client for a UniFi OS console, signed in with a UI.com account.

Every call reuses the latest stored console session. With none, or once the
console refuses it, the call signs in with the saved username and password;
when UI.com wants a 2FA code, the user is asked for it and the call waits,
with no time limit, until it is typed, the ask is dismissed or the run stops.
The caller (UbiquitiCapability) owns where the session is stored.

A sign-in failure the model reads is raised as :class:`UnifiCallFailed`
carrying one fixed text. The 2FA code and the half-made sign-in cookie live in
local variables only: never logged, stored, raised or returned.
"""

from __future__ import annotations

import logging
import ssl
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, cast

import requests

from exceptions import UnifiCallFailed
from services.ask_user import AskReply, AskUser, CodeAsk, CodeReply, Origin

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_TIMEOUT = 15
# What a console answers a UI.com account with 2FA on, until a code is sent with it.
_MFA_REQUIRED = 499
_LOGIN_PATH = "/api/auth/login"
_POLL_SECONDS = 0.25

_CANCELLED = "UniFi sign-in cancelled before a 2FA code was entered."
_TOO_MANY = "UI.com blocked sign-ins for a few minutes after too many attempts."
_REJECTED = "UniFi rejected the saved username or password."
_CODE_REFUSED = "Failed - 2FA Error"
_UNREACHABLE = "Failed - network unreachable"
_CERT_UNTRUSTED = "Failed - certificate not trusted"

# Held only while signing in, never across a data call: a call that has a
# session runs while another waits for a code.
_sign_in_lock = threading.Lock()


class Session(NamedTuple):
    """A signed-in console session: the cookie header and the CSRF token sent with every call."""

    cookie: str
    csrf: str


@dataclass(frozen=True)
class Console:
    """Everything a UniFi call needs to reach the console and sign in to it.

    ``stored`` returns the latest stored session, or ``None`` when there is
    none; ``keep`` stores a new one. ``origin`` is the turn a code ask belongs
    to, and ``should_stop`` ends the wait for a code when the run stops.
    """

    url: str
    site: str
    username: str
    password: str = field(repr=False)
    verify_ssl: bool
    stored: Callable[[], Session | None]
    keep: Callable[[Session], None]
    origin: dict[str, object] | None
    should_stop: Callable[[], bool] | None


# ── Sign-in ──────────────────────────────────────────────────────────────────


def _mfa_cookie(resp: requests.Response) -> str | None:
    """The half-made sign-in a 499 reply hands back, as the ``name=value`` cookie that finishes it."""
    try:
        data = resp.json().get("data")
    except (ValueError, AttributeError):
        return None
    raw = data.get("mfaCookie") if isinstance(data, dict) else None
    cookie = raw.split(";")[0].strip() if isinstance(raw, str) else ""
    return cookie if "=" in cookie else None


def _ask_code(console: Console) -> str:
    """The 2FA code the user types. Waits for a reply, a dismiss or a stop, with no time limit."""
    done = threading.Event()
    got: dict[str, str | None] = {}

    def on_reply(reply: AskReply) -> None:
        got["code"] = cast("CodeReply", reply.value).code if reply.answered else None
        done.set()

    origin = None if console.origin is None else Origin.model_validate(console.origin)
    ask_id = AskUser.code(CodeAsk(service="unifi", origin=origin), on_reply)
    try:
        while not done.wait(_POLL_SECONDS):
            if console.should_stop is not None and console.should_stop():
                AskUser.cancel(ask_id)
    finally:
        # An exception out of the wait still closes the ask; once it is
        # answered or cancelled this is a no-op.
        AskUser.cancel(ask_id)
    code = got["code"]
    if code is None:
        raise UnifiCallFailed(_CANCELLED)
    return code


def _post_login(console: Console, body: dict[str, object], headers: dict[str, str]) -> requests.Response:
    resp = _send("post", console, _LOGIN_PATH, headers, body)
    if not resp.ok:
        logger.warning("[unifi] sign-in %s returned %d", _LOGIN_PATH, resp.status_code)
    return resp


def _log_in(console: Console) -> Session:
    """Sign in with the saved username and password, asking for a 2FA code when UI.com wants one."""
    body: dict[str, object] = {"username": console.username, "password": console.password, "rememberMe": True}
    resp = _post_login(console, body, {})
    if resp.status_code in (401, 403):
        raise UnifiCallFailed(_REJECTED)
    if resp.status_code == _MFA_REQUIRED and (mfa_cookie := _mfa_cookie(resp)):
        # The cookie goes as a header: a cookie jar can drop one set for a console addressed by IP.
        resp = _post_login(console, {**body, "token": _ask_code(console)}, {"Cookie": mfa_cookie})
        if resp.status_code == _MFA_REQUIRED:
            raise UnifiCallFailed(_CODE_REFUSED)
    if resp.status_code == 429:
        raise UnifiCallFailed(_TOO_MANY)
    resp.raise_for_status()
    session = Session(
        cookie="; ".join(f"{c.name}={c.value}" for c in resp.cookies),
        csrf=resp.headers.get("x-csrf-token", ""),
    )
    console.keep(session)
    return session


def _sign_in(console: Console, stale: Session | None) -> Session:
    """A fresh session in place of *stale*. Signs in once for every call waiting
    here: a call that finds a newer stored session uses it instead."""
    while not _sign_in_lock.acquire(timeout=_POLL_SECONDS):
        if console.should_stop is not None and console.should_stop():
            raise UnifiCallFailed(_CANCELLED)
    try:
        latest = console.stored()
        if latest is not None and latest != stale:
            return latest
        return _log_in(console)
    finally:
        _sign_in_lock.release()


# ── Core request helper ──────────────────────────────────────────────────────


def _send(
    method: str, console: Console, path: str, headers: dict[str, str], body: dict[str, object] | None,
) -> requests.Response:
    """One HTTP call. No answer is raised as the unreachable text, a certificate that fails
    the check as the untrusted text, and any other TLS failure as it is."""
    try:
        return cast("Callable[..., requests.Response]", requests.request)(
            method, f"{console.url.rstrip('/')}{path}",
            headers=headers, json=body, verify=console.verify_ssl, timeout=_TIMEOUT,
        )
    except requests.exceptions.SSLError as exc:
        if _certificate_refused(exc):
            raise UnifiCallFailed(_CERT_UNTRUSTED) from exc
        raise
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise UnifiCallFailed(_UNREACHABLE) from exc


def _certificate_refused(exc: BaseException) -> bool:
    """Whether *exc* comes from the console's certificate failing the check, rather
    than another TLS failure such as an https address on a plain-http port."""
    seen: list[BaseException] = []
    cause: BaseException | None = exc
    while cause is not None and cause not in seen:
        if isinstance(cause, ssl.SSLCertVerificationError):
            return True
        seen.append(cause)
        cause = cause.__cause__ or cause.__context__
    return False


def _call(
    method: str, console: Console, path: str, session: Session, body: dict[str, object] | None,
) -> requests.Response:
    headers = {"Content-Type": "application/json", "X-CSRF-Token": session.csrf, "Cookie": session.cookie}
    return _send(method, console, path, headers, body)


def _request(
    method: str, console: Console, path: str, body: dict[str, object] | None = None,
) -> requests.Response:
    session = console.stored()
    if session is None:
        session = _sign_in(console, None)
    resp = _call(method, console, path, session, body)

    if resp.status_code == 401:
        resp = _call(method, console, path, _sign_in(console, session), body)

    resp.raise_for_status()
    return resp


# ── Path helpers ─────────────────────────────────────────────────────────────


def _site_path(site: str, endpoint: str) -> str:
    return f"/proxy/network/api/s/{site}/{endpoint}"


def _v2_path(site: str, endpoint: str) -> str:
    return f"/v2/api/site/{site}/{endpoint}"


# ── Public API ───────────────────────────────────────────────────────────────


def list_devices(console: Console) -> dict[str, object]:
    raw = cast(list[object], _request("get", console, _site_path(console.site, "stat/device")).json().get("data", []))
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


def list_clients(console: Console, active_only: bool = True) -> dict[str, object]:
    endpoint = "stat/sta" if active_only else "rest/user"
    raw = cast(list[object], _request("get", console, _site_path(console.site, endpoint)).json().get("data", []))
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


def get_info(console: Console, target: str) -> dict[str, object]:
    if target == "health":
        data = cast(dict[str, object], _request("get", console, _site_path(console.site, "stat/health")).json())
        return {"health": data.get("data", [])}

    data = cast(dict[str, object], _request("get", console, _site_path(console.site, "stat/device")).json())
    mac_lower = target.lower()
    for device in cast(list[object], data.get("data", [])):
        if cast(str, cast(dict[str, object], device).get("mac", "")).lower() == mac_lower:
            return cast(dict[str, object], device)
    return {"error": f"Device with MAC {target!r} not found"}


def control_client(console: Console, mac: str, command: str, **kwargs: object) -> dict[str, object]:
    _request("post", console, _site_path(console.site, "cmd/stamgr"), {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def control_device(console: Console, mac: str, command: str, **kwargs: object) -> dict[str, object]:
    _request("post", console, _site_path(console.site, "cmd/devmgr"), {"cmd": command, "mac": mac, **kwargs})
    return {"status": "ok", "command": command, "mac": mac}


def list_wlans(console: Console) -> dict[str, object]:
    raw = cast(list[object], _request("get", console, _site_path(console.site, "rest/wlanconf")).json().get("data", []))
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


def update_wlan(console: Console, wlan_id: str, updates: dict[str, object]) -> dict[str, object]:
    _request("put", console, _site_path(console.site, f"rest/wlanconf/{wlan_id}"), updates)
    return {"status": "ok", "wlan_id": wlan_id, "updated": list(updates.keys())}


def list_port_forwards(console: Console) -> dict[str, object]:
    raw = cast(list[object], _request("get", console, _site_path(console.site, "rest/portforward")).json().get("data", []))
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


def update_port_forward(console: Console, rule_id: str, updates: dict[str, object]) -> dict[str, object]:
    _request("put", console, _site_path(console.site, f"rest/portforward/{rule_id}"), updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}


def create_port_forward(console: Console, rule: dict[str, object]) -> dict[str, object]:
    resp = _request("post", console, _site_path(console.site, "rest/portforward"), rule)
    created = cast(list[object], resp.json().get("data", [{}]))
    return {"status": "ok", "rule": created[0] if created else {}}


def list_traffic_rules(console: Console) -> dict[str, object]:
    data = cast(dict[str, object], _request("get", console, _v2_path(console.site, "trafficrules")).json())
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


def update_traffic_rule(console: Console, rule_id: str, updates: dict[str, object]) -> dict[str, object]:
    _request("put", console, _v2_path(console.site, f"trafficrules/{rule_id}"), updates)
    return {"status": "ok", "rule_id": rule_id, "updated": list(updates.keys())}
