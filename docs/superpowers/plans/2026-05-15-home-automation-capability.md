# Home Automation Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Home Assistant integration capability with 6 actions (list_devices, get_state, control, list_automations, trigger_automation, subscribe_events), Brain UI setup, policy defaults, and nightly test infrastructure.

**Architecture:** Dual-protocol — REST (`requests`) for 5 command/query actions, persistent WebSocket (`websocket-client`) in a daemon thread for subscribe_events and _do_monitor(). Events forwarded via Redis pub/sub to the Chalie WebSocket.

**Tech Stack:** Python 3, Flask, requests, websocket-client, Redis, YAML, SQLite, vanilla JS

**Spec:** `docs/superpowers/specs/2026-05-15-home-automation-capability-design.md`
**Research:** `/tmp/build-feature-tkt395-home-automation-1778877918.md`

---

## File Map

### New Files (Create)
| File | Responsibility |
|---|---|
| `backend/capabilities/home_capability/__init__.py` | Empty package marker |
| `backend/capabilities/home_capability/manifest.yaml` | Capability identity + fields spec |
| `backend/capabilities/home_capability/capability.py` | HomeCapability — lifecycle, tools, monitor |
| `backend/capabilities/home_capability/ha_rest_handler.py` | Stateless REST client for HA API |
| `backend/capabilities/home_capability/ha_ws_handler.py` | Persistent WebSocket client + daemon thread |
| `backend/abilities/home.py` | HomeAbility — 6-action ability |
| `chalie-nightly-test/mock_services/homeassistant/app.py` | Mock HA Flask server |
| `chalie-nightly-test/mock_services/homeassistant/Dockerfile` | Container for mock HA |
| `chalie-nightly-test/mock_services/homeassistant/requirements.txt` | Flask dependency |

### Modified Files
| File | Change |
|---|---|
| `backend/capabilities/mail_capability/manifest.yaml` | Add `fields:` key for unified rendering |
| `backend/services/policy_service.py` | Add `home.*` defaults to `_CHAT_ALLOW`, `_CHAT_ASK`, `_SUBCONSCIOUS_ALLOW` |
| `backend/services/user_message_processor.py` | Add `"home"` to `DISCOVERABLE` list |
| `backend/api/capabilities.py` | Expose `fields` from manifest in list_capabilities response |
| `frontend/brain/app.js` | Dynamic form fields from manifest `fields` key |
| `docker-compose.yml` (at `/Volumes/llm/`) | Add `homeassistant-mock` service |
| `chalie-nightly-test/scenario_runner/tools.py` | Add `homeassistant` to `ALLOWED_EXTERNAL_SERVICES` |

### Rebuilt Artifacts
| File | Trigger |
|---|---|
| `backend/abilities/assets/abilities.sqlite` | `python -m utils.build_ability_db` after creating home.py |
| `resources/pre-trained/abilities_sha.json` | Same rebuild command |

---

## Task 1: Mock HA Flask Server

**Files:**
- Create: `/Volumes/llm/chalie-nightly-test/mock_services/homeassistant/app.py`
- Create: `/Volumes/llm/chalie-nightly-test/mock_services/homeassistant/Dockerfile`
- Create: `/Volumes/llm/chalie-nightly-test/mock_services/homeassistant/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
# /Volumes/llm/chalie-nightly-test/mock_services/homeassistant/requirements.txt
flask==3.1.*
gunicorn==23.*
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
# /Volumes/llm/chalie-nightly-test/mock_services/homeassistant/Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8123
CMD ["gunicorn", "-b", "0.0.0.0:8123", "-w", "1", "--threads", "4", "app:app"]
```

- [ ] **Step 3: Create app.py — mock HA REST server**

```python
# /Volumes/llm/chalie-nightly-test/mock_services/homeassistant/app.py
"""Mock Home Assistant REST API for nightly test scenarios.

Implements the subset of HA endpoints used by HomeCapability:
  GET  /api/              — health check + auth validation
  GET  /api/states        — list all seeded entity states
  GET  /api/states/<id>   — single entity state
  POST /api/services/<domain>/<service> — record a service call
  GET  /api/config/area_registry/list   — area registry

Test-only endpoints (not part of real HA API):
  POST /api/test/seed     — seed entity states
  POST /api/test/reset    — clear all state + service call log
  GET  /api/test/service_calls — read recorded service calls
"""

import threading
from flask import Flask, jsonify, request

app = Flask(__name__)

_TEST_TOKEN = "test-token-nightly"

_lock = threading.Lock()
_entities: dict[str, dict] = {}
_service_calls: list[dict] = []
_areas: list[dict] = []


def _check_auth():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != _TEST_TOKEN:
        return jsonify({"message": "Invalid access token"}), 401
    return None


# ── Real HA endpoints ────────────────────────────────────────────────


@app.route("/api/", methods=["GET"])
def api_root():
    err = _check_auth()
    if err:
        return err
    return jsonify({"message": "API running."})


@app.route("/api/states", methods=["GET"])
def list_states():
    err = _check_auth()
    if err:
        return err
    with _lock:
        return jsonify(list(_entities.values()))


@app.route("/api/states/<path:entity_id>", methods=["GET"])
def get_state(entity_id: str):
    err = _check_auth()
    if err:
        return err
    with _lock:
        entity = _entities.get(entity_id)
    if entity is None:
        return jsonify({"message": f"Entity not found: {entity_id}"}), 404
    return jsonify(entity)


@app.route("/api/services/<domain>/<service>", methods=["POST"])
def call_service(domain: str, service: str):
    err = _check_auth()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    with _lock:
        _service_calls.append({
            "domain": domain,
            "service": service,
            "data": body,
        })
    return jsonify([])


@app.route("/api/config/area_registry/list", methods=["GET"])
def list_areas():
    err = _check_auth()
    if err:
        return err
    with _lock:
        return jsonify(list(_areas))


# ── Test-only endpoints ──────────────────────────────────────────────


@app.route("/api/test/seed", methods=["POST"])
def seed():
    body = request.get_json(force=True)
    with _lock:
        for e in body.get("entities", []):
            eid = e["entity_id"]
            _entities[eid] = {
                "entity_id": eid,
                "state": e.get("state", "unknown"),
                "attributes": e.get("attributes", {}),
                "last_changed": "2026-05-15T00:00:00+00:00",
                "last_updated": "2026-05-15T00:00:00+00:00",
            }
        for a in body.get("areas", []):
            _areas.append(a)
    return jsonify({"seeded": len(body.get("entities", []))})


@app.route("/api/test/reset", methods=["POST"])
def reset():
    with _lock:
        _entities.clear()
        _service_calls.clear()
        _areas.clear()
    return jsonify({"status": "reset"})


@app.route("/api/test/service_calls", methods=["GET"])
def get_service_calls():
    with _lock:
        return jsonify(list(_service_calls))
```

- [ ] **Step 4: Commit mock HA server**

```bash
cd /Volumes/llm/chalie-nightly-test
git add mock_services/homeassistant/
git commit -m "feat(nightly): add mock Home Assistant REST server for scenarios 136-139"
```

---

## Task 2: Docker Compose + Nightly Harness Registration

**Files:**
- Modify: `/Volumes/llm/docker-compose.yml` (after the radicale service, ~line 293)
- Modify: `/Volumes/llm/chalie-nightly-test/scenario_runner/tools.py:37-40`

- [ ] **Step 1: Add homeassistant-mock service to docker-compose.yml**

Insert after the `radicale` service block (after line 293):

```yaml
  # Mock Home Assistant — deterministic REST API stub for nightly scenarios
  homeassistant-mock:
    <<: *common-hosts
    build: /mnt/Apps/llm/chalie-nightly-test/mock_services/homeassistant
    container_name: homeassistant-mock
    ports:
      - "8123:8123"
    healthcheck:
      test: ["CMD", "curl", "-sf", "-H", "Authorization: Bearer test-token-nightly", "http://localhost:8123/api/"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 5s
```

- [ ] **Step 2: Register homeassistant in ALLOWED_EXTERNAL_SERVICES**

In `/Volumes/llm/chalie-nightly-test/scenario_runner/tools.py`, change lines 37-40:

```python
ALLOWED_EXTERNAL_SERVICES = {
    "greenmail": "http://greenmail:8080",
    "radicale": "http://radicale:5232",
    "homeassistant": "http://homeassistant-mock:8123",
}
```

- [ ] **Step 3: Commit infrastructure changes**

```bash
cd /Volumes/llm
git add docker-compose.yml
cd /Volumes/llm/chalie-nightly-test
git add scenario_runner/tools.py
git commit -m "feat(nightly): register homeassistant-mock in docker-compose and harness"
```

---

## Task 3: HA REST Handler

**Files:**
- Create: `/Volumes/llm/Chalie/backend/capabilities/home_capability/ha_rest_handler.py`

- [ ] **Step 1: Create ha_rest_handler.py**

```python
# /Volumes/llm/Chalie/backend/capabilities/home_capability/ha_rest_handler.py
"""Stateless REST client for the Home Assistant API.

Every function accepts connection parameters (url, token, verify_ssl) rather
than storing them — the caller (HomeCapability) owns credential lifecycle.
"""

import logging

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15


def _headers(token: str) -> dict:
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


def _post(url: str, path: str, token: str, verify_ssl: bool, body: dict | None = None) -> requests.Response:
    resp = requests.post(
        f"{url.rstrip('/')}{path}",
        headers=_headers(token),
        json=body or {},
        verify=verify_ssl,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp


def probe(url: str, token: str, verify_ssl: bool) -> dict:
    """Health check — GET /api/. Returns the JSON body or raises."""
    return _get(url, "/api/", token, verify_ssl).json()


def list_devices(url: str, token: str, verify_ssl: bool,
                 domain: str | None = None, area: str | None = None,
                 limit: int = 50) -> dict:
    """List HA entities, optionally filtered by domain and/or area."""
    states = _get(url, "/api/states", token, verify_ssl).json()

    if domain:
        states = [s for s in states if s["entity_id"].startswith(f"{domain}.")]

    if area:
        try:
            areas = _get(url, "/api/config/area_registry/list", token, verify_ssl).json()
            area_ids = {a["area_id"] for a in areas if area.lower() in a.get("name", "").lower()}
            if area_ids:
                states = [s for s in states if s.get("attributes", {}).get("area_id") in area_ids]
        except requests.HTTPError:
            logger.debug("Area registry not available — skipping area filter")

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


def get_state(url: str, token: str, verify_ssl: bool, entity_id: str) -> dict:
    """Get the full state of a single entity."""
    data = _get(url, f"/api/states/{entity_id}", token, verify_ssl).json()
    return {
        "entity_id": data["entity_id"],
        "state": data["state"],
        "attributes": data.get("attributes", {}),
        "last_changed": data.get("last_changed"),
        "last_updated": data.get("last_updated"),
    }


def control(url: str, token: str, verify_ssl: bool,
            entity_id: str, service: str, service_data: dict | None = None) -> dict:
    """Call a service on an entity (e.g. light/turn_on)."""
    domain = entity_id.split(".")[0]
    body = {"entity_id": entity_id}
    if service_data:
        body.update(service_data)
    _post(url, f"/api/services/{domain}/{service}", token, verify_ssl, body)
    return {"status": "ok", "entity_id": entity_id, "service": f"{domain}.{service}"}


def list_automations(url: str, token: str, verify_ssl: bool) -> dict:
    """List all automation entities."""
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


def trigger_automation(url: str, token: str, verify_ssl: bool, automation_id: str) -> dict:
    """Manually trigger an automation."""
    _post(url, "/api/services/automation/trigger", token, verify_ssl, {"entity_id": automation_id})
    return {"status": "ok", "automation_id": automation_id, "action": "triggered"}
```

- [ ] **Step 2: Commit**

```bash
cd /Volumes/llm/Chalie
git add backend/capabilities/home_capability/ha_rest_handler.py
git commit -m "feat(home): add HA REST handler — list_devices, get_state, control, automations"
```

---

## Task 4: HA WebSocket Handler

**Files:**
- Create: `/Volumes/llm/Chalie/backend/capabilities/home_capability/ha_ws_handler.py`

- [ ] **Step 1: Create ha_ws_handler.py**

```python
# /Volumes/llm/Chalie/backend/capabilities/home_capability/ha_ws_handler.py
"""Persistent WebSocket client for Home Assistant event subscriptions.

Runs in a dedicated daemon thread. Subscribes to state_changed events and
forwards matching entity changes to Redis output:events for the Chalie
WebSocket to pick up.
"""

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

_RECONNECT_BASE = 5
_RECONNECT_MAX = 60


class HaWebSocketHandler:
    """Manages a persistent WebSocket connection to Home Assistant."""

    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connected = False
        self._url: str = ""
        self._token: str = ""
        self._msg_id = 0

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, ws_url: str, token: str) -> None:
        """Start the WebSocket listener thread (idempotent)."""
        if self.is_alive:
            return
        self._url = ws_url
        self._token = token
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ha-ws-listener",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to exit."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._connected = False

    def subscribe(self, entity_id: str) -> None:
        with self._lock:
            self._subscriptions.add(entity_id)

    def unsubscribe(self, entity_id: str) -> None:
        with self._lock:
            self._subscriptions.discard(entity_id)

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _run_loop(self) -> None:
        """Reconnect loop — runs until stop is signalled."""
        backoff = _RECONNECT_BASE
        while not self._stop.is_set():
            try:
                self._connect_and_listen()
                backoff = _RECONNECT_BASE
            except Exception as exc:
                logger.warning("[ha-ws] connection error: %s — retrying in %ds", exc, backoff)
                self._connected = False
                if self._stop.wait(timeout=backoff):
                    break
                backoff = min(backoff * 2, _RECONNECT_MAX)

    def _connect_and_listen(self) -> None:
        import websocket as ws_lib

        sock = ws_lib.create_connection(self._url, timeout=10)
        try:
            auth_req = json.loads(sock.recv())
            if auth_req.get("type") != "auth_required":
                raise ConnectionError(f"Unexpected HA message: {auth_req.get('type')}")

            sock.send(json.dumps({"type": "auth", "access_token": self._token}))
            auth_resp = json.loads(sock.recv())
            if auth_resp.get("type") != "auth_ok":
                raise ConnectionError(f"HA auth failed: {auth_resp.get('message', 'unknown')}")

            sock.send(json.dumps({
                "id": self._next_id(),
                "type": "subscribe_events",
                "event_type": "state_changed",
            }))
            sub_resp = json.loads(sock.recv())
            if not sub_resp.get("success"):
                raise ConnectionError("Failed to subscribe to state_changed events")

            self._connected = True
            logger.info("[ha-ws] connected and subscribed to state_changed")

            sock.settimeout(30)
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except ws_lib.WebSocketTimeoutException:
                    sock.send(json.dumps({"id": self._next_id(), "type": "ping"}))
                    continue

                msg = json.loads(raw)
                if msg.get("type") == "event":
                    self._handle_event(msg)

        finally:
            self._connected = False
            try:
                sock.close()
            except Exception:
                pass

    def _handle_event(self, msg: dict) -> None:
        event = msg.get("event", {})
        data = event.get("data", {})
        entity_id = data.get("entity_id", "")

        with self._lock:
            if entity_id not in self._subscriptions:
                return

        new_state = data.get("new_state", {})
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            payload = {
                "type": "home_state_changed",
                "entity_id": entity_id,
                "state": new_state.get("state"),
                "friendly_name": new_state.get("attributes", {}).get("friendly_name", entity_id),
                "attributes": new_state.get("attributes", {}),
            }
            store.publish("output:events", json.dumps(payload))
        except Exception as exc:
            logger.debug("[ha-ws] failed to publish state change: %s", exc)
```

- [ ] **Step 2: Commit**

```bash
cd /Volumes/llm/Chalie
git add backend/capabilities/home_capability/ha_ws_handler.py
git commit -m "feat(home): add HA WebSocket handler — persistent event subscriptions"
```

---

## Task 5: HomeCapability Class

**Files:**
- Create: `/Volumes/llm/Chalie/backend/capabilities/home_capability/__init__.py`
- Create: `/Volumes/llm/Chalie/backend/capabilities/home_capability/manifest.yaml`
- Create: `/Volumes/llm/Chalie/backend/capabilities/home_capability/capability.py`

- [ ] **Step 1: Create __init__.py (empty)**

```python
# /Volumes/llm/Chalie/backend/capabilities/home_capability/__init__.py
```

- [ ] **Step 2: Create manifest.yaml**

```yaml
# /Volumes/llm/Chalie/backend/capabilities/home_capability/manifest.yaml
id: home
name: Home
version: 1.0.0
entry_class: HomeCapability
description: Control smart home devices and automations via Home Assistant.
providers:
  - local
  - cloud
fields:
  - name: url
    label: Home Assistant URL
    type: text
    placeholder: "http://homeassistant.local:8123"
    required: true
  - name: token
    label: Long-Lived Access Token
    type: password
    required: true
  - name: verify_ssl
    label: Verify SSL
    type: checkbox
    default: true
```

- [ ] **Step 3: Create capability.py**

```python
# /Volumes/llm/Chalie/backend/capabilities/home_capability/capability.py
"""HomeCapability — Home Assistant integration via REST + WebSocket.

REST for commands/queries (list_devices, get_state, control, list_automations,
trigger_automation). Persistent WebSocket for subscribe_events and efficient
_do_monitor() liveness checks.

Credential storage: home:url, home:token, home:verify_ssl (encrypted via
VaultService in tool_configs).
"""

from __future__ import annotations

import json
import logging
import pathlib

import yaml

from capabilities.base import AbstractCapability
from capabilities.home_capability import ha_rest_handler as rest
from capabilities.home_capability.ha_ws_handler import HaWebSocketHandler

logger = logging.getLogger(__name__)

_MANIFEST_PATH = pathlib.Path(__file__).parent / "manifest.yaml"

_K_URL = "home:url"
_K_TOKEN = "home:token"
_K_VERIFY_SSL = "home:verify_ssl"


class HomeCapability(AbstractCapability):

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._ws_handler = HaWebSocketHandler()
        self._url: str = ""
        self._token: str = ""
        self._verify_ssl: bool = True

    # ── Identity ─────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "home"

    def get_manifest(self) -> dict:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH) as f:
                self._manifest_cache = yaml.safe_load(f)
        return self._manifest_cache

    # ── Lifecycle ────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        url = (credentials.get("url") or "").strip().rstrip("/")
        token = (credentials.get("token") or "").strip()
        if not url:
            raise ValueError("Home Assistant URL is required")
        if not token:
            raise ValueError("Long-Lived Access Token is required")

        verify_ssl = credentials.get("verify_ssl", True)
        if isinstance(verify_ssl, str):
            verify_ssl = verify_ssl.lower() not in ("0", "false", "no")

        rest.probe(url, token, verify_ssl)

        self.store_credential(_K_URL, url)
        self.store_credential(_K_TOKEN, token)
        self.store_credential(_K_VERIFY_SSL, "1" if verify_ssl else "0")

        self._url = url
        self._token = token
        self._verify_ssl = verify_ssl
        self.connect()

    def connect(self) -> bool:
        url = self.load_credential(_K_URL)
        token = self.load_credential(_K_TOKEN)
        ssl_raw = self.load_credential(_K_VERIFY_SSL)
        if not url or not token:
            self._connected = False
            return False

        self._url = url
        self._token = token
        self._verify_ssl = ssl_raw != "0"

        try:
            rest.probe(self._url, self._token, self._verify_ssl)
            self._connected = True
        except Exception as exc:
            logger.warning("[home] connect probe failed: %s", exc)
            self._connected = False
        return self._connected

    def disconnect(self) -> None:
        self._connected = False
        self._ws_handler.stop()
        self.delete_credentials()

    # ── Cognitive pipeline ───────────────────────────────────────────

    def ingest(self) -> list:
        if not self.is_connected():
            return []
        try:
            return rest.list_devices(self._url, self._token, self._verify_ssl, limit=200).get("devices", [])
        except Exception as exc:
            logger.warning("[home] ingest failed: %s", exc)
            return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        if self._ws_handler.is_alive:
            return
        rest.probe(self._url, self._token, self._verify_ssl)

    def act(self, action: str, params: dict) -> dict:
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"error": f"Unknown action: {action}"}
        return handler(topic="", params=params)

    # ── Tools ────────────────────────────────────────────────────────

    def get_tools(self) -> list:
        cap = self

        def _check() -> dict | None:
            if not cap.is_connected():
                return {"error": "Home capability not connected. Configure it in the Brain dashboard."}
            return None

        def _list_devices(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            return rest.list_devices(
                cap._url, cap._token, cap._verify_ssl,
                domain=params.get("domain"),
                area=params.get("area"),
                limit=int(params.get("limit", 50)),
            )

        def _get_state(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            eid = params.get("entity_id")
            if not eid:
                return {"error": "entity_id is required"}
            return rest.get_state(cap._url, cap._token, cap._verify_ssl, eid)

        def _control(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            eid = params.get("entity_id")
            svc = params.get("service")
            if not eid or not svc:
                return {"error": "entity_id and service are required"}
            return rest.control(
                cap._url, cap._token, cap._verify_ssl,
                eid, svc, params.get("service_data"),
            )

        def _list_automations(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            return rest.list_automations(cap._url, cap._token, cap._verify_ssl)

        def _trigger_automation(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            aid = params.get("automation_id")
            if not aid:
                return {"error": "automation_id is required"}
            return rest.trigger_automation(cap._url, cap._token, cap._verify_ssl, aid)

        def _subscribe_events(topic, params, config=None, telemetry=None) -> dict:
            err = _check()
            if err:
                return err
            eid = params.get("entity_id")
            if not eid:
                return {"error": "entity_id is required"}
            ws_url = cap._url.replace("http://", "ws://").replace("https://", "wss://")
            ws_url = f"{ws_url}/api/websocket"
            cap._ws_handler.start(ws_url, cap._token)
            cap._ws_handler.subscribe(eid)
            return {"status": "subscribed", "entity_id": eid}

        return [
            {"name": "list_devices", "handler": _list_devices, "timeout": 30,
             "description": "List smart home entities", "parameters": {}},
            {"name": "get_state", "handler": _get_state, "timeout": 15,
             "description": "Get entity state", "parameters": {}},
            {"name": "control", "handler": _control, "timeout": 15,
             "description": "Call a service on an entity", "parameters": {}},
            {"name": "list_automations", "handler": _list_automations, "timeout": 30,
             "description": "List automations", "parameters": {}},
            {"name": "trigger_automation", "handler": _trigger_automation, "timeout": 15,
             "description": "Trigger an automation", "parameters": {}},
            {"name": "subscribe_events", "handler": _subscribe_events, "timeout": 15,
             "description": "Subscribe to entity state changes", "parameters": {}},
        ]
```

- [ ] **Step 4: Commit capability**

```bash
cd /Volumes/llm/Chalie
git add backend/capabilities/home_capability/
git commit -m "feat(home): add HomeCapability — REST + WebSocket lifecycle, tools, manifest"
```

---

## Task 6: HomeAbility

**Files:**
- Create: `/Volumes/llm/Chalie/backend/abilities/home.py`

- [ ] **Step 1: Create home.py**

```python
# /Volumes/llm/Chalie/backend/abilities/home.py
"""HomeAbility — control smart home devices and automations via Home Assistant.

Delegates to HomeCapability's tool handlers. No rich-media — uses skill tag
wrapper like email.py.
"""

import json
import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class HomeAbility(Ability):
    NAME = "home"
    SUMMARY = (
        "Control smart home devices and automations via the connected "
        "Home Assistant instance. Available when the user asks about "
        "devices, lights, climate, sensors, or automations."
    )
    EXAMPLES = [
        "turn on the living room light",
        "what's the temperature in the bedroom",
        "is the front door locked",
        "turn off all lights downstairs",
        "what automations do I have",
        "trigger the good morning routine",
        "what devices are currently on",
        "set the thermostat to 21 degrees",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_devices", "get_state", "control",
                    "list_automations", "trigger_automation",
                    "subscribe_events",
                ],
                "description": (
                    "list_devices — list entities, optionally filtered by domain or area. "
                    "get_state — get current state of one entity by entity_id. "
                    "control — call a service on an entity (turn_on, turn_off, set_temperature, etc.). "
                    "list_automations — list all automations with their enabled state. "
                    "trigger_automation — manually trigger an automation by automation_id. "
                    "subscribe_events — subscribe to real-time state changes for an entity."
                ),
            },
            "entity_id": {
                "type": "string",
                "description": "HA entity ID, e.g. 'light.living_room'.",
            },
            "domain": {
                "type": "string",
                "description": "list_devices: filter by HA domain (light, switch, sensor, climate, lock, etc.).",
            },
            "area": {
                "type": "string",
                "description": "list_devices: filter by area name.",
            },
            "service": {
                "type": "string",
                "description": "control: service to call, e.g. 'turn_on', 'turn_off', 'toggle'.",
            },
            "service_data": {
                "type": "object",
                "description": "control: extra service data (brightness, temperature, etc.).",
            },
            "automation_id": {
                "type": "string",
                "description": "trigger_automation: automation entity_id.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 30

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_devices").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("home")

        if cap is None or not cap.is_connected():
            result = {
                "status": "error",
                "error": "Home capability not connected. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("home", json.dumps(result), action=action)}

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            result = {"status": "error", "error": f"Unknown home action: {action}"}
            return {"text": _skill_tag("home", json.dumps(result), action=action)}

        try:
            action_params = {k: v for k, v in params.items() if not k.startswith("_") and k != "action"}
            raw = handler(topic="", params=action_params, telemetry=telemetry)
            result = raw if isinstance(raw, dict) else {"status": "ok", "data": raw}
        except Exception as exc:
            logger.error("[HOME] action=%s failed: %s", action, exc, exc_info=True)
            result = {"status": "error", "error": str(exc)}

        return {"text": _skill_tag("home", json.dumps(result), action=action)}
```

- [ ] **Step 2: Commit**

```bash
cd /Volumes/llm/Chalie
git add backend/abilities/home.py
git commit -m "feat(home): add HomeAbility — 6 actions, skill-tag output"
```

---

## Task 7: Policy Defaults + Tool Classification

**Files:**
- Modify: `/Volumes/llm/Chalie/backend/services/policy_service.py:37-103`
- Modify: `/Volumes/llm/Chalie/backend/services/user_message_processor.py:61-72`

- [ ] **Step 1: Add home.* entries to _CHAT_ALLOW**

In `policy_service.py`, add to the `_CHAT_ALLOW` dict (insert after the existing `find_tools` line, alphabetically):

```python
    "home.get_state": "allow",
    "home.list_automations": "allow",
    "home.list_devices": "allow",
    "home.subscribe_events": "allow",
```

- [ ] **Step 2: Add home.* entries to _CHAT_ASK**

In `policy_service.py`, add to the `_CHAT_ASK` dict (insert alphabetically after `email.*`):

```python
    "home.control": "ask",
    "home.trigger_automation": "ask",
```

- [ ] **Step 3: Add home.* entries to _SUBCONSCIOUS_ALLOW**

In `policy_service.py`, add to the `_SUBCONSCIOUS_ALLOW` dict (insert after `find_tools`):

```python
    "home.get_state": "allow",
    "home.list_automations": "allow",
    "home.list_devices": "allow",
```

- [ ] **Step 4: Add "home" to DISCOVERABLE**

In `user_message_processor.py`, add `"home"` to the `DISCOVERABLE` list (insert alphabetically after `"email"`):

```python
    DISCOVERABLE: list[str] = [
        "browser",
        "calendar",
        "code_eval",
        "contacts",
        "email",
        "home",
        "news",
        "programming_docs_search",
        "search",
        "subagent",
        "weather",
    ]
```

- [ ] **Step 5: Commit**

```bash
cd /Volumes/llm/Chalie
git add backend/services/policy_service.py backend/services/user_message_processor.py
git commit -m "feat(home): add policy defaults and DISCOVERABLE classification"
```

---

## Task 8: Brain UI — Dynamic Manifest Fields

**Files:**
- Modify: `/Volumes/llm/Chalie/backend/api/capabilities.py:106-119`
- Modify: `/Volumes/llm/Chalie/frontend/brain/app.js:2977-3146`
- Modify: `/Volumes/llm/Chalie/backend/capabilities/mail_capability/manifest.yaml`

- [ ] **Step 1: Expose manifest fields in capabilities API**

In `backend/api/capabilities.py`, in the `list_capabilities()` function, add `"fields"` to the result dict (line ~117):

```python
            result.append({
                "id": cap_id,
                "name": manifest.get("name", cap_id),
                "version": manifest.get("version", ""),
                "connected": cap.is_connected(),
                "last_sync_at": _get_last_sync_at(cap_id),
                "providers": manifest.get("providers", []),
                "fields": manifest.get("fields"),
            })
```

- [ ] **Step 2: Add fields to mail_capability manifest**

Replace the entire `/Volumes/llm/Chalie/backend/capabilities/mail_capability/manifest.yaml`:

```yaml
id: mail
name: Mail
version: 1.0.0
entry_class: MailCapability
description: Unified email, calendar, and contacts via IMAP/SMTP, CalDAV, and CardDAV.
providers:
  - google
  - apple
  - yahoo
  - outlook
fields:
  - name: email
    label: Email Address
    type: text
    placeholder: "you@example.com"
    required: true
  - name: password
    label: App Password
    type: password
    required: true
```

- [ ] **Step 3: Update openCapSetup() for dynamic fields**

Replace the `openCapSetup` function in `frontend/brain/app.js` (lines 3051-3075) with:

```javascript
function openCapSetup(capId) {
    const cap = capabilitiesData.find(c => c.id === capId);
    const overlay = document.getElementById('capSetupOverlay');
    document.getElementById('capSetupId').value = capId;
    document.getElementById('capSetupTitle').textContent = `Connect ${cap ? cap.name : capId}`;

    const formBody = document.getElementById('capSetupFormBody');

    if (cap && cap.fields) {
        // Dynamic fields from manifest
        formBody.innerHTML = cap.fields.map(f => {
            if (f.type === 'checkbox') {
                return `<div class="form-group form-group-checkbox">
                    <label><input type="checkbox" name="${escapeHtml(f.name)}"
                        ${f.default !== false ? 'checked' : ''}> ${escapeHtml(f.label)}</label>
                </div>`;
            }
            const inputType = f.type === 'password' ? 'password' : 'text';
            return `<div class="form-group">
                <label>${escapeHtml(f.label)}</label>
                <input type="${inputType}" name="${escapeHtml(f.name)}"
                    placeholder="${escapeHtml(f.placeholder || '')}"
                    ${f.required ? 'required' : ''}>
            </div>`;
        }).join('');
    } else {
        // Fallback: legacy username/password form
        formBody.innerHTML = `
            <div class="form-group">
                <label>Provider</label>
                <select id="capProvider" required>
                    <option value="">Select provider...</option>
                    ${(cap && cap.providers || []).map(p =>
                        `<option value="${escapeHtml(p)}">${escapeHtml(p.charAt(0).toUpperCase() + p.slice(1))}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="form-group">
                <label>Username</label>
                <input type="text" id="capUsername" required>
            </div>
            <div class="form-group">
                <label>App Password</label>
                <input type="password" id="capPassword" required>
                <small id="capPasswordHint"></small>
            </div>
            <div class="form-group hidden" id="capServerUrlGroup">
                <label>Server URL</label>
                <input type="text" id="capServerUrl" placeholder="https://...">
            </div>`;
    }

    overlay.classList.remove('hidden');
}
```

- [ ] **Step 4: Update form submit handler for dynamic fields**

Replace the submit event listener (lines 3102-3146) with:

```javascript
document.getElementById('capSetupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const capId = document.getElementById('capSetupId').value;
    const cap = capabilitiesData.find(c => c.id === capId);
    const btn = document.getElementById('capSetupSubmit');

    let body;
    if (cap && cap.fields) {
        // Collect dynamic field values
        body = {};
        for (const f of cap.fields) {
            const el = document.querySelector(`#capSetupFormBody [name="${f.name}"]`);
            if (!el) continue;
            if (f.type === 'checkbox') {
                body[f.name] = el.checked;
            } else {
                const val = el.value.trim();
                if (f.required && !val) {
                    showToast(`${f.label} is required`, 'error');
                    return;
                }
                body[f.name] = val;
            }
        }
    } else {
        // Legacy form
        const provider = document.getElementById('capProvider').value;
        const username = document.getElementById('capUsername').value.trim();
        const password = document.getElementById('capPassword').value;
        const serverUrl = document.getElementById('capServerUrl').value.trim();
        if (!provider) { showToast('Select a provider', 'error'); return; }
        if (!username) { showToast('Username is required', 'error'); return; }
        if (!password) { showToast('App password is required', 'error'); return; }
        if (_SELF_HOSTED_PROVIDERS.has(provider) && !serverUrl) {
            showToast('Server URL is required for self-hosted providers', 'error');
            return;
        }
        body = { provider, username, password };
        if (serverUrl) body.server_url = serverUrl;
    }

    btn.disabled = true;
    btn.textContent = 'Connecting...';

    try {
        const res = await apiFetch(`/api/capabilities/${capId}/setup`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
        if (res.status === 401) { window.location.replace('/login/?next=/brain/'); return; }
        const data = await res.json();
        if (res.ok) {
            document.getElementById('capSetupOverlay').classList.add('hidden');
            showToast('Connected successfully', 'success');
            await loadCapabilities();
        } else {
            showToast(data.error || 'Connection failed', 'error');
        }
    } catch (err) {
        console.warn('[brain/app] capability setup failed:', err);
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Connect';
    }
});
```

- [ ] **Step 5: Update the HTML form to use a dynamic body container**

In `frontend/brain/index.html` (the capabilities setup modal), replace the hardcoded form fields between the title and submit button with a single container div:

```html
<div id="capSetupFormBody"></div>
```

The `openCapSetup()` function will populate this div dynamically.

- [ ] **Step 6: Commit**

```bash
cd /Volumes/llm/Chalie
git add backend/api/capabilities.py backend/capabilities/mail_capability/manifest.yaml frontend/brain/app.js frontend/brain/index.html
git commit -m "feat(home): dynamic manifest fields for Brain UI capability setup"
```

---

## Task 9: Ability DB Rebuild + Unit Tests

**Files:**
- Rebuild: `backend/abilities/assets/abilities.sqlite`
- Rebuild: `resources/pre-trained/abilities_sha.json`

- [ ] **Step 1: Rebuild ability database**

```bash
cd /Volumes/llm/Chalie/backend && python -m utils.build_ability_db
```

Expected: exits 0, prints ability count including `home`.

- [ ] **Step 2: Verify home ability is indexed**

```bash
cd /Volumes/llm/Chalie/backend && python -c "
import sqlite3
db = sqlite3.connect('abilities/assets/abilities.sqlite')
rows = db.execute('SELECT name FROM abilities WHERE name = ?', ('home',)).fetchall()
print('Found' if rows else 'MISSING')
db.close()
"
```

Expected: `Found`

- [ ] **Step 3: Run unit tests**

```bash
cd /Volumes/llm/Chalie/backend && pytest -m unit -q
```

Expected: all pass, 0 failures.

- [ ] **Step 4: Commit rebuilt artifacts**

```bash
cd /Volumes/llm/Chalie
git add backend/abilities/assets/abilities.sqlite resources/pre-trained/abilities_sha.json
git commit -m "chore: rebuild abilities.sqlite with home ability"
```

---

## Task 10: Final Integration Commit

- [ ] **Step 1: Verify all files are staged**

```bash
cd /Volumes/llm/Chalie && git status
```

- [ ] **Step 2: Run unit tests one final time**

```bash
cd /Volumes/llm/Chalie/backend && pytest -m unit -q
```

- [ ] **Step 3: Run linters**

```bash
cd /Volumes/llm/Chalie/backend && python -m ruff check .
```

- [ ] **Step 4: Fix any lint issues, then commit**

If tasks 1-9 were committed individually, this is a no-op. Otherwise:

```bash
cd /Volumes/llm/Chalie
git add -A
git commit -m "feat(home): Home Automation capability — HA REST + WebSocket, 6 actions, Brain UI, policy defaults

TKT-395: Adds HomeCapability (REST + WebSocket), HomeAbility (list_devices,
get_state, control, list_automations, trigger_automation, subscribe_events),
dynamic manifest fields for Brain UI, policy defaults, mock HA server for
nightly tests, and Docker Compose service."
```

- [ ] **Step 5: Restart chalie-dev**

```bash
ssh grck.lan 'sudo docker restart chalie-dev'
```
