# TKT-395: Home Automation Capability — Home Assistant Integration

**Date**: 2026-05-15
**Status**: Draft
**Branch**: rc-0.7.0

---

## Overview

Add a Home Automation capability that integrates with Home Assistant via its REST and WebSocket APIs. HA bridges all major smart home platforms (Matter, Google Home, Alexa, HomeKit, Z-Wave, Zigbee, 2000+ integrations), making it the single integration point for Chalie.

## Architecture

**Dual-protocol**:
- **REST** (`requests`) for 5 command/query actions — stateless, one HTTP call per invocation
- **WebSocket** (`websocket-client`, synchronous) in a dedicated daemon thread for `subscribe_events` and efficient `_do_monitor()` state-change detection

Events from the HA WebSocket are forwarded through Redis `output:events` to the Chalie WebSocket and on to the browser — the same path used by capability health alerts.

## New Capability: `backend/capabilities/home_capability/`

### Directory Structure

```
home_capability/
    __init__.py
    manifest.yaml
    capability.py          # HomeCapability(AbstractCapability)
    ha_rest_handler.py     # REST client
    ha_ws_handler.py       # WebSocket client
```

### manifest.yaml

```yaml
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

The `fields` key is a new addition to the manifest format. The Brain UI renders these dynamically instead of the hardcoded username/password form. Existing capabilities (mail) will also get `fields` added to their manifest so the rendering path is unified.

### HomeCapability Class

Subclasses `AbstractCapability`. Credential keys: `home:url`, `home:token`, `home:verify_ssl`.

**`configure(credentials)`** — Validates URL and token fields are present. Probes `GET {url}/api/` with `Authorization: Bearer {token}`. On success, persists all credentials via `store_credential()`. On failure, raises `ValueError`. Calls `self.connect()` at the end.

**`connect()`** — Loads credentials via `load_credential()`. Probes REST `GET /api/`. Sets `self._connected`. Returns bool. Does not raise on transient failures.

**`disconnect()`** — Sets `self._connected = False`. Stops the WS handler thread if running. Calls `self.delete_credentials()`.

**`ingest()`** — `GET /api/states`. Returns list of entity state dicts. Checks `self.is_connected()` first; returns `[]` when disconnected.

**`understand(items)`** — Passthrough (`return items`). HA entity states are already structured — no LLM enrichment needed.

**`_do_monitor()`** — If the WS handler is running, piggybacks on it for liveness (the WS handler pings HA automatically). If not running, does a REST `GET /api/` health probe. State changes from the WS `state_changed` subscription are published to Redis as structured notifications.

**`act(action, params)`** — Delegates to the handler map from `get_tools()`.

**`get_tools()`** — Returns 6 tool definitions. Each handler is a closure capturing `self`, checks connection state, returns `{"error": "..."}` when disconnected.

### ha_rest_handler.py

Stateless HTTP client. All methods accept the HA base URL, token, and verify_ssl flag.

- `list_devices(url, token, verify_ssl, domain=None, area=None, limit=50)` — `GET /api/states` for entities, `GET /api/config/area_registry/list` for area resolution (cached per-session). Filters by domain prefix on entity_id and/or area name match. Caps at `limit` entities. Returns `{devices: [...], count: N, total: N}`.
- `get_state(url, token, verify_ssl, entity_id)` — `GET /api/states/{entity_id}`. Returns the full state dict.
- `control(url, token, verify_ssl, entity_id, service, service_data=None)` — `POST /api/services/{domain}/{service}` with body `{"entity_id": entity_id, **service_data}`. Domain extracted from entity_id prefix (e.g., `light.living_room` → domain `light`).
- `list_automations(url, token, verify_ssl)` — `GET /api/states` filtered to `automation.*` domain. Returns `{automations: [...], count: N}`.
- `trigger_automation(url, token, verify_ssl, automation_id)` — `POST /api/services/automation/trigger` with body `{"entity_id": automation_id}`.

All methods set `Authorization: Bearer {token}` header. All methods raise on non-2xx responses with the HA error message.

### ha_ws_handler.py

Persistent WebSocket client using `websocket-client` (synchronous) in a dedicated daemon thread.

**Connection lifecycle**:
1. Connect to `ws://{host}:{port}/api/websocket`
2. HA sends `auth_required` message
3. Client sends `{"type": "auth", "access_token": token}`
4. HA sends `auth_ok` or `auth_invalid`
5. Client subscribes to `state_changed` events: `{"type": "subscribe_events", "event_type": "state_changed"}`

**Entity subscriptions**: Maintained as a set of entity_ids. When a `state_changed` event arrives, check if `entity_id` is in the subscription set. If yes, publish to Redis `output:events` with type `home_state_changed`.

**Reconnect**: On disconnect, exponential backoff (5s → 60s max). Thread exits cleanly on `disconnect()` call (via a threading.Event stop signal).

**Thread management**: Started lazily on first `subscribe_events` call or when `_do_monitor()` runs. Stopped on `disconnect()`. The thread is a daemon thread so it doesn't block process shutdown.

## New Ability: `backend/abilities/home.py`

Follows the `email.py` pattern — capability-backed, multi-action, no rich-media.

```python
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
```

8 EXAMPLES (within the 6-8 hard constraint).

### INPUT_SCHEMA

```python
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
```

### execute() Flow

1. Load capability: `load_capabilities().get("home")`
2. Check connection: return error dict if not connected
3. Build tool map from `cap.get_tools()`
4. Dispatch to handler by `action`
5. Return via `_skill_tag("home", json.dumps(result), action=action)`

### Tool Classification

Add `"home"` to **DISCOVERABLE** list in `user_message_processor.py`. NOT to ALWAYS_AVAILABLE (TKT-411: 9 entries is the safe ceiling; promoting a 10th caused routing degradation).

## Policy Manager Integration

In `backend/services/policy_service.py`:

```python
# _CHAT_ALLOW (reads — execute without confirmation):
"home.list_devices": "allow",
"home.get_state": "allow",
"home.list_automations": "allow",
"home.subscribe_events": "allow",

# _CHAT_ASK (writes — require user confirmation):
"home.control": "ask",
"home.trigger_automation": "ask",

# _SUBCONSCIOUS_ALLOW (background worker — reads only):
"home.list_devices": "allow",
"home.get_state": "allow",
"home.list_automations": "allow",
```

`home.control` and `home.trigger_automation` are excluded from subconscious — destructive actions require explicit user confirmation.

## Brain UI — Dynamic Manifest Fields

### manifest.yaml Extension

All capabilities get a `fields` key in their manifest:

```yaml
fields:
  - name: url
    label: Home Assistant URL
    type: text           # text | password | checkbox
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

### frontend/brain/app.js Changes

`openCapSetup()` reads `manifest.fields` (if present) and renders form fields dynamically instead of the hardcoded username/password block. When `fields` is absent (backward compat during rollout), the UI falls back to the existing hardcoded username/password form. Field types map to HTML input types:
- `text` → `<input type="text">`
- `password` → `<input type="password">`
- `checkbox` → `<input type="checkbox">`

The submit handler collects field values by name and sends them as the credentials dict to `POST /api/capabilities/{cap_id}/setup`.

Existing mail capability gets `fields` added to its manifest (username, password, server_url) so the rendering is unified — no special-casing.

## Docker — Testing Infrastructure

### Real HA Instance (development testing)

In `/Volumes/llm/docker.yml`:

```yaml
homeassistant-test:
  image: homeassistant/home-assistant:stable
  container_name: homeassistant-test
  volumes:
    - /mnt/Apps/llm/homeassistant-test-config:/config
  ports:
    - "8123:8123"
  restart: unless-stopped
```

Pre-configured `configuration.yaml` with the `demo` integration enabled (creates fake lights, switches, sensors, climate, locks, media players). A long-lived access token is generated on first boot.

### Mock HA (nightly test determinism)

Lightweight Flask app at `chalie-nightly-test/mock_services/homeassistant/`:

**Endpoints**:
- `GET /api/` — `{"message": "API running."}` (validates Bearer token)
- `GET /api/states` — returns seeded entity states
- `GET /api/states/<entity_id>` — returns single entity state
- `POST /api/services/<domain>/<service>` — records the call, returns `[]`
- `GET /api/states` (filtered to `automation.*`) — returns automation entities
- `POST /api/test/seed` — seed entity states (test-only)
- `POST /api/test/reset` — clear all state (test-only)

Added to docker-compose.yml and registered in `ALLOWED_EXTERNAL_SERVICES` in the nightly harness.

## Nightly Test Scenarios

### 136 — capability-home-list-devices.yaml

**Setup**: Configure home capability against mock HA. Seed mock with entities: `light.living_room` (on), `climate.bedroom` (heat, 21C), `switch.kitchen_fan` (off), `sensor.outdoor_temperature` (18.5C), `lock.front_door` (locked).

**Conversation**: "What smart home devices do I have set up?"

**Assertions**:
- `tool_calls` row: `tool_name='home'`, params contains `action='list_devices'`
- LLM response mentions at least 3 device names from seed data

### 137 — capability-home-get-state.yaml

**Setup**: Configure home capability. Seed mock with `climate.bedroom` in state `heat`, attributes `{current_temperature: 23, temperature: 21}`.

**Conversation**: "Is the AC in the bedroom on?"

**Assertions**:
- `tool_calls` row: `tool_name='home'`, params contains `action='get_state'`, `entity_id` contains `bedroom`
- LLM response references the bedroom climate state (temperature or heat mode)

### 138 — capability-home-control.yaml

**Setup**: Patch policy `home.control` to `allow`. Configure home capability. Seed mock with `light.living_room` in `off` state.

**Conversation**: "Turn on my living room light"

**Assertions**:
- `tool_calls` row: `tool_name='home'`, params contains `action='control'`, `service='turn_on'`
- Mock HA recorded a `POST /api/services/light/turn_on` with `entity_id='light.living_room'`

**Cleanup**: Restore policy defaults.

### 139 — capability-home-automations.yaml

**Setup**: Patch policy `home.trigger_automation` to `allow`. Configure home capability. Seed mock with automations: `automation.good_morning` (enabled, "Good Morning Routine"), `automation.away_mode` (enabled, "Away Mode").

**Conversation** (2 turns):
1. "What automations have I got running?"
2. "Run the good morning routine"

**Assertions**:
- Turn 1: `tool_calls` with `action='list_automations'`, response mentions "Good Morning"
- Turn 2: `tool_calls` with `action='trigger_automation'`, `automation_id` contains `good_morning`
- Mock HA recorded `POST /api/services/automation/trigger`

**Cleanup**: Restore policy defaults.

## Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Brain UI form mismatch | HIGH | Manifest `fields` spec — dynamic rendering |
| HA entity payload overflow | MEDIUM | `list_devices` caps at 50 entities, filterable by domain/area |
| WS thread lifecycle | MEDIUM | Daemon thread with stop event; clean teardown on `disconnect()` |
| Policy gate in nightly tests | MEDIUM | `patch_config` steps for ask-gated write actions |
| Token security | MEDIUM | `store_credential()` only; never logged at INFO+ |
| Self-signed SSL | LOW | `verify_ssl` credential, default True |
| Ability DB rebuild | LOW | Part of build checklist; committed with feature |

## Scenarios Not Covered

`subscribe_events` does not have a nightly scenario in this build. The mock HA server implements REST only; WebSocket subscription testing requires a real HA instance or a WebSocket-capable mock, which is out of scope for the initial 4 scenarios. The action is implemented and manually testable against the real HA dev instance.

## Out of Scope

- Rich-media cards for device state
- Room-aware context inference
- Scheduled home automations via Chalie's scheduler
- Proactive alerts (e.g., "front door unlocked for 30 minutes")
- Energy monitoring dashboards

## Post-Build Checklist

1. `cd backend && python -m utils.build_ability_db` — rebuild ability index
2. Commit `abilities.sqlite` + `abilities_sha.json`
3. `cd backend && pytest -m unit -q` — all unit tests pass
4. Restart chalie-dev: `ssh grck.lan 'sudo docker restart chalie-dev'`
