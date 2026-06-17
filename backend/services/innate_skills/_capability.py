"""Capability-handler dispatch helper."""

def dispatch_capability_handler(handler: object, params: dict, telemetry: "dict | None") -> dict:
    """Invoke *handler* with framework keys stripped from *params*.

    Strips leading-underscore keys and the ``action`` selector, then calls
    ``handler(topic="", params=<clean>, telemetry=telemetry)``.

    Returns the handler's result dict unchanged.
    """
    action_params = {
        k: v for k, v in params.items() if not k.startswith("_") and k != "action"
    }
    return handler(topic="", params=action_params, telemetry=telemetry)  # type: ignore[operator]
