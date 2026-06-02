"""Capability-handler dispatch helper.

Shared by the capability-backed abilities (calendar, contacts, email, home,
ubiquiti) to invoke a loaded capability's named tool handler.  Replaces the
former ``Ability.handle()`` static helper (removed in the ACT-loop refactor,
spec §5 / Q4).
"""


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
