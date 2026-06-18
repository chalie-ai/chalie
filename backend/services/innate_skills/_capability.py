"""Capability-handler dispatch helper."""

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    class _CapabilityHandler(Protocol):
        def __call__(
            self, *, topic: str, params: dict[str, object], telemetry: "dict[str, object] | None"
        ) -> dict[str, object]: ...


def dispatch_capability_handler(handler: object, params: dict[str, object], telemetry: "dict[str, object] | None") -> dict[str, object]:
    """Invoke *handler* with framework keys stripped from *params*.

    Strips leading-underscore keys and the ``action`` selector, then calls
    ``handler(topic="", params=<clean>, telemetry=telemetry)``.

    Returns the handler's result dict unchanged.
    """
    action_params = {
        k: v for k, v in params.items() if not k.startswith("_") and k != "action"
    }
    return cast("_CapabilityHandler", handler)(topic="", params=action_params, telemetry=telemetry)
