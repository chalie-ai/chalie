"""``CapabilityAbility`` — the shared base for capability-backed tools.

``home``, ``ubiquiti``, ``contacts``, ``email`` and ``calendar`` all do the same
thing: load a named capability, refuse if it is not connected, map the model's
``action`` onto a capability handler, dispatch it, and wrap the result. That
block was copy-pasted five times and had already drifted (different error
wording, an unused rich-media path in two of them).

This base owns the flow once. A concrete capability tool declares four
``ClassVar``s and the metadata getters; the base supplies ``run()``:

* :attr:`CAPABILITY_KEY` — the key passed to ``load_capabilities().get(...)``.
* :attr:`ACTION_HANDLERS` — ``{action: capability_handler_name}``.
* :attr:`DEFAULT_ACTION` — the action assumed when the model omits one.
* :attr:`NOT_CONNECTED_HINT` — the remediation sentence for the not-connected
  error (e.g. "Configure the mail integration in the Brain dashboard.").

Errors are returned as structured :class:`ToolResult` values (``code`` +
``hint`` / ``valid``), never raised — the ACT loop surfaces them to the model.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from abilities._ability import Ability
from abilities._result import ToolResult


class CapabilityAbility(Ability, ABC):
    """Base for an ability that delegates to a connected capability handler.

    Listing ``ABC`` directly in the bases makes ``Ability.__init_subclass__``
    skip the metadata probe for this base itself; concrete subclasses (which do
    NOT list ABC) are probed as usual.
    """

    #: Capability key for ``load_capabilities().get(...)``. Subclass MUST set.
    CAPABILITY_KEY: ClassVar[str] = ""

    #: Maps the model-facing ``action`` to the capability's handler name.
    ACTION_HANDLERS: ClassVar[dict[str, str]] = {}

    #: Action assumed when the model omits ``action``.
    DEFAULT_ACTION: ClassVar[str] = ""

    #: Remediation sentence appended to the not-connected error.
    NOT_CONNECTED_HINT: ClassVar[str] = "Configure it in the Brain dashboard."

    def run(self, params: "dict[str, object]") -> ToolResult:
        action = str(params.get("action", self.DEFAULT_ACTION)).lower()

        from capabilities import load_capabilities

        cap = load_capabilities().get(self.CAPABILITY_KEY)
        if cap is None or not cap.is_connected():
            return ToolResult.err(
                f"{self.get_name().capitalize()} capability not connected.",
                code="not-connected",
                hint=self.NOT_CONNECTED_HINT,
                action=action,
            )

        handler_name = self.ACTION_HANDLERS.get(action)
        if handler_name is None:
            return ToolResult.err(
                f"Unknown {self.get_name()} action: {action}",
                code="unknown-action",
                valid=tuple(self.ACTION_HANDLERS),
                action=action,
            )

        tool_map = {t["name"]: t["handler"] for t in cap.get_tools()}
        handler = tool_map.get(handler_name)
        if handler is None:
            return ToolResult.err(
                f"Handler '{handler_name}' not available — the required protocol "
                "may not be connected.",
                code="handler-unavailable",
                action=action,
            )

        action_params = {
            k: v for k, v in params.items() if not k.startswith("_") and k != "action"
        }
        result = cast("Callable[..., dict[str, object]]", handler)(action_params, self.telemetry)
        # A handler signals failure with a truthy ``error`` key. Wrapping that in
        # ``ok`` would tag a genuine failure ``status=success`` — the model never
        # sees an error envelope, and the dispatcher's repeat-error steer (which
        # only fires on ``status=error``) can't break a retry loop. Route it to
        # ``err`` so the failure is loud and self-correctable.
        error = result.get("error") if isinstance(result, dict) else None
        if error:
            return ToolResult.err(str(error), code="capability-error", action=action)
        return ToolResult.ok(result, action=action)
