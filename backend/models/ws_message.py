"""Transient WS wire-model base for every frame that crosses the socket.

Rule 8 / §4.1: a websocket message is a data-model just like a persisted
row, except it never touches disk. So this base is deliberately NOT a
``Model`` subclass — it has no ``save``/``delete`` and no query entry. It
shares the JSON step (``to_json``/``_json_default``) with ``Model`` through
their common :class:`~models.serializable.Serializable` ancestor — one
wire-encoding source of truth (Essential 8) — and overrides only ``to_dict``,
because a frame projects every public field it was handed rather than a fixed
column set. It holds no ``mp``, imports no service, opens no connection.

``TurnSignal`` subclasses this — a genuinely-transient emission with no
persisted counterpart (§3.5). Persisted models emit their own ``to_json``
and are not routed through here.
"""

from __future__ import annotations

from models.serializable import Serializable


class WsMessage(Serializable):
    """Base transient WS frame: kwargs field storage + JSON projection."""

    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, object]:
        """Project every public field (name not starting with ``_``) to a dict."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if not name.startswith("_")
        }
