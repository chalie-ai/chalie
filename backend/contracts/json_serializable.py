"""JsonSerializable — the wire contract every WebSocket frame satisfies.

A :class:`~models.serializable.Serializable` (persisted rows and transient
WS frames alike) implements ``to_json``; this Protocol names that contract
alone, so :meth:`Websocket.broadcast` can type-hint against the narrow
interface it needs without coupling to any one model hierarchy. It is
structural — every existing model conforms automatically, no inheritance
required.

Lives in ``contracts`` (not ``models``) because it is a pure interface: it
declares the shape, it does not store data.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JsonSerializable(Protocol):
    """Anything that can project itself to a JSON string for the wire."""

    def to_json(self) -> str: ...
