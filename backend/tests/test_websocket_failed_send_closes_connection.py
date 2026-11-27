"""Feature test: a connection the broadcast cannot write to is closed, not just forgotten.

Dropping it from the registry alone leaves it open but deaf — the browser keeps
waiting on a socket that will never deliver another frame instead of reconnecting
and catching up. Real registry, real broadcast; the subscribers implement the
registry's socket Protocol.
"""

import json
from collections.abc import Iterator

import pytest

from models.ws_message import WsMessage
from services.websocket import Websocket

pytestmark = pytest.mark.unit


class _HalfOpenClient:
    """A connection whose writes fail, as a half-open TCP socket's do."""

    def __init__(self) -> None:
        self.closed = False

    def send(self, _data: str) -> None:
        raise BrokenPipeError("peer stopped reading")

    def close(self) -> None:
        self.closed = True


class _HealthyClient:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def send(self, data: str) -> None:
        self.frames.append(json.loads(data))

    def close(self) -> None:
        """Nothing to release."""


@pytest.fixture
def clients() -> Iterator[tuple[_HalfOpenClient, _HealthyClient]]:
    broken, healthy = _HalfOpenClient(), _HealthyClient()
    Websocket._connect(broken)
    Websocket._connect(healthy)
    yield broken, healthy
    Websocket._disconnect(broken)
    Websocket._disconnect(healthy)


def test_a_failed_send_closes_that_connection_and_keeps_serving_the_rest(
    clients: tuple[_HalfOpenClient, _HealthyClient],
) -> None:
    broken, healthy = clients

    Websocket.broadcast(WsMessage(type="ping"))

    assert broken.closed, "the unwritable connection was left open, so its browser never reconnects"
    assert broken not in Websocket._connections
    assert [f["type"] for f in healthy.frames] == ["ping"]

    Websocket.broadcast(WsMessage(type="ping"))

    assert [f["type"] for f in healthy.frames] == ["ping", "ping"]
