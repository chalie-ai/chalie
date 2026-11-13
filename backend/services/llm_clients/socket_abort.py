"""Let another thread end an HTTP call that is blocked on a socket read.

A thread blocked reading a socket cannot be interrupted, and closing the HTTP
client does not wake it; shutting the socket down does — the blocked read
returns at once and the SDK raises a connection error. Sockets are captured
through httpcore's ``trace`` request extension. The TLS handshake is the one
stretch where no captured socket can be shut down — the wrap detaches the plain
socket before the handshake reads, and the TLS socket is only handed over once
the handshake is done — so a duplicate of its descriptor is held for exactly
that stretch. Once aborted, any further request or connection attempt raises,
so an abort that lands before the connection exists still ends the call.
"""
import logging
import socket
import threading
from collections.abc import Callable, Mapping, MutableMapping
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_CONNECTED = ("connection.connect_tcp.complete", "connection.start_tls.complete")
_TLS_STARTED = "connection.start_tls.started"
_TLS_ENDED = ("connection.start_tls.complete", "connection.start_tls.failed")


class _Request(Protocol):
    @property
    def extensions(self) -> MutableMapping[str, object]: ...


@runtime_checkable
class _Stream(Protocol):
    def get_extra_info(self, info: str) -> object: ...


class SocketAbort:
    """Lets another thread end an HTTP call blocked on a socket read.

    One instance serves one client, and a client makes one call at a time, so
    the socket recorded last is the one a TLS handshake is wrapping."""

    def __init__(self) -> None:
        """Initialize the lock, the captured sockets, and the abort flag."""
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self._handshake: socket.socket | None = None
        self._aborted = False

    def hooks(self) -> dict[str, list[Callable[[_Request], None]]]:
        """Return the hook callbacks that gate new requests and capture sockets."""
        return {"request": [self._on_request]}

    def _on_request(self, request: _Request) -> None:
        with self._lock:
            if self._aborted:
                raise ConnectionAbortedError("call aborted")
        request.extensions["trace"] = self._trace

    def _trace(self, name: str, info: Mapping[str, object]) -> None:
        with self._lock:
            if name in _TLS_ENDED and self._handshake is not None:
                self._handshake.close()
                self._handshake = None
            if self._aborted:
                raise ConnectionAbortedError("call aborted")
            if name == _TLS_STARTED and self._sockets and self._sockets[-1].fileno() != -1:
                plain = self._sockets[-1]
                self._handshake = socket.fromfd(plain.fileno(), plain.family, plain.type)
            if name in _CONNECTED:
                stream = info.get("return_value")
                if isinstance(stream, _Stream):
                    sock = stream.get_extra_info("socket")
                    if isinstance(sock, socket.socket):
                        self._sockets.append(sock)

    def abort(self) -> None:
        """Set the abort flag and shut down every captured socket."""
        with self._lock:
            self._aborted = True
            # A detached plain socket reads -1: its TLS socket or the handshake
            # duplicate stands in for it.
            live = [sock for sock in self._sockets if sock.fileno() != -1]
            if self._handshake is not None:
                live.append(self._handshake)
            for sock in live:
                try:
                    socket.socket.shutdown(sock, socket.SHUT_RDWR)  # unbound: skips SSLSocket.shutdown, which unwraps TLS and is not thread-safe
                except OSError as exc:
                    logger.debug("socket shutdown on abort failed: %s", exc)
