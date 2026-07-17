"""CapabilityContract — the full contract every capability plugin satisfies.

Declares the complete method surface of the capability family:

- ``get_id`` / ``get_manifest`` / ``configure`` / ``connect`` / ``disconnect`` /
  ``ingest`` / ``understand`` / ``_do_monitor`` / ``get_tools`` — **required**.
  Each concrete capability supplies them (enforced by
  :class:`capabilities.base.AbstractCapability`'s ``@abstractmethod``).
- ``monitor`` / ``store_credential`` / ``load_credential`` /
  ``delete_credentials`` / ``is_connected`` — **shared**. Supplied by the
  ``AbstractCapability`` ABC base, which every capability inherits.

The ABC contributes the credential helpers + connection state and leaves the
four-phase cognitive pipeline methods abstract; concrete capabilities inherit
the helpers and MUST supply the required ones.
"""

from __future__ import annotations

from typing import Protocol


class CapabilityContract(Protocol):
    """Anything that implements the four-phase capability pipeline."""

    def get_id(self) -> str: ...

    def get_manifest(self) -> dict[str, object]: ...

    def configure(self, credentials: dict[str, object]) -> None: ...

    def connect(self) -> bool: ...

    def disconnect(self) -> None: ...

    def ingest(self) -> list[object]: ...

    def understand(self, items: list[object]) -> list[object]: ...

    def _do_monitor(self) -> None: ...

    def get_tools(self) -> list[dict[str, object]]: ...

    def monitor(self) -> None: ...

    def store_credential(self, key: str, value: str) -> None: ...

    def load_credential(self, key: str) -> "str | None": ...

    def delete_credentials(self) -> None: ...

    def is_connected(self) -> bool: ...
