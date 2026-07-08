"""``SystemMemoryRow`` — the searchable ``system`` vertical of ``data_graph``:
knowledge memories the user/agent saves through the memory tool (``source``
prefixed ``skill:memory:store:``) about how Chalie operates — rules, decisions,
analysis. Recalled by semantic search like every other memory vertical, so it
keeps the ``"system"`` kind and inherits the shared FTS/vec write-sync unchanged.
Adds nothing beyond the kind: the exact-key store/forget/supersede lifecycle
lives on :class:`~models.exact_key.ExactKeyRow`."""

from __future__ import annotations

from typing import ClassVar

from models.exact_key import ExactKeyRow


class SystemMemoryRow(ExactKeyRow):
    KIND: ClassVar[str] = "system"
