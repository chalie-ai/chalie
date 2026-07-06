"""The ``Decayable`` contract — the shape every off-turn decaying subsystem
satisfies so the :class:`~orchestrators.decay_engine.DecayEngine` can sequence
each one uniformly.

A Decayable performs its own maintenance off-turn: no ``MessageProcessor``, no
arguments (T2), reporting how many rows it maintained. It is a structural
(duck-typed) :class:`typing.Protocol`, so a subsystem satisfies it merely by
exposing a matching ``decay()`` — nothing needs to inherit. ``runtime_checkable``
so the engine (or a test) can ``isinstance``-guard a registration if useful.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Decayable(Protocol):
    """A subsystem that runs its own off-turn decay / GC maintenance."""

    def decay(self) -> int:
        """Run one maintenance pass, returning the number of rows maintained
        (``0`` = ran, nothing to do — distinct from a failure, which raises)."""
        ...
