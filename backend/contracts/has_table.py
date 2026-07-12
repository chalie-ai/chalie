"""HasTable — the contract every persisted active-record model satisfies.

Declares the one method the engine (:class:`~models.model.Model`,
:class:`~models.query.Query`) needs to render SQL against a concrete table:
``get_table() -> str``. :class:`~models.model.Model` explicitly subclasses
this Protocol (rather than merely structurally matching it) so the
``@abstractmethod`` below turns into a real enforcement mechanism, mirroring
ABC semantics: a concrete model that forgets to override ``get_table()``
inherits the unimplemented abstract method, which makes it abstract in turn —
instantiating it raises ``TypeError`` naming the offending class, and mypy
flags the same instantiation statically (``Cannot instantiate abstract
class ... with abstract attribute "get_table"``). Calling ``get_table()``
directly on an abstract class (no instantiation involved, e.g. through the
class object alone) still fails loudly via the body below, rather than
silently returning ``None``.

Lives in ``contracts`` (not ``models``) because it is a pure interface: it
declares the shape, it does not store data — mirrors
:class:`~contracts.json_serializable.JsonSerializable`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class HasTable(Protocol):
    """Anything that names the SQL table it persists to."""

    @classmethod
    @abstractmethod
    def get_table(cls) -> str:
        """The table this model's rows live in. Every concrete model MUST
        override this; the base implementation only exists so a direct,
        non-instantiating call (e.g. on a class that forgot to override)
        fails loudly instead of silently returning ``None``."""
        raise NotImplementedError(f"{cls.__name__} must implement get_table()")
