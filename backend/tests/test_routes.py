"""The route table is the only mount point — nothing may be written and forgotten.

Paths used to be conjured by the mere existence of a file: boot walked
``api/endpoints`` and ``api/actions`` and registered whatever ``Endpoint``
subclass it found. ``api/routes.py`` replaced that with an explicit table, which
buys a single place to read every URL from — and costs the guarantee that a new
controller is reachable at all. These tests are that guarantee: an unmounted
controller fails here instead of 404-ing in production, and a duplicated mount
fails here instead of raising on the first boot that registers it.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from api.action import Action
from api.endpoint import Endpoint
from api.routes import ROUTES

pytestmark = pytest.mark.unit


def _declared_controllers() -> set[type[Endpoint]]:
    """Every controller class defined under ``api/endpoints`` and ``api/actions``.

    Matched on ``__module__`` so a class merely imported into another module
    (``Action`` itself, a shared helper) is counted once, where it is defined.
    """
    found: set[type[Endpoint]] = set()
    for subpackage_name in ("endpoints", "actions"):
        subpackage = importlib.import_module(f"api.{subpackage_name}")
        for info in pkgutil.walk_packages(subpackage.__path__, prefix=f"{subpackage.__name__}."):
            module = importlib.import_module(info.name)
            for attr in vars(module).values():
                if isinstance(attr, type) and issubclass(attr, Endpoint) and attr.__module__ == info.name:
                    found.add(attr)
    return found


def test_every_controller_is_mounted() -> None:
    """A controller absent from ROUTES has no URL — that is a bug, not a default."""
    unmounted = _declared_controllers() - {type(c) for c in ROUTES}
    assert not unmounted, (
        "controllers defined but not mounted in api/routes.py: "
        + ", ".join(sorted(f"{c.__module__}.{c.__name__}" for c in unmounted))
    )


def test_no_controller_is_mounted_twice_at_the_same_path() -> None:
    """Duplicate mounts collide on the generated Flask endpoint name at boot."""
    paths = [(c.slug, c.verb if isinstance(c, Action) else None) for c in ROUTES]
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    assert not duplicates, f"the same path is mounted more than once: {duplicates}"
