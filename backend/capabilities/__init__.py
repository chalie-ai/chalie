"""
Capability framework — discovery, loading, and management of capability plugins.

Each capability lives in its own sub-package under ``capabilities/`` and must
provide a ``manifest.yaml`` file plus a ``capability.py`` module that exposes a
concrete subclass of :class:`capabilities.base.AbstractCapability`.

Public API
----------
- :func:`load_capabilities` — scan the filesystem and return all discovered
  capability instances keyed by their ``id``.
"""

import importlib
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_capabilities() -> dict:
    """Discover, load, and return all capability plugins found under this package.

    Discovery algorithm:

    1. Scan every direct sub-directory of the ``capabilities/`` package directory
       for a ``manifest.yaml`` file.
    2. Parse each manifest with :func:`yaml.safe_load`.
    3. Use the ``entry_class`` field from the manifest to import the concrete
       class from ``capabilities.<subdir>.capability``.
    4. Instantiate the class (no constructor arguments).
    5. Return a :class:`dict` keyed by the manifest ``id`` field.

    Each capability is loaded inside a ``try/except`` block so that a broken or
    misconfigured capability never prevents healthy ones from loading.

    The function is *idempotent* — calling it multiple times is safe; each call
    performs a fresh filesystem scan and returns a new dict of fresh instances.

    Returns:
        dict[str, AbstractCapability]: Mapping of capability ``id`` →
        instantiated capability object.  Returns an empty dict if no
        ``manifest.yaml`` files are found or all loads fail.

    Examples:
        >>> from capabilities import load_capabilities
        >>> caps = load_capabilities()
        >>> "caldav" in caps  # True once caldav_capability/ exists
        False
    """
    capabilities_dir = Path(__file__).parent
    discovered: dict = {}

    for subdir in sorted(capabilities_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('_'):
            continue

        manifest_path = subdir / "manifest.yaml"
        if not manifest_path.exists():
            continue

        capability_id = subdir.name  # fallback before manifest is parsed
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = yaml.safe_load(fh)

            capability_id = manifest.get("id", subdir.name)
            entry_class_name = manifest["entry_class"]

            # e.g. "capabilities.caldav_capability.capability"
            module_path = f"capabilities.{subdir.name}.capability"
            module = importlib.import_module(module_path)
            cls = getattr(module, entry_class_name)
            instance = cls()

            discovered[capability_id] = instance
            logger.info(
                "Loaded capability '%s' from %s (class=%s)",
                capability_id,
                subdir.name,
                entry_class_name,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load capability '%s' from %s: %s",
                capability_id,
                subdir.name,
                exc,
                exc_info=True,
            )

    logger.info(
        "Capability discovery complete: %d capability(ies) loaded — %s",
        len(discovered),
        list(discovered.keys()) if discovered else "none",
    )
    return discovered
