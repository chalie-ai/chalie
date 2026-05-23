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

import yaml

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

# Singleton cache — capability instances persist for the process lifetime so
# that in-memory state (e.g. ``_connected``) survives across API calls.
_capabilities_cache: dict | None = None


def load_capabilities() -> dict:
    """Discover, load, and return all capability plugin instances.

    On the first call, scans every direct sub-directory of ``capabilities/``
    for a ``manifest.yaml``, imports and instantiates the declared
    ``entry_class``, and caches the result.  Subsequent calls return the
    same cached instances so that in-memory state (``_connected``, etc.)
    is preserved across API requests.

    Returns:
        dict[str, AbstractCapability]: Mapping of capability ``id`` →
        instantiated capability object.  Returns an empty dict if no
        ``manifest.yaml`` files are found or all loads fail.
    """
    global _capabilities_cache
    if _capabilities_cache is not None:
        return _capabilities_cache

    capabilities_dir = FileMapperService.get_capabilities_path()
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

        except Exception as exc:
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
    _capabilities_cache = discovered
    return _capabilities_cache
