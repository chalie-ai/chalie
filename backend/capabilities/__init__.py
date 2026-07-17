import importlib
import logging
from typing import TYPE_CHECKING

import yaml

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    from capabilities.base import AbstractCapability

logger = logging.getLogger(__name__)

# Singleton cache — capability instances persist for the process lifetime so
# that in-memory state (e.g. ``_connected``) survives across API calls.
_capabilities_cache: "dict[str, AbstractCapability] | None" = None


def load_capabilities() -> "dict[str, AbstractCapability]":
    global _capabilities_cache
    if _capabilities_cache is not None:
        return _capabilities_cache

    capabilities_dir = FileMapperService.get_capabilities_path()
    discovered: "dict[str, AbstractCapability]" = {}

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
            logger.exception(
                "Failed to load capability '%s' from %s: %s",
                capability_id,
                subdir.name,
                exc,
            )

    logger.info(
        "Capability discovery complete: %d capability(ies) loaded — %s",
        len(discovered),
        list(discovered.keys()) if discovered else "none",
    )
    _capabilities_cache = discovered
    return _capabilities_cache
