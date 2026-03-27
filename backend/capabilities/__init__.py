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
