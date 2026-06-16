"""Runtime configuration — process-local key-value store.

Populated by run.py from CLI args. Modules needing runtime values (port,
host) import this instead of reading env vars.
"""

_config = {}


def set(cfg: dict):
    _config.update(cfg)


def get(key: str, default=None):
    return _config.get(key, default)
