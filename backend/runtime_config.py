"""Runtime configuration — process-local key-value store.

Populated by run.py from CLI args. Modules needing runtime values (port,
host) import this instead of reading env vars.
"""

_config: dict[str, object] = {}


def set(cfg: dict[str, object]) -> None:
    _config.update(cfg)


def get(key: str, default: object = None) -> object:
    return _config.get(key, default)
