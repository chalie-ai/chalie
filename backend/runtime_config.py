"""
Runtime configuration — process-local key-value store.

Populated by run.py from CLI args. Any module that needs runtime values
(port, host) imports this module instead of reading env vars.

    import runtime_config
    port = runtime_config.get("port", 31025)
"""

_config = {}


def set(cfg: dict):
    """Merge a dict of runtime values into the config store."""
    _config.update(cfg)


def get(key: str, default=None):
    """Retrieve a runtime config value by key."""
    return _config.get(key, default)
