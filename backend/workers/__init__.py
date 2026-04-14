"""
Workers package — Background worker callables.

Each worker module exposes a single callable that is launched as a
long-running background process by the service runtime.

Exported workers:
    - ``rest_api_worker``: REST API request handler worker.
"""

from .rest_api_worker import rest_api_worker

__all__ = ['rest_api_worker']
