"""
Episodic Storage Service — backward compatibility shim.

All functionality has been merged into :mod:`services.episodic_service`.
This module re-exports ``EpisodicService`` as ``EpisodicStorageService``
so existing callers continue to work.
"""

from services.episodic_service import EpisodicService as EpisodicStorageService  # noqa: F401

__all__ = ['EpisodicStorageService']
