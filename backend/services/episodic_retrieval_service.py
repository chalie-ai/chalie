"""
Episodic Retrieval Service — backward compatibility shim.

All functionality has been merged into :mod:`services.episodic_service`.
This module re-exports ``EpisodicService`` as ``EpisodicRetrievalService``
so existing callers continue to work.
"""

from services.episodic_service import EpisodicService as EpisodicRetrievalService  # noqa: F401

__all__ = ['EpisodicRetrievalService']
