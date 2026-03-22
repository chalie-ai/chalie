"""
Semantic Retrieval Service — backward compatibility shim.

All functionality has been merged into :mod:`services.semantic_service`.
This module re-exports ``SemanticService`` as ``SemanticRetrievalService``
so existing callers continue to work.
"""

from services.semantic_service import SemanticService as SemanticRetrievalService  # noqa: F401

__all__ = ['SemanticRetrievalService']
