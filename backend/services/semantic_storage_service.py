"""
Semantic Storage Service — backward compatibility shim.

All functionality has been merged into :mod:`services.semantic_service`.
This module re-exports ``SemanticService`` as ``SemanticStorageService``
so existing callers continue to work.
"""

from services.semantic_service import SemanticService as SemanticStorageService  # noqa: F401
from services.semantic_service import _json_default  # noqa: F401
from services.embedding_utils import pack_embedding as _pack_embedding  # noqa: F401

__all__ = ['SemanticStorageService', '_json_default', '_pack_embedding']
