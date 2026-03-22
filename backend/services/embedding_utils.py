"""
Shared embedding utilities.

Provides ``pack_embedding()`` — the single implementation for converting
embedding lists/tuples to binary blobs for sqlite-vec virtual tables.
Previously duplicated across 10+ service files.
"""

import struct
from typing import Optional


def pack_embedding(embedding) -> Optional[bytes]:
    """Pack a list/tuple/ndarray of floats into a binary blob for sqlite-vec.

    Args:
        embedding: Embedding data as list, tuple, numpy ndarray, bytes, or None.

    Returns:
        Packed bytes blob, or None if embedding is None.
    """
    if embedding is None:
        return None
    if isinstance(embedding, bytes):
        return embedding
    if isinstance(embedding, (list, tuple)):
        return struct.pack(f'{len(embedding)}f', *embedding)
    # numpy arrays and similar array-like objects
    if hasattr(embedding, 'tolist'):
        flat = embedding.tolist()
        return struct.pack(f'{len(flat)}f', *flat)
    return embedding
