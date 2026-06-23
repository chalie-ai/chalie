
import struct
from typing import Optional, cast


def pack_embedding(embedding: object) -> Optional[bytes]:
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
    return cast(bytes, embedding)
