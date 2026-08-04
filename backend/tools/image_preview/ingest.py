# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ingestion primitives for ``image_preview`` — content sniffing and landing.

``sniff_image`` is a SECURITY BOUNDARY, not a convenience. ``image_preview`` is
an INTERNAL tool: it bypasses the policy gate on every channel, so this byte
check is the only thing between a model-supplied path and a file being copied
into the browser-served document store. It reads MAGIC BYTES ONLY — never the
extension — so a private key renamed ``key.png`` is rejected, and there is no
permissive fallback: anything unrecognised is ``None``, and ``None`` means stop.
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Final

from services.file_mapper_service import FileMapperService

# (signature, mime) — checked in order, so the weakest (2-byte BMP) comes last.
_SIGNATURES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
    (b"BM", "image/bmp"),
)

# Still-image brands of the ISO base-media container, read from its `ftyp` box.
_AVIF_BRANDS: Final[frozenset[bytes]] = frozenset({b"avif", b"avis"})
_HEIF_BRANDS: Final[frozenset[bytes]] = frozenset(
    {b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1"}
)

# SVG is markup, not magic bytes: strip the prolog/doctype/comments a real file
# may carry ahead of its root element, then require an actual <svg root.
_XML_NOISE: Final[re.Pattern[str]] = re.compile(
    r"<\?xml.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>", re.DOTALL | re.IGNORECASE
)
_SVG_ROOT: Final[re.Pattern[str]] = re.compile(r"<svg\b", re.IGNORECASE)

_SAFE_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]")
_PREVIEWS_DIR: Final[str] = "previews"


def sniff_image(head: bytes) -> str | None:
    """Return the image MIME type for ``head``, or ``None`` if it is not an image.

    Args:
        head: The first bytes of the candidate file; 512 covers every signature
            below, including an SVG prolog.

    Returns:
        The detected MIME type, or ``None`` when nothing matches. Callers MUST
        treat ``None`` as a hard rejection — it is the gate that keeps non-image
        files out of the served document store.
    """
    for signature, mime in _SIGNATURES:
        if head.startswith(signature):
            return mime

    # RIFF container: "RIFF" at 0, the form type at 8 — WEBP is one of many forms.
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    # ISO base-media: "ftyp" box at 4, the major brand at 8.
    if head[4:8] == b"ftyp":
        brand = head[8:12].lower()
        if brand in _AVIF_BRANDS:
            return "image/avif"
        if brand in _HEIF_BRANDS:
            return "image/heic"

    if _SVG_ROOT.search(_XML_NOISE.sub("", head.decode("utf-8", errors="ignore"))):
        return "image/svg+xml"

    return None


def land_image(src_path: str, original_name: str) -> str:
    """Copy ``src_path`` into the served previews directory under a unique name.

    Args:
        src_path: Absolute path of the already-verified image to copy.
        original_name: Name the image arrived as; only its basename survives,
            stripped to ``[A-Za-z0-9._-]``.

    Returns:
        The destination path RELATIVE to the documents root, forward-slashed —
        ready to append to ``/api/files/preview/``.
    """
    previews = FileMapperService.get_documents_path(_PREVIEWS_DIR)
    previews.mkdir(parents=True, exist_ok=True)

    # Sanitise first, THEN fall back: a name that is entirely non-ASCII strips to
    # empty, and the uuid prefix keeps a dot-only basename ("..") from ever
    # forming a traversal.
    safe = _SAFE_CHARS.sub("", Path(original_name).name) or "image"
    dest = previews / f"{uuid.uuid4().hex[:8]}_{safe}"

    shutil.copyfile(src_path, dest)
    return dest.relative_to(FileMapperService.get_documents_path()).as_posix()
