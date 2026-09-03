# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Content sniffing for ``image_preview`` — is this file really an image?

``sniff_image`` is a SECURITY BOUNDARY, not a convenience. ``image_preview`` is
an INTERNAL tool: it bypasses the policy gate on every channel, so this byte
check is the only thing between a model-supplied path and a file being copied
into the browser-served document store. It reads MAGIC BYTES ONLY — never the
extension — so a private key renamed ``key.png`` is rejected, and there is no
permissive fallback: anything unrecognised is ``None``, and ``None`` means stop.
"""

from __future__ import annotations

import re
from typing import Final

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
