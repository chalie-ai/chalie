"""Ending-aware file text I/O — the single chokepoint for tools that read a
user file as text, change a span of it, and write it back.

The invariant this class exists to keep: match on LF-normalized text (what the
``read`` tool's universal-newlines display shows the model) and persist with
the file's ORIGINAL ending, so every line outside the replaced span stays
byte-identical.

``newline=""`` on BOTH ends is the crux. A text-mode ``open`` with the default
(``newline=None``) translates: on read it folds every ``\r\n`` and lone ``\r``
into ``\n`` (universal newlines), and on write it maps ``\n`` to ``os.linesep``.
A naive read-modify-write of a CRLF file therefore ate every ``\r\n`` and wrote
the normalized text back — silently rewriting every line the edit never
touched. ``newline=""`` disables both directions of
translation, so the read is the file's exact bytes and the write is the
string's exact bytes.

BOM policy is plain ``utf-8`` on both ends, NOT ``utf-8-sig``: a leading
U+FEFF survives the round-trip byte-identical, and matching stays consistent
with the ``read`` display (``_extract_plain`` also decodes plain ``utf-8``, so
the model sees the BOM as a character and an anchor covering it matches).

Mixed endings are a REFUSAL, never a normalization: a file that mixes ``\r\n``
and ``\n`` (or lone ``\r``) cannot be restored byte-identical for every
untouched line under any single choice of ending — any write would silently
rewrite the lines carrying the other convention. The caller must refuse loudly
(``edit_file`` returns ``code=mixed-line-endings``) and leave the file alone.

Likewise, an undecodable file is never half-edited: ``read_raw`` lets the
strict ``UnicodeDecodeError`` propagate to the caller, which maps it to a clean
tool error (``edit_file`` returns ``code=decode-error``) — never a stack
trace, and never a rewrite with replacement characters baked in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


class FileTextService:
    """Ending-aware file text I/O.

    Stateless by design: every method is static; the zero-arg constructor
    exists only so a caller may instantiate one — there is nothing to configure.
    """

    def __init__(self) -> None:
        # Nothing to hold — the stateless zero-arg constructor the class
        # contract promises.
        pass

    @staticmethod
    def read_raw(path: Path) -> str:
        """Read a file as text with NO newline translation, strict UTF-8.

        ``newline=""`` keeps every ``\r\n`` / ``\r`` exactly as stored — the
        default (``None``) would fold them all to ``\n`` before the caller ever
        sees them. Decoding is strict: a ``UnicodeDecodeError`` propagates to
        the caller, which decides how to report it (``edit_file`` maps it to
        ``code=decode-error``).
        """
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()

    @staticmethod
    def detect_ending(text: str) -> Literal["crlf", "lf", "mixed", "none"]:
        """Classify the line endings of RAW (untranslated) text.

        ``crlf`` — every ending is ``\\r\\n``; ``lf`` — every ending is
        ``\\n``; ``none`` — the text has no line ending at all (a single-line
        or empty file, nothing to preserve); ``mixed`` — at least two
        conventions coexist (including lone-``\\r`` files, which no modern
        writer can round-trip faithfully), and which the caller must refuse.
        """
        crlf = text.count("\r\n")
        lf_only = text.count("\n") - crlf
        cr_only = text.count("\r") - crlf
        if crlf and (lf_only or cr_only):
            return "mixed"
        if cr_only:
            return "mixed"
        if crlf:
            return "crlf"
        if lf_only:
            return "lf"
        return "none"

    @staticmethod
    def normalize(text: str) -> str:
        """Collapse ``\\r\\n`` to ``\\n`` — and ONLY ``\\r\\n``.

        This is the form the ``read`` tool's universal-newlines display shows
        the model, so an anchor quoted from ``read`` matches here. Lone ``\\r``
        is deliberately left alone: a file carrying one is already ``mixed``
        and is refused before any matching happens.
        """
        return text.replace("\r\n", "\n")

    @staticmethod
    def restore(text: str, ending: str) -> str:
        """Re-apply the file's original ending before the text is persisted.

        Only ``crlf`` is restored (``\\n`` -> ``\\r\\n``); ``lf`` and ``none``
        need no change. ``mixed`` must never reach this method — the caller
        refuses it, because no single ending can keep it byte-identical.
        """
        if ending == "crlf":
            return text.replace("\n", "\r\n")
        return text

    @staticmethod
    def write_raw(path: Path, text: str) -> None:
        """Write text with NO newline translation, strict UTF-8.

        ``newline=""`` makes the bytes on disk exactly the string's UTF-8
        encoding — the default (``None``) would rewrite every ``\n`` to
        ``os.linesep``, silently converting the file to the host OS's
        convention.
        """
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
