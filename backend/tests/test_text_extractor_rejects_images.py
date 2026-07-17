"""Feature test: extract_text is TEXT-ONLY and rejects images LOUDLY.

Replaces test_text_extractor_image_describe.py, whose premise (images route
through extract_text via the _extract_image shim) is deleted: an image is
DESCRIBED by the vision core, never 'extracted'.

The rejection must be an exception, not a fallthrough. extract_text's last
resort for an unknown mime is a plain UTF-8 read with errors='replace' — handed
image bytes it would return mojibake and look like a SUCCESSFUL extraction, and
that mojibake would be embedded and indexed as the document's content.
"""

import sqlite3
from pathlib import Path

import pytest

from services.text_extractor import extract_text
from tests.helpers import blank_png_bytes, ocrable_png_bytes

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["x.png", "x.jpg", "x.webp", "x.gif", "x.bmp", "x.tiff"])
def test_extract_text_rejects_every_image_type(tmp_path: Path, name: str) -> None:
    """Every image type formerly in the dispatch table now raises."""
    p = tmp_path / name
    p.write_bytes(blank_png_bytes())

    with pytest.raises(ValueError, match="images are described, not extracted"):
        extract_text(str(p))


def test_extract_text_rejects_image_by_explicit_mime(tmp_path: Path) -> None:
    """The reject keys off the mime, not the extension: an image passed with an
    explicit image/* mime is refused even when the path lies about the type."""
    p = tmp_path / "not_really.txt"
    p.write_bytes(blank_png_bytes())

    with pytest.raises(ValueError, match="images are described, not extracted"):
        extract_text(str(p), "image/heic")


def test_image_bytes_are_never_returned_as_mojibake(tmp_path: Path) -> None:
    """The regression this guards: a real PNG must NEVER come back as text. Before
    the reject, an unlisted image mime fell through to the plain-read fallback and
    returned decoded binary as a 'successful' extraction."""
    p = tmp_path / "invoice.png"
    p.write_bytes(ocrable_png_bytes())

    with pytest.raises(ValueError):
        extract_text(str(p), "image/heic")


def test_text_extraction_still_works(tmp_path: Path) -> None:
    """The reject is surgical: real text files are unaffected."""
    p = tmp_path / "notes.txt"
    p.write_text("hello world")

    assert extract_text(str(p)) == "hello world"


def test_read_ability_routes_images_to_vision(db: sqlite3.Connection, tmp_path: Path) -> None:
    """read is a text reader: handed an image it returns an actionable error with a
    route to the vision tool — never a stack trace, never mojibake."""
    from abilities.read import ReadAbility

    p = tmp_path / "photo.png"
    p.write_bytes(blank_png_bytes())

    result = ReadAbility(mp=None).run({"source": str(p)})

    assert result.status == "error"
    assert result.code == "not-text"
    assert "vision" in (result.hint or "").lower()
