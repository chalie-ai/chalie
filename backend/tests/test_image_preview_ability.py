# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests: ``ImagePreviewAbility`` and ``ingest.sniff_image``.

Covers the four branches exercised by ``ImagePreviewAbility.run()``:
(local file outside docs root -> landed under previews/, local file inside docs
root -> served in place, non-image bytes -> error, subtitle trimming logic).

Also exercises ``tools/image_preview/ingest.sniff_image`` directly with four
different magic signatures (plus a None-return case).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from abilities.image_preview import ImagePreviewAbility
from contracts.params.image_preview_params_bag import ImagePreviewParamsBag
from services.file_mapper_service import FileMapperService
from tools.image_preview.ingest import sniff_image

MINIMAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

pytestmark = pytest.mark.unit


def _local_png_outside_docs(tmp_path: Path, name: str = "pic.png") -> Path:
    p = tmp_path / name
    p.write_bytes(MINIMAL_PNG)
    return p


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", tmp_path / "docs")


class TestLocalPngOutsideDocsRoot:
    """File lands under ``previews/`` and survives a byte-identical disk copy."""

    def test_lands_under_previews_with_matching_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _setup(monkeypatch, tmp_path)
        src = _local_png_outside_docs(tmp_path, "photo.png")

        tr = ImagePreviewAbility().run(
            ImagePreviewParamsBag(file_path=str(src), subtitle=None)
        )

        assert tr.status == "success"
        rich = tr.rich
        assert rich is not None
        url = cast("str", rich["url"])
        assert url.startswith("/api/files/preview/previews/")
        dest = FileMapperService.get_documents_path() / url.replace(
            "/api/files/preview/", ""
        )
        assert dest.exists()
        assert dest.read_bytes() == MINIMAL_PNG

    def test_inside_docs_root_served_in_place_no_copy_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _setup(monkeypatch, tmp_path)
        docs = FileMapperService.get_documents_path()
        docs.mkdir(parents=True, exist_ok=True)
        internal = docs / "photos" / "vacation.png"
        internal.parent.mkdir(parents=True, exist_ok=True)
        internal.write_bytes(MINIMAL_PNG)

        tr = ImagePreviewAbility().run(
            ImagePreviewParamsBag(file_path=str(internal), subtitle=None)
        )

        assert tr.status == "success"
        rich = tr.rich
        assert rich is not None
        rel = str(internal.relative_to(docs)).replace("\\", "/")
        assert rich["url"] == f"/api/files/preview/{rel}"
        previews = FileMapperService.get_documents_path("previews")
        assert not previews.exists()


class TestNonImageNamedPng:
    """Bytes that aren't an image MUST be rejected even when the extension lies."""

    def test_private_key_rejected_not_an_image(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 ) -> None:
        _setup(monkeypatch, tmp_path)
        fake = tmp_path / "secret.png"
        fake.write_bytes(b"-----BEGIN PRIVATE KEY-----\n" + b"x" * 64)

        tr = ImagePreviewAbility().run(
            ImagePreviewParamsBag(file_path=str(fake), subtitle=None)
        )

        assert tr.status == "error"
        assert tr.code == "not-an-image"
        previews = FileMapperService.get_documents_path("previews")
        assert not previews.exists()


class TestSubtitleTruncation:
    """Both halves of the subtitle clause:
    - over-limit subtitles get trimmed to eight words + U+2026 and flagged
      subtitle_truncated=True;
    - at-or-under-limit subtitles survive verbatim with subtitle_truncated=False.
    """

    _OVER_LIMIT = (
        "alpha bravo charlie delta echo foxtrot golf hotel india"
    )
    _UNDER_LIMIT = "alpha bravo charlie delta echo foxtrot golf hotel"

    def test_truncation_and_preserve_both_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _setup(monkeypatch, tmp_path)

        # Branch A: over the word limit -> trimmed + flagged.
        src_a = _local_png_outside_docs(tmp_path, "a.png")
        tr_a = ImagePreviewAbility().run(
            ImagePreviewParamsBag(
                file_path=str(src_a), subtitle=self._OVER_LIMIT
            )
        )
        assert tr_a.status == "success"
        rich_a = tr_a.rich
        assert rich_a is not None
        expected_trimmed = (
            " ".join(self._OVER_LIMIT.split()[:8]) + "\u2026"
        )
        assert rich_a["subtitle"] == expected_trimmed
        assert tr_a.meta["subtitle_truncated"] is True

        # Branch B: within the limit -> survives verbatim, flag off.
        src_b = _local_png_outside_docs(tmp_path, "b.png")
        tr_b = ImagePreviewAbility().run(
            ImagePreviewParamsBag(
                file_path=str(src_b), subtitle=self._UNDER_LIMIT
            )
        )
        assert tr_b.status == "success"
        rich_b = tr_b.rich
        assert rich_b is not None
        assert rich_b["subtitle"] == self._UNDER_LIMIT
        assert tr_b.meta["subtitle_truncated"] is False


class TestSniffImageDirectly:
    """``ingest.sniff_image`` recognizes PNG, SVG, WEBP, and refuses junk."""

    def test_returns_mime_for_known_formats_and_none_for_junk(self) -> None:
        png_head = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        assert sniff_image(png_head) == "image/png"

        svg_head = (
            b'<?xml version="1.0"?>\n'
            b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        )
        assert sniff_image(svg_head) == "image/svg+xml"

        webp_head = b"RIFF" + b"\x00" * 4 + b"WEBP"
        assert sniff_image(webp_head) == "image/webp"

        assert sniff_image(b"not an image at all") is None
