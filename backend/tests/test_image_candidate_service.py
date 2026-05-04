"""Feature tests for services.image_candidate_service.

Real production stack: spins up a local HTTP server serving real PNG bytes.
No mocks. Each test exercises the fetch + caption derivation path end-to-end.
"""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = pytest.mark.unit  # no external deps once the local HTTP server is up


def _png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    """Return minimal valid PNG bytes (no text needed — caption is URL-derived)."""
    img = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _ImageHandler(BaseHTTPRequestHandler):
    images: dict[str, bytes] = {}
    fail_paths: set[str] = set()

    def do_GET(self):
        if self.path in self.fail_paths:
            self.send_response(500)
            self.end_headers()
            return
        if self.path not in self.images:
            self.send_response(404)
            self.end_headers()
            return
        body = self.images[self.path]
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass  # silence test output


@pytest.fixture(scope="module")
def image_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    yield base, _ImageHandler
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _reset_handler():
    _ImageHandler.images = {}
    _ImageHandler.fail_paths = set()
    yield


class TestBuildImageCandidates:
    def test_empty_input_returns_empty(self):
        from services.image_candidate_service import build_image_candidates
        assert build_image_candidates([]) == []

    def test_filters_non_string_and_empty(self):
        from services.image_candidate_service import build_image_candidates
        assert build_image_candidates(["", None, 0]) == []  # type: ignore[list-item]

    def test_dedup_by_url_in_input_order(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/dup.png"] = _png_bytes()
        url = f"{base}/dup.png"
        out = build_image_candidates([url, url, url])
        assert len(out) == 1
        assert out[0]["url"] == url

    def test_caps_at_top_n(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        urls = []
        for i in range(5):
            path = f"/cap{i}.png"
            handler.images[path] = _png_bytes()
            urls.append(f"{base}{path}")
        out = build_image_candidates(urls, top_n=3)
        assert len(out) == 3
        assert [c["url"] for c in out] == urls[:3]

    def test_failed_fetch_drops_silently(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/ok.png"] = _png_bytes()
        handler.fail_paths = {"/fail.png"}
        out = build_image_candidates([
            f"{base}/fail.png",
            f"{base}/ok.png",
            f"{base}/nope.png",  # 404
        ])
        urls = [c["url"] for c in out]
        assert urls == [f"{base}/ok.png"]

    def test_returns_url_verbatim(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/anything.png"] = _png_bytes()
        url = f"{base}/anything.png"
        out = build_image_candidates([url])
        assert out[0]["url"] == url

    def test_bare_url_yields_empty_source_title(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/x.png"] = _png_bytes()
        out = build_image_candidates([f"{base}/x.png"])
        assert out[0]["source_title"] == ""

    def test_tuple_input_carries_source_title(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/a.png"] = _png_bytes()
        handler.images["/b.png"] = _png_bytes()
        items = [
            (f"{base}/a.png", "Reuters: Election called"),
            (f"{base}/b.png", "Politico: Snap election"),
        ]
        out = build_image_candidates(items)
        assert [c["source_title"] for c in out] == [
            "Reuters: Election called",
            "Politico: Snap election",
        ]

    def test_caption_field_present_in_output(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/photo.png"] = _png_bytes()
        out = build_image_candidates([f"{base}/photo.png"])
        assert len(out) == 1
        assert "caption" in out[0]

    def test_no_ocr_text_field_in_output(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/photo.png"] = _png_bytes()
        out = build_image_candidates([f"{base}/photo.png"])
        assert "ocr_text" not in out[0]


class TestCaptionFromFilename:
    """Unit tests for the URL-filename caption heuristic."""

    def test_caption_from_wikipedia_thumbnail_url(self):
        from services.image_candidate_service import _caption_from_filename
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/"
            "Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg/"
            "1280px-Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg"
        )
        caption = _caption_from_filename(url)
        assert caption == "Mt. Everest from Gokyo Ri November 5, 2012"

    def test_caption_from_filename_with_url_encoding(self):
        from services.image_candidate_service import _caption_from_filename
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/"
            "Lotte_World_day_view_2.jpg/960px-Lotte_World_day_view_2.jpg"
        )
        caption = _caption_from_filename(url)
        assert caption == "Lotte World day view 2"

    def test_caption_strips_wiki_prefix_multiple_widths(self):
        from services.image_candidate_service import _caption_from_filename
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/x/xx/"
            "Mount_Everest_North_Face.jpg/320px-Mount_Everest_North_Face.jpg"
        )
        caption = _caption_from_filename(url)
        assert caption == "Mount Everest North Face"

    def test_caption_empty_when_filename_is_hash(self):
        from services.image_candidate_service import _caption_from_filename
        # Pure hex/digit hash filenames should yield empty string
        url = "https://cdn.example.com/images/a3f9b2c1d4e5f678.jpg"
        caption = _caption_from_filename(url)
        assert caption == ""

    def test_caption_empty_for_uuid_style_filename(self):
        from services.image_candidate_service import _caption_from_filename
        url = "https://media.example.com/1234567890abcdef1234567890abcdef.jpg"
        caption = _caption_from_filename(url)
        assert caption == ""

    def test_caption_capped_at_200_chars(self):
        from services.image_candidate_service import _caption_from_filename
        # Build a long but valid descriptive filename
        long_name = "_".join(["word"] * 60)
        url = f"https://example.com/{long_name}.jpg"
        caption = _caption_from_filename(url)
        assert len(caption) <= 200


class TestCaptionUsesOgDescription:
    """build_image_candidates uses og:description from og_meta when available."""

    def test_caption_uses_og_description_when_available(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/article_thumb.jpg"] = _png_bytes()
        img_url = f"{base}/article_thumb.jpg"
        og_meta = {
            img_url: {
                "image_url": img_url,
                "description": "Scientists discover record-breaking glacier retreat in Antarctica.",
            }
        }
        out = build_image_candidates([(img_url, "Science Daily")], og_meta=og_meta)
        assert len(out) == 1
        assert out[0]["caption"] == "Scientists discover record-breaking glacier retreat in Antarctica."

    def test_og_description_ignored_when_shorter_than_20_chars(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/Mount_Everest_peak.jpg"] = _png_bytes()
        img_url = f"{base}/Mount_Everest_peak.jpg"
        og_meta = {
            img_url: {
                "image_url": img_url,
                "description": "Too short",  # < 20 chars → falls through to filename
            }
        }
        out = build_image_candidates([(img_url, "Wikipedia")], og_meta=og_meta)
        assert len(out) == 1
        # Falls back to filename heuristic
        assert "Mount Everest peak" in out[0]["caption"]
