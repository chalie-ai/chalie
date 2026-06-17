

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

pytestmark = pytest.mark.unit  # no external deps once the local HTTP server is up


def _png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    """Caption is URL-derived so no image text metadata is needed."""
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


class TestCaptionFromFilename:
    def test_caption_from_wikipedia_thumbnail_url(self):
        from services.image_candidate_service import _caption_from_filename
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/"
            "Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg/"
            "1280px-Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg"
        )
        caption = _caption_from_filename(url)
        assert caption == "Mt. Everest from Gokyo Ri November 5, 2012"

    def test_caption_empty_when_filename_is_hash(self):
        from services.image_candidate_service import _caption_from_filename
        # Pure hex/digit hash filenames should yield empty string
        url = "https://cdn.example.com/images/a3f9b2c1d4e5f678.jpg"
        caption = _caption_from_filename(url)
        assert caption == ""

class TestCaptionUsesOgDescription:
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
