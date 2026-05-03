"""Feature tests for services.image_candidate_service.

Real production stack: spins up a local HTTP server serving generated images
(real PNG bytes, real OCR via RapidOCR). No mocks. Each test exercises the
fetch + OCR path end-to-end against the live OCR engine.
"""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = pytest.mark.unit  # no external deps once the local HTTP server is up


def _png_with_text(text: str, size: tuple[int, int] = (320, 96)) -> bytes:
    """Render text onto a white PNG and return the encoded bytes.

    Uses the default PIL font to keep the dependency surface zero — it's
    pixelated but RapidOCR reads it fine for short labels.
    """
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=36)
    except (TypeError, AttributeError):
        font = ImageFont.load_default()
    draw.text((20, 20), text, fill="black", font=font)
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
        handler.images["/dup.png"] = _png_with_text("DUPE")
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
            handler.images[path] = _png_with_text(f"IMG{i}")
            urls.append(f"{base}{path}")
        out = build_image_candidates(urls, top_n=3)
        assert len(out) == 3
        assert [c["url"] for c in out] == urls[:3]

    def test_ocr_text_is_extracted(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/headline.png"] = _png_with_text("BREAKING")
        out = build_image_candidates([f"{base}/headline.png"])
        assert len(out) == 1
        # RapidOCR may segment differently — check the dominant token survives.
        assert "BREAKING" in out[0]["ocr_text"].upper()

    def test_failed_fetch_drops_silently(self, image_server):
        from services.image_candidate_service import build_image_candidates
        base, handler = image_server
        handler.images["/ok.png"] = _png_with_text("OK")
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
        handler.images["/anything.png"] = _png_with_text("X")
        url = f"{base}/anything.png"
        out = build_image_candidates([url])
        assert out[0]["url"] == url
