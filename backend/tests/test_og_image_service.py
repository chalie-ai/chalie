

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.unit


class _HtmlHandler(BaseHTTPRequestHandler):
    pages: dict[str, tuple[int, dict, bytes]] = {}

    def do_GET(self):
        if self.path not in self.pages:
            self.send_response(404)
            self.end_headers()
            return
        status, headers, body = self.pages[self.path]
        if status >= 300 and status < 400 and "Location" in headers:
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            return
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture(scope="module")
def html_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HtmlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    yield base, _HtmlHandler
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _reset():
    _HtmlHandler.pages = {}
    yield


def _html(meta_tags: str, title: str = "Page Title") -> bytes:
    return (
        f"<html><head>{meta_tags}<title>{title}</title></head><body>x</body></html>"
    ).encode()


class TestResolveOgImages:
    def test_og_image_extracted(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        body = _html('<meta property="og:image" content="https://cdn.example.com/og.jpg">')
        handler.pages["/article"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/article"])
        assert out[f"{base}/article"]["image_url"] == "https://cdn.example.com/og.jpg"

    def test_relative_url_resolved_against_page(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        body = _html('<meta property="og:image" content="/img/hero.png">')
        handler.pages["/relative"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/relative"])
        assert out[f"{base}/relative"]["image_url"] == f"{base}/img/hero.png"

    def test_redirect_followed_and_relative_resolves_to_final(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        # /redirect → /final ; og:image is relative, must resolve to /final's host (same here)
        handler.pages["/redirect"] = (302, {"Location": f"{base}/final"}, b"")
        body = _html('<meta property="og:image" content="/img/x.jpg">')
        handler.pages["/final"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/redirect"])
        assert out[f"{base}/redirect"]["image_url"] == f"{base}/img/x.jpg"

    def test_missing_meta_returns_no_entry(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        handler.pages["/bare"] = (200, {"Content-Type": "text/html"}, _html(""))
        out = resolve_og_images([f"{base}/bare"])
        assert out == {}

    def test_multiple_urls_parallel(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        handler.pages["/a"] = (200, {"Content-Type": "text/html"},
                               _html('<meta property="og:image" content="https://a.example/a.jpg">'))
        handler.pages["/b"] = (200, {"Content-Type": "text/html"},
                               _html('<meta property="og:image" content="https://b.example/b.jpg">'))
        urls = [f"{base}/a", f"{base}/b"]
        out = resolve_og_images(urls)
        assert out[f"{base}/a"]["image_url"] == "https://a.example/a.jpg"
        assert out[f"{base}/b"]["image_url"] == "https://b.example/b.jpg"

    def test_double_quoted_attribute_order_swapped(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        body = _html('<meta content="https://cdn.example.com/swap.jpg" property="og:image"/>')
        handler.pages["/swap"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/swap"])
        assert out[f"{base}/swap"]["image_url"] == "https://cdn.example.com/swap.jpg"


class TestResolveOgDescription:
    """og:description / og:title / <title> extraction alongside og:image."""

    def test_og_description_extracted(self, html_server):
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        body = _html(
            '<meta property="og:image" content="https://cdn.example.com/img.jpg">'
            '<meta property="og:description" content="Scientists find water on Mars.">'
        )
        handler.pages["/with-desc"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/with-desc"])
        assert out[f"{base}/with-desc"]["description"] == "Scientists find water on Mars."

    def test_og_image_resolved_when_buried_after_large_inline_payload(self, html_server):
        """Real-world parity: Google News article pages bury og:image past ~575KB.

        Regression guard against re-introducing a byte cap. The fetcher must
        keep streaming until ``</head>`` regardless of head size; if anyone
        reintroduces a truncation cap below the head's natural length, news
        thumbnails will silently disappear and this test will fail.
        """
        from services.og_image_service import resolve_og_images
        base, handler = html_server
        # Pad the head with ~610KB of inline content before the og:image tag —
        # mirrors the layout of real Google News article pages.
        padding = ("<!--" + "x" * 100 + "-->") * 5800
        body = (
            "<html><head>"
            + padding
            + '<meta property="og:image" content="https://cdn.example.com/buried.jpg">'
            + "<title>Buried</title></head><body>x</body></html>"
        ).encode()
        assert len(body) > 600 * 1024, "fixture must exceed 600KB to exercise the cap"
        handler.pages["/buried"] = (200, {"Content-Type": "text/html"}, body)
        out = resolve_og_images([f"{base}/buried"])
        assert out.get(f"{base}/buried", {}).get("image_url") == "https://cdn.example.com/buried.jpg"
