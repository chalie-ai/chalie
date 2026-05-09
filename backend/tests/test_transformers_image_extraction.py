"""Unit tests for per-provider image extraction in tools.search.transformers.

Pure functions: feed shaped dicts / XML strings, assert URL extracted.
Covers GitHub, Reddit, Open Library, iTunes, and the generic RSS thumbnail
patterns (media:thumbnail, media:content, enclosure, <image><url>).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tools.search.transformers import (
    FIELD_MAPS,
    _extract_image,
    _rss_item_image,
    _transform_json,
    _transform_rss,
)

pytestmark = pytest.mark.unit


# ── _extract_image: provider-specific paths ──────────────────────────────────


class TestExtractImageGithub:
    def test_owner_avatar_url_returned(self):
        item = {"full_name": "rust-lang/rust", "owner": {"avatar_url": "https://avatars.githubusercontent.com/u/5430905"}}
        assert _extract_image(item, FIELD_MAPS["github"]) == "https://avatars.githubusercontent.com/u/5430905"

    def test_missing_owner_returns_empty(self):
        assert _extract_image({"full_name": "x/y"}, FIELD_MAPS["github"]) == ""


class TestExtractImageReddit:
    def test_preview_source_url_preferred(self):
        item = {
            "title": "x",
            "preview": {"images": [{"source": {"url": "https://preview.redd.it/abc.jpg?width=640&amp;auto=webp"}}]},
            "thumbnail": "https://b.thumbs.redditmedia.com/x.jpg",
        }
        # html.unescape is applied so &amp; collapses to &
        assert _extract_image(item, FIELD_MAPS["reddit"]) == "https://preview.redd.it/abc.jpg?width=640&auto=webp"

    def test_falls_back_to_thumbnail_url(self):
        item = {"title": "x", "thumbnail": "https://b.thumbs.redditmedia.com/x.jpg"}
        assert _extract_image(item, FIELD_MAPS["reddit"]) == "https://b.thumbs.redditmedia.com/x.jpg"

    def test_self_thumbnail_placeholder_dropped(self):
        # Reddit emits "self" / "default" / "nsfw" sentinel strings in `thumbnail`
        # for posts without a real image. They aren't URLs and must be skipped.
        item = {"title": "x", "thumbnail": "self"}
        assert _extract_image(item, FIELD_MAPS["reddit"]) == ""


class TestExtractImageOpenLibrary:
    def test_cover_id_synthesised_into_covers_url(self):
        item = {"title": "Book", "cover_i": 12345}
        assert _extract_image(item, FIELD_MAPS["open_library"]) == "https://covers.openlibrary.org/b/id/12345-L.jpg"

    def test_no_cover_id_returns_empty(self):
        assert _extract_image({"title": "x"}, FIELD_MAPS["open_library"]) == ""


class TestExtractImageItunes:
    def test_artwork_100_used(self):
        item = {"trackName": "Song", "artworkUrl100": "https://is1-ssl.mzstatic.com/image/100x100bb.jpg"}
        assert _extract_image(item, FIELD_MAPS["itunes"]) == "https://is1-ssl.mzstatic.com/image/100x100bb.jpg"

    def test_falls_back_through_chain(self):
        item = {"trackName": "Song", "artworkUrl60": "https://is1-ssl.mzstatic.com/image/60x60bb.jpg"}
        assert _extract_image(item, FIELD_MAPS["itunes"]) == "https://is1-ssl.mzstatic.com/image/60x60bb.jpg"


class TestExtractImageGeneric:
    def test_no_image_field_returns_empty(self):
        # Wikipedia field map has no `image` key at all
        assert _extract_image({"title": "x"}, FIELD_MAPS["wikipedia"]) == ""

    def test_non_http_value_dropped_for_simple_field(self):
        # If artworkUrl* is a relative path, drop it (we can't fetch it standalone)
        item = {"trackName": "Song", "artworkUrl100": "//cdn/x.jpg"}
        assert _extract_image(item, FIELD_MAPS["itunes"]) == ""


# ── End-to-end transform: image flows through to result dicts ────────────────


class TestTransformJsonResultsCarryImage:
    def test_github_results_have_image_field(self):
        data = {"items": [{
            "full_name": "owner/repo",
            "description": "x",
            "html_url": "https://github.com/owner/repo",
            "owner": {"avatar_url": "https://avatars.githubusercontent.com/u/1"},
        }]}
        results = _transform_json("github", data, limit=5)
        assert len(results) == 1
        assert results[0]["image"] == "https://avatars.githubusercontent.com/u/1"

    def test_wikipedia_results_have_empty_image(self):
        data = {"query": {"search": [{"title": "Rust", "snippet": "lang"}]}}
        results = _transform_json("wikipedia", data, limit=5)
        assert results[0]["image"] == ""


# ── RSS thumbnail patterns (Google News-style feeds) ─────────────────────────


class TestRssItemImage:
    def test_media_thumbnail_extracted(self):
        xml = """<item xmlns:media='http://search.yahoo.com/mrss/'>
            <title>t</title><link>http://x</link>
            <media:thumbnail url='https://thumb.example.com/a.jpg'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://thumb.example.com/a.jpg"

    def test_media_content_image_type(self):
        xml = """<item xmlns:media='http://search.yahoo.com/mrss/'>
            <title>t</title><link>http://x</link>
            <media:content url='https://cdn.example.com/p.jpg' type='image/jpeg'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/p.jpg"

    def test_media_content_non_image_skipped(self):
        xml = """<item xmlns:media='http://search.yahoo.com/mrss/'>
            <title>t</title><link>http://x</link>
            <media:content url='https://cdn.example.com/v.mp4' type='video/mp4'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == ""

    def test_enclosure_image(self):
        xml = """<item>
            <title>t</title><link>http://x</link>
            <enclosure url='https://cdn.example.com/e.png' type='image/png' length='12345'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/e.png"

    def test_image_url_child(self):
        xml = """<item>
            <title>t</title><link>http://x</link>
            <image><url>https://cdn.example.com/i.jpg</url></image>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/i.jpg"

    def test_no_image_returns_empty(self):
        xml = "<item><title>t</title><link>http://x</link></item>"
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == ""


class TestTransformRssCarriesImage:
    def test_google_news_style_rss_with_thumbnails(self):
        xml = """<?xml version='1.0'?>
        <rss xmlns:media='http://search.yahoo.com/mrss/' version='2.0'><channel>
            <item>
              <title>Story A</title><link>http://a</link>
              <media:thumbnail url='https://t.example.com/a.jpg'/>
            </item>
            <item>
              <title>Story B</title><link>http://b</link>
            </item>
        </channel></rss>"""
        results = _transform_rss("google_news", xml, limit=5)
        assert results[0]["image"] == "https://t.example.com/a.jpg"
        assert results[1]["image"] == ""
