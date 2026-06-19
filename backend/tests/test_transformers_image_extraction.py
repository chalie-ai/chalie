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
    def test_owner_avatar_url_returned(self) -> None:
        item: dict[str, object] = {"full_name": "rust-lang/rust", "owner": {"avatar_url": "https://avatars.githubusercontent.com/u/5430905"}}
        assert _extract_image(item, FIELD_MAPS["github"]) == "https://avatars.githubusercontent.com/u/5430905"




class TestExtractImageReddit:
    def test_preview_source_url_preferred(self) -> None:
        item: dict[str, object] = {
            "title": "x",
            "preview": {"images": [{"source": {"url": "https://preview.redd.it/abc.jpg?width=640&amp;auto=webp"}}]},
            "thumbnail": "https://b.thumbs.redditmedia.com/x.jpg",
        }
        # html.unescape is applied so &amp; collapses to &
        assert _extract_image(item, FIELD_MAPS["reddit"]) == "https://preview.redd.it/abc.jpg?width=640&auto=webp"

    def test_self_thumbnail_placeholder_dropped(self) -> None:
        # Reddit emits "self" / "default" / "nsfw" sentinel strings in `thumbnail`
        # for posts without a real image. They aren't URLs and must be skipped.
        item: dict[str, object] = {"title": "x", "thumbnail": "self"}
        assert _extract_image(item, FIELD_MAPS["reddit"]) == ""


class TestExtractImageOpenLibrary:
    def test_cover_id_synthesised_into_covers_url(self) -> None:
        item: dict[str, object] = {"title": "Book", "cover_i": 12345}
        assert _extract_image(item, FIELD_MAPS["open_library"]) == "https://covers.openlibrary.org/b/id/12345-L.jpg"




class TestExtractImageItunes:
    def test_artwork_100_used(self) -> None:
        item: dict[str, object] = {"trackName": "Song", "artworkUrl100": "https://is1-ssl.mzstatic.com/image/100x100bb.jpg"}
        assert _extract_image(item, FIELD_MAPS["itunes"]) == "https://is1-ssl.mzstatic.com/image/100x100bb.jpg"




# ── End-to-end transform: image flows through to result dicts ────────────────


class TestTransformJsonResultsCarryImage:
    def test_github_results_have_image_field(self) -> None:
        data: dict[str, object] = {"items": [{
            "full_name": "owner/repo",
            "description": "x",
            "html_url": "https://github.com/owner/repo",
            "owner": {"avatar_url": "https://avatars.githubusercontent.com/u/1"},
        }]}
        results = _transform_json("github", data, limit=5)
        assert len(results) == 1
        assert results[0]["image"] == "https://avatars.githubusercontent.com/u/1"


# ── RSS thumbnail patterns (Google News-style feeds) ─────────────────────────


class TestRssItemImage:
    def test_media_thumbnail_extracted(self) -> None:
        xml = """<item xmlns:media='http://search.yahoo.com/mrss/'>
            <title>t</title><link>http://x</link>
            <media:thumbnail url='https://thumb.example.com/a.jpg'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://thumb.example.com/a.jpg"

    def test_media_content_image_type(self) -> None:
        xml = """<item xmlns:media='http://search.yahoo.com/mrss/'>
            <title>t</title><link>http://x</link>
            <media:content url='https://cdn.example.com/p.jpg' type='image/jpeg'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/p.jpg"

    def test_enclosure_image(self) -> None:
        xml = """<item>
            <title>t</title><link>http://x</link>
            <enclosure url='https://cdn.example.com/e.png' type='image/png' length='12345'/>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/e.png"

    def test_image_url_child(self) -> None:
        xml = """<item>
            <title>t</title><link>http://x</link>
            <image><url>https://cdn.example.com/i.jpg</url></image>
        </item>"""
        item = ET.fromstring(xml)
        assert _rss_item_image(item) == "https://cdn.example.com/i.jpg"




class TestTransformRssCarriesImage:
    def test_google_news_style_rss_with_thumbnails(self) -> None:
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
