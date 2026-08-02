"""Tests for UrlToMarkdownService — HTML→markdown fidelity, naming, persistence.

The conversion fixture is deliberately padded to realistic article length:
trafilatura degrades below its minimum-size heuristics (structure lost,
content duplicated), a regime real pages never occupy. The fidelity
assertions here are the go/no-go gate on its markdown mode — headings,
links, images, and tables must survive, relative links must come back
absolute, and the body must not be duplicated.
"""

from pathlib import Path

import pytest

from services.file_mapper_service import FileMapperService
from services.url_to_markdown_service import UrlToMarkdownService

_URL = "https://example.com/studies/monarchs"

_PARAGRAPH = (
    "The migration of monarch butterflies across North America remains one of "
    "the most studied phenomena in entomology, spanning thousands of kilometres "
    "and multiple generations in a single seasonal cycle, with each cohort "
    "navigating by a combination of solar position and an inherited magnetic "
    "compass that researchers only partially understand."
)

_PADDING = "\n".join(f"<p>Paragraph {i}: {_PARAGRAPH}</p>" for i in range(1, 9))

_ARTICLE_HTML = f"""<html><head><title>Monarch Migration Study</title></head><body>
<article>
<h1>Monarch Migration Study</h1>
<p>UNIQUE-MARKER-SENTENCE: this exact sentence appears once in the source. See the
<a href="/library/monarchs">library entry</a> and the
<a href="https://other.example.org/absolute-ref">external reference</a>.</p>
{_PADDING}
<h2>Observed Counts</h2>
<img src="/img/count-chart.png" alt="Count chart">
<table><tr><th>Season</th><th>Count</th></tr><tr><td>Autumn</td><td>1200</td></tr><tr><td>Spring</td><td>800</td></tr></table>
<h2>Field Recordings</h2>
<iframe src="https://player.example.com/embed/42"></iframe>
<video src="/media/field-clip.mp4"></video>
<p>Concluding remarks close the study with acknowledgements to the volunteer network.</p>
</article></body></html>"""


@pytest.mark.unit
class TestConvertFidelity:

    @pytest.fixture(scope="class")
    def markdown(self) -> str:
        return UrlToMarkdownService().convert(_ARTICLE_HTML, _URL)

    def test_headings_survive(self, markdown: str) -> None:
        assert "# Monarch Migration Study" in markdown
        assert "## Observed Counts" in markdown

    def test_relative_link_comes_back_absolute(self, markdown: str) -> None:
        assert "(https://example.com/library/monarchs)" in markdown

    def test_absolute_link_intact(self, markdown: str) -> None:
        assert "(https://other.example.org/absolute-ref)" in markdown

    def test_image_survives(self, markdown: str) -> None:
        assert "![Count chart]" in markdown

    def test_table_survives_as_markdown(self, markdown: str) -> None:
        assert "| Season | Count |" in markdown
        assert "| Autumn | 1200 |" in markdown

    def test_body_not_duplicated(self, markdown: str) -> None:
        """Below trafilatura's size heuristics the body comes back twice —
        this asserts the real-page regime, where it must appear once."""
        assert markdown.count("UNIQUE-MARKER-SENTENCE") == 1
        assert markdown.count("Concluding remarks") == 1

    def test_embedded_media_section_lists_absolute_urls(self, markdown: str) -> None:
        assert "## Embedded media" in markdown
        assert "(https://player.example.com/embed/42)" in markdown
        # Relative video src resolved against the page URL.
        assert "(https://example.com/media/field-clip.mp4)" in markdown

    def test_no_media_means_no_section(self) -> None:
        html = _ARTICLE_HTML.replace(
            '<iframe src="https://player.example.com/embed/42"></iframe>', ""
        ).replace('<video src="/media/field-clip.mp4"></video>', "")
        markdown = UrlToMarkdownService().convert(html, _URL)
        assert "## Embedded media" not in markdown

    def test_bytes_input_respects_declared_charset(self) -> None:
        """Raw bytes go to trafilatura verbatim so its charset sniffing runs —
        a latin-1 page's accents must survive without mojibake."""
        para = "Un texte suffisamment long pour la taille minimale. " * 30
        html = (
            '<html><head><meta charset="iso-8859-1"></head><body><article>'
            f"<h1>Café Étude</h1><p>naïveté, déjà vu - accents survive. {para}</p>"
            f"<p>{para}</p></article></body></html>"
        ).encode("iso-8859-1")
        markdown = UrlToMarkdownService().convert(html, "https://example.com/cafe")
        assert "Café Étude" in markdown
        assert "naïveté, déjà vu" in markdown

    def test_empty_html_raises(self) -> None:
        with pytest.raises(ValueError, match="no content"):
            UrlToMarkdownService().convert("", _URL)

    def test_unextractable_html_raises(self) -> None:
        with pytest.raises(ValueError, match="no content"):
            UrlToMarkdownService().convert("<html><body></body></html>", _URL)


@pytest.mark.unit
class TestFilenameFor:

    def test_host_and_path_slug(self) -> None:
        name = UrlToMarkdownService().filename_for(
            "https://example.com/Blog/Entry_One", "md"
        )
        assert name == "example.com--blog-entry-one.md"

    def test_empty_path_gives_host_only(self) -> None:
        assert UrlToMarkdownService().filename_for("https://example.com/", "md") == "example.com.md"
        assert UrlToMarkdownService().filename_for("https://example.com", "html") == "example.com.html"

    def test_non_alnum_runs_collapse_to_single_dash(self) -> None:
        name = UrlToMarkdownService().filename_for(
            "https://example.com/a//b__c.d", "md"
        )
        assert name == "example.com--a-b-c-d.md"

    def test_slug_capped_at_120_chars(self) -> None:
        long_path = "/" + "segment/" * 40
        name = UrlToMarkdownService().filename_for(f"https://example.com{long_path}", "md")
        slug = name.removeprefix("example.com--").removesuffix(".md")
        assert len(slug) <= 120

    def test_same_url_always_same_name(self) -> None:
        svc = UrlToMarkdownService()
        assert svc.filename_for(_URL, "md") == svc.filename_for(_URL, "md")


@pytest.mark.unit
class TestPersist:

    @pytest.fixture()
    def data_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
        return tmp_path

    def test_writes_under_web_pages_and_returns_path(self, data_dir: Path) -> None:
        written = UrlToMarkdownService().persist("# Body", _URL, "md")
        assert written.is_file()
        assert written.parent == data_dir / "web" / "pages"
        assert written.read_text(encoding="utf-8") == "# Body"

    def test_refetch_overwrites_in_place(self, data_dir: Path) -> None:
        svc = UrlToMarkdownService()
        first = svc.persist("old content", _URL, "md")
        second = svc.persist("new content", _URL, "md")
        assert first == second
        assert second.read_text(encoding="utf-8") == "new content"
        pages = list((data_dir / "web" / "pages").iterdir())
        assert len(pages) == 1, f"expected one file per URL, found {pages}"
