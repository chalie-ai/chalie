"""Feature tests: the ``web_fetch`` ability's fetch-and-persist contract.

The network edge is the only stand-in: a stub for ``stream_to_file`` (patched
at the ability's import site) writes a canned payload to the exact destination
the ability chose and returns ``(bytes, content_type)``. Everything downstream
is the real code on a tmp-rooted data dir — trafilatura conversion,
FileMapperService persistence, the 20 000-character gate, and temp-file
cleanup — so these tests fail when any of those seams actually break.
"""

from pathlib import Path

import pytest
import requests

from abilities._result import ToolResult
from abilities.web_fetch import WebFetchAbility
from contracts.params.web_fetch_params_bag import WebFetchParamsBag
from exceptions import DownloadTooLarge, FetchBlocked
from services.file_mapper_service import FileMapperService
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

_URL = "https://example.com/studies/monarchs"

_PARAGRAPH = (
    "The migration of monarch butterflies across North America remains one of "
    "the most studied phenomena in entomology, spanning thousands of kilometres "
    "and multiple generations in a single seasonal cycle, with each cohort "
    "navigating by a combination of solar position and an inherited magnetic "
    "compass that researchers only partially understand."
)


def _article_html(paragraphs: int) -> bytes:
    body = "\n".join(f"<p>Paragraph {i}: {_PARAGRAPH}</p>" for i in range(1, paragraphs + 1))
    return (
        "<html><head><title>Monarch Migration Study</title></head><body><article>"
        f"<h1>Monarch Migration Study</h1>{body}</article></body></html>"
    ).encode("utf-8")


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", tmp_path)
    return tmp_path


def _stub_fetch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: bytes = b"",
    content_type: str = "text/html",
    raising: BaseException | None = None,
    calls: list[dict[str, object]] | None = None,
) -> None:
    """Replace the network with a stub that persists *payload* exactly where
    the ability pointed it — the on-disk contract stays fully exercised."""

    def fake(
        url: str,
        dest_path: str,
        *,
        profile: object,
        timeout: float,
        chunk_size: int = 8192,
        max_bytes: int | None = None,
    ) -> tuple[int, str]:
        if calls is not None:
            calls.append({"url": url, "dest": dest_path, "timeout": timeout, "max_bytes": max_bytes})
        if raising is not None:
            raise raising
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(payload)
        return len(payload), content_type

    monkeypatch.setattr("abilities.web_fetch.stream_to_file", fake)


def _run(url: str, **extra: object) -> ToolResult:
    params: dict[str, object] = {"url": url, **extra}
    bag = WebFetchParamsBag.from_params(params)
    if isinstance(bag, ToolResult):
        return bag
    return WebFetchAbility().run(bag)


class TestParamValidation:

    def test_missing_url_rejected(self) -> None:
        result = _run("")
        assert result.status == "error"
        assert result.code == "missing-params"

    def test_non_boolean_convert_rejected(self) -> None:
        result = _run(_URL, convert_to_markdown="yes")
        assert result.status == "error"
        assert result.code == "invalid-param"

    def test_convert_defaults_true_and_timeout_clamps(self) -> None:
        bag = built(WebFetchParamsBag.from_params({"url": _URL, "timeout": 999}))
        assert bag.convert_to_markdown is True
        assert bag.timeout == 120

    def test_timeout_reaches_the_fetch_in_seconds(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        _stub_fetch(monkeypatch, payload=_article_html(9), calls=calls)
        _run(_URL, timeout=2)
        assert calls[0]["timeout"] == 120.0


class TestBlockedUrls:

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/html,<p>x</p>"])
    def test_blocked_scheme_never_touches_network(
        self, url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        _stub_fetch(monkeypatch, calls=calls)
        result = _run(url)
        assert result.status == "error"
        assert result.code == "blocked-url"
        assert calls == []

    def test_ssrf_refusal_maps_to_blocked_url(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, raising=FetchBlocked("private host"))
        result = _run("https://10.0.0.1/admin")
        assert result.status == "error"
        assert result.code == "blocked-url"


class TestHtmlToMarkdown:

    def test_small_page_returns_markdown_and_persists(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=_article_html(9))
        result = _run(_URL)
        assert result.status == "success"
        assert isinstance(result.body, str)
        assert "# Monarch Migration Study" in result.body
        page = data_dir / "web" / "pages" / "example.com--studies-monarchs.md"
        assert page.read_text(encoding="utf-8") == result.body
        assert result.meta["url"] == _URL
        assert result.meta["path"] == str(page.absolute())

    def test_downloads_temp_is_removed_for_pages(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=_article_html(9))
        _run(_URL)
        downloads = data_dir / "downloads"
        assert not downloads.exists() or list(downloads.iterdir()) == []

    def test_refetch_overwrites_in_place(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=_article_html(9))
        _run(_URL)
        pages = data_dir / "web" / "pages"
        _stub_fetch(monkeypatch, payload=_article_html(12))
        second = _run(_URL)
        assert len(list(pages.iterdir())) == 1
        assert "Paragraph 12" in (pages / "example.com--studies-monarchs.md").read_text(
            encoding="utf-8"
        )
        assert second.status == "success"

    def test_over_20k_returns_pointer_but_persists_everything(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=_article_html(80))
        result = _run(_URL)
        assert result.status == "success"
        page = data_dir / "web" / "pages" / "example.com--studies-monarchs.md"
        assert result.body == (
            f"URL fetched and persisted at; `{page.absolute()}`. "
            "Use the `read` tool to load it in context in chunks"
        )
        assert len(page.read_text(encoding="utf-8")) > 20_000
        assert result.meta["path"] == str(page.absolute())

    def test_unextractable_page_is_loud_and_leaves_no_temp(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=b"<html><body></body></html>")
        result = _run(_URL)
        assert result.status == "error"
        assert result.code == "no-readable-content"
        downloads = data_dir / "downloads"
        assert not downloads.exists() or list(downloads.iterdir()) == []


class TestHtmlVerbatim:

    def test_small_page_returns_raw_html_at_html_path(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _article_html(2)
        _stub_fetch(monkeypatch, payload=payload)
        result = _run(_URL, convert_to_markdown=False)
        assert result.status == "success"
        assert result.body == payload.decode("utf-8")
        page = data_dir / "web" / "pages" / "example.com--studies-monarchs.html"
        assert page.read_bytes() == payload

    def test_string_false_parses_and_over_20k_points_at_read(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = b"<html><body>" + b"x" * 21_000 + b"</body></html>"
        _stub_fetch(monkeypatch, payload=payload)
        result = _run(_URL, convert_to_markdown="false")
        assert result.status == "success"
        assert isinstance(result.body, str)
        assert result.body.startswith("URL fetched and persisted at; `")
        page = data_dir / "web" / "pages" / "example.com--studies-monarchs.html"
        assert page.read_bytes() == payload


class TestNonHtml:

    def test_image_stays_in_downloads_and_points_at_vision(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=b"\x89PNG-bytes", content_type="image/png")
        result = _run("https://example.com/img/chart.png")
        assert result.status == "success"
        saved = data_dir / "downloads" / "example.com--img-chart-png.png"
        assert saved.read_bytes() == b"\x89PNG-bytes"
        assert isinstance(result.body, str)
        assert "vision" in result.body
        assert str(saved) in result.body
        assert result.meta["content_type"] == "image/png"
        assert result.meta["bytes"] == len(b"\x89PNG-bytes")

    def test_pdf_points_at_read(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_fetch(monkeypatch, payload=b"%PDF-1.7", content_type="application/pdf")
        result = _run("https://example.com/paper.pdf")
        assert result.status == "success"
        assert isinstance(result.body, str)
        assert "`read`" in result.body
        assert (data_dir / "downloads" / "example.com--paper-pdf.pdf").exists()

    def test_unknown_binary_gets_path_only(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=b"\x00\x01", content_type="application/octet-stream")
        result = _run("https://example.com/blob.bin")
        assert result.status == "success"
        assert isinstance(result.body, str)
        assert "read" not in result.body
        assert "vision" not in result.body
        assert str(data_dir / "downloads" / "example.com--blob-bin.bin") in result.body


class TestDownloadNaming:
    """Downloads carry the same host-qualified slug as web pages — the defect
    class where two hosts' ``report.pdf`` silently overwrote each other."""

    def test_same_basename_on_two_hosts_saves_two_files(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=b"%PDF-A", content_type="application/pdf")
        first = _run("https://sitea.com/reports/report.pdf")
        _stub_fetch(monkeypatch, payload=b"%PDF-B", content_type="application/pdf")
        second = _run("https://siteb.com/downloads/report.pdf")

        assert first.meta["path"] != second.meta["path"]
        assert Path(str(first.meta["path"])).read_bytes() == b"%PDF-A"
        assert Path(str(second.meta["path"])).read_bytes() == b"%PDF-B"

    def test_refetch_of_the_same_url_overwrites_in_place(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, payload=b"%PDF-old", content_type="application/pdf")
        first = _run("https://sitea.com/reports/report.pdf")
        _stub_fetch(monkeypatch, payload=b"%PDF-new", content_type="application/pdf")
        second = _run("https://sitea.com/reports/report.pdf")

        assert first.meta["path"] == second.meta["path"]
        assert Path(str(second.meta["path"])).read_bytes() == b"%PDF-new"

    def test_traversal_shaped_url_stays_flat_in_downloads(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``..`` path segment must not escape the downloads directory or
        resolve to a directory name the write would crash on."""
        _stub_fetch(monkeypatch, payload=b"\x00", content_type="application/octet-stream")
        result = _run("https://evil.com/foo/..")
        assert result.status == "success"
        saved = Path(str(result.meta["path"]))
        assert saved.parent == data_dir / "downloads"
        assert saved.name == "evil.com--foo"
        assert saved.exists()


class TestFetchErrors:

    def test_over_cap_is_too_large(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cap = 100 * 1024 * 1024
        _stub_fetch(monkeypatch, raising=DownloadTooLarge(cap))
        result = _run("https://example.com/huge.iso")
        assert result.status == "error"
        assert result.code == "too-large"
        assert "100 MB" in str(result.body)
        assert result.meta["max_bytes"] == cap

    def test_network_failure_is_download_failed(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_fetch(monkeypatch, raising=requests.ConnectionError("boom"))
        result = _run(_URL)
        assert result.status == "error"
        assert result.code == "download-failed"
