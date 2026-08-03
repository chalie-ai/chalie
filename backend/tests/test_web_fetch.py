"""Feature tests: the ``web_fetch`` ability's fetch-and-persist contract.

Only the network edge is stubbed: a fake ``stream_to_file`` writes the canned
payload to the exact destination the ability chose. Everything downstream runs
for real on a tmp-rooted data dir — trafilatura conversion, the embedded-media
post-pass, FileMapperService persistence, the 20 000-character gate, and
temp-file cleanup — so these tests fail when any of those seams break.
"""

from pathlib import Path

import pytest
import requests

from abilities._result import ToolResult
from abilities.web_fetch import WebFetchAbility
from contracts.params.web_fetch_params_bag import WebFetchParamsBag
from exceptions import DownloadTooLarge
from services.file_mapper_service import FileMapperService
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

_URL = "https://example.com/studies/monarchs"

# Long enough that trafilatura's minimum-size heuristics see a real article.
_PARAGRAPH = (
    "The migration of monarch butterflies across North America remains one of "
    "the most studied phenomena in entomology, spanning thousands of kilometres "
    "and multiple generations in a single seasonal cycle, with each cohort "
    "navigating by a combination of solar position and an inherited magnetic "
    "compass that researchers only partially understand."
)


def _article_html(paragraphs: int, *, media: bool = True) -> bytes:
    body = "\n".join(f"<p>Paragraph {i}: {_PARAGRAPH}</p>" for i in range(1, paragraphs + 1))
    iframe = '<iframe src="/embed/42"></iframe>' if media else ""
    return (
        "<html><head><title>Monarch Migration Study</title></head><body><article>"
        f"<h1>Monarch Migration Study</h1>{iframe}{body}</article></body></html>"
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
            calls.append({"url": url, "timeout": timeout, "max_bytes": max_bytes})
        if raising is not None:
            raise raising
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(payload)
        return len(payload), content_type

    monkeypatch.setattr("abilities.web_fetch.stream_to_file", fake)


def _run(url: str, **extra: object) -> ToolResult:
    bag = WebFetchParamsBag.from_params({"url": url, **extra})
    return bag if isinstance(bag, ToolResult) else WebFetchAbility().run(bag)


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"url": ""}, "missing-params"),
        ({"url": _URL, "convert_to_markdown": "yes"}, "invalid-param"),
    ],
)
def test_invalid_params_rejected(params: dict[str, object], code: str) -> None:
    result = WebFetchParamsBag.from_params(params)
    assert isinstance(result, ToolResult)
    assert (result.status, result.code) == ("error", code)


def test_defaults_clamp_and_timeout_reaches_fetch_in_seconds(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag = built(WebFetchParamsBag.from_params({"url": _URL, "timeout": 999}))
    assert (bag.convert_to_markdown, bag.timeout) == (True, 120)
    calls: list[dict[str, object]] = []
    _stub_fetch(monkeypatch, payload=_article_html(9), calls=calls)
    _run(_URL, timeout=2)
    assert calls[0]["timeout"] == 120.0


@pytest.mark.parametrize(
    ("raising", "code"),
    [
        (DownloadTooLarge(100 * 1024 * 1024), "too-large"),
        (requests.ConnectionError("boom"), "download-failed"),
    ],
)
def test_fetch_failures_map_to_stable_codes(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, raising: BaseException, code: str
) -> None:
    _stub_fetch(monkeypatch, raising=raising)
    result = _run(_URL)
    assert (result.status, result.code) == ("error", code)
    if code == "too-large":
        assert "100 MB" in str(result.body)


def test_small_page_returns_markdown_with_media_and_persists(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_fetch(monkeypatch, payload=_article_html(9))
    result = _run(_URL)
    assert result.status == "success"
    assert isinstance(result.body, str)
    assert "# Monarch Migration Study" in result.body
    # The embedded-media post-pass resolves the iframe src against the page URL.
    assert "(https://example.com/embed/42)" in result.body
    page = data_dir / "web" / "pages" / "example.com--studies-monarchs.md"
    assert page.read_text(encoding="utf-8") == result.body
    assert (result.meta["url"], result.meta["path"]) == (_URL, str(page.absolute()))
    assert list((data_dir / "downloads").iterdir()) == []  # fetch buffer removed


def test_refetch_overwrites_in_place(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_fetch(monkeypatch, payload=_article_html(9))
    _run(_URL)
    _stub_fetch(monkeypatch, payload=_article_html(12, media=False))
    _run(_URL)
    pages = list((data_dir / "web" / "pages").iterdir())
    assert len(pages) == 1
    text = pages[0].read_text(encoding="utf-8")
    assert "Paragraph 12" in text
    assert "## Embedded media" not in text  # no media elements, no section


def test_over_20k_returns_pointer_but_persists_everything(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_fetch(monkeypatch, payload=_article_html(80))
    result = _run(_URL)
    page = data_dir / "web" / "pages" / "example.com--studies-monarchs.md"
    assert result.status == "success"
    assert result.body == (
        f"URL fetched and persisted at; `{page.absolute()}`. "
        "Use the `read` tool to load it in context in chunks"
    )
    assert len(page.read_text(encoding="utf-8")) > 20_000


def test_unextractable_page_is_loud_and_leaves_no_temp(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_fetch(monkeypatch, payload=b"<html><body></body></html>")
    result = _run(_URL)
    assert (result.status, result.code) == ("error", "no-readable-content")
    assert list((data_dir / "downloads").iterdir()) == []


def test_convert_false_keeps_verbatim_html(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _article_html(2)
    _stub_fetch(monkeypatch, payload=payload)
    result = _run(_URL, convert_to_markdown="false")  # the string form parses too
    assert result.status == "success"
    assert result.body == payload.decode("utf-8")
    page = data_dir / "web" / "pages" / "example.com--studies-monarchs.html"
    assert page.read_bytes() == payload


@pytest.mark.parametrize(
    ("url", "content_type", "saved", "tool"),
    [
        ("https://example.com/img/chart.png", "image/png", "example.com--img-chart-png.png", "vision"),
        ("https://example.com/paper.pdf", "application/pdf", "example.com--paper-pdf.pdf", "`read`"),
        ("https://example.com/blob.bin", "application/octet-stream", "example.com--blob-bin.bin", None),
    ],
)
def test_non_html_stays_in_downloads_with_per_mime_route(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    content_type: str,
    saved: str,
    tool: str | None,
) -> None:
    _stub_fetch(monkeypatch, payload=b"\x00payload", content_type=content_type)
    result = _run(url)
    assert result.status == "success"
    path = data_dir / "downloads" / saved
    assert path.read_bytes() == b"\x00payload"
    assert isinstance(result.body, str)
    assert str(path) in result.body
    if tool is None:
        assert "vision" not in result.body
        assert "read" not in result.body
    else:
        assert tool in result.body
    assert (result.meta["content_type"], result.meta["bytes"]) == (content_type, 8)


def test_download_names_are_host_qualified_and_traversal_safe(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two hosts' ``report.pdf`` must save as two files (the silent-overwrite
    defect the PR review caught), and a ``..`` segment must slug flat instead
    of escaping the downloads directory."""
    _stub_fetch(monkeypatch, payload=b"%PDF-A", content_type="application/pdf")
    first = _run("https://sitea.com/reports/report.pdf")
    _stub_fetch(monkeypatch, payload=b"%PDF-B", content_type="application/pdf")
    second = _run("https://siteb.com/downloads/report.pdf")
    assert first.meta["path"] != second.meta["path"]
    assert Path(str(first.meta["path"])).read_bytes() == b"%PDF-A"
    assert Path(str(second.meta["path"])).read_bytes() == b"%PDF-B"

    _stub_fetch(monkeypatch, payload=b"\x00", content_type="application/octet-stream")
    traversal = _run("https://evil.com/foo/..")
    saved_path = Path(str(traversal.meta["path"]))
    assert (saved_path.parent, saved_path.name) == (data_dir / "downloads", "evil.com--foo")
    assert saved_path.exists()
