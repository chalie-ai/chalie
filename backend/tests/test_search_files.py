import os
import time
from pathlib import Path
from typing import cast

import pytest

from abilities import search_files
from abilities._result import ToolResult
from abilities.search_files import SearchFilesAbility
from contracts.params.search_files_params_bag import SearchFilesParamsBag

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the /tmp scratchpad to a per-test SIBLING of the search root so the
    real cache constant is exercised end-to-end without tests polluting each other
    (or the host /tmp). A sibling — never under ``tmp_path`` — guarantees a walk of
    the search root can never pick up a cache file.
    """
    cache = f"{tmp_path}_cache"
    os.makedirs(cache, exist_ok=True)
    monkeypatch.setattr(search_files, "_CACHE_DIR", cache)


def _run(action: str, query: str, directory: str | None = None, **extra: object) -> ToolResult:
    params: dict[str, object] = {"action": action, "query": query, **extra}
    if directory is not None:
        params["directory"] = directory
    # Build the bag exactly as the dispatch seam does — from_params returns the
    # error ToolResult directly, so the error-path tests still see the envelope.
    bag = SearchFilesParamsBag.from_params(params)
    if isinstance(bag, ToolResult):
        return bag
    return SearchFilesAbility().run(bag)


def _glob_files(tr: ToolResult) -> list[str]:
    """The list of matched paths from a glob success body."""
    assert tr.status == "success"
    assert isinstance(tr.body, dict)
    return cast(list[str], tr.body["files"])


def _grep_rows(tr: ToolResult) -> list[dict[str, object]]:
    """The match rows from a grep success body (always under ``matches``)."""
    assert tr.status == "success"
    assert isinstance(tr.body, dict)
    return cast(list[dict[str, object]], tr.body["matches"])


# ── validation ────────────────────────────────────────────────────


def test_page_below_one_clamps_to_first_page(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    tr = _run("glob", "*.py", str(tmp_path), page=0)
    assert tr.status == "success"
    assert tr.meta["page"] == 1
    assert len(_glob_files(tr)) == 1


def test_context_lines_over_max_clamps_to_max(tmp_path: Path) -> None:
    # 61 lines with the match in the middle: the clamped width of 20 renders
    # 20 above + the match + 20 below = 41 context lines (an unclamped 21
    # would render 43).
    content = "\n".join(f"line {i}" for i in range(1, 62))
    (tmp_path / "a.py").write_text(content)
    tr = _run("grep", "line 31", str(tmp_path), context_lines=21)
    rows = _grep_rows(tr)
    assert len(rows) == 1
    assert len(cast(str, rows[0]["context"]).split("\n")) == 41


def test_directory_not_found_returns_error() -> None:
    tr = _run("glob", "*.py", "/nonexistent_dir_xyz_12345")
    assert tr.status == "error"
    assert tr.code == "directory-not-found"
    assert "not found" in str(tr.body).lower()


def test_not_a_directory_returns_error(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    tr = _run("glob", "*.py", str(f))
    assert tr.status == "error"
    assert tr.code == "not-a-directory"
    assert "Not a directory" in str(tr.body)


# ── glob ──────────────────────────────────────────────────────────


def test_glob_returns_absolute_paths(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("# a")
    (tmp_path / "beta.txt").write_text("# b")
    (tmp_path / "gamma.py").write_text("# c")

    tr = _run("glob", "*.py", str(tmp_path))
    paths = _glob_files(tr)
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["alpha.py", "gamma.py"]
    for p in paths:
        assert os.path.isabs(p)
    assert tr.meta["count"] == 2


def test_glob_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "found.log").write_text("x")
    (tmp_path / "top.log").write_text("x")

    tr = _run("glob", "**/*.log", str(tmp_path))
    names = sorted(os.path.basename(p) for p in _glob_files(tr))
    assert names == ["found.log", "top.log"]


def test_glob_no_match_is_success_with_broaden_note(tmp_path: Path) -> None:
    tr = _run("glob", "*.nonexistent", str(tmp_path))
    assert tr.status == "success"
    assert tr.code is None
    assert _glob_files(tr) == []
    assert tr.meta["count"] == 0
    assert "broaden" in str(tr.body).lower()


# ── pagination (5 results per page) ───────────────────────────────


def test_glob_paginates_five_per_page(tmp_path: Path) -> None:
    for i in range(12):
        (tmp_path / f"f{i:02d}.dat").write_text("x")

    p1 = _run("glob", "*.dat", str(tmp_path))  # page defaults to 1
    assert len(_glob_files(p1)) == 5
    assert p1.meta["page"] == 1
    assert p1.meta["page_count"] == 3
    assert p1.meta["count"] == 12
    assert "You are currently on page 1 from 3 pages" in str(p1.body)

    p2 = _run("glob", "*.dat", str(tmp_path), page=2)
    p3 = _run("glob", "*.dat", str(tmp_path), page=3)
    assert len(_glob_files(p2)) == 5
    assert len(_glob_files(p3)) == 2
    # every file appears exactly once across the three pages
    seen = set(_glob_files(p1)) | set(_glob_files(p2)) | set(_glob_files(p3))
    assert len(seen) == 12

    # an out-of-range page clamps to the last page rather than erroring/emptying
    clamp = _run("glob", "*.dat", str(tmp_path), page=99)
    assert clamp.meta["page"] == 3
    assert len(_glob_files(clamp)) == 2


def test_single_page_carries_no_pagination_note(tmp_path: Path) -> None:
    (tmp_path / "a.dat").write_text("x")
    tr = _run("glob", "*.dat", str(tmp_path))
    assert tr.meta["page_count"] == 1
    assert "currently on page" not in str(tr.body)


def test_grep_paginates_one_row_per_matched_line(tmp_path: Path) -> None:
    # a single file with 12 matching lines = 12 results = 3 pages (NOT 1 result).
    (tmp_path / "log.txt").write_text("".join(f"hit {i}\n" for i in range(12)))

    p1 = _run("grep", "hit", str(tmp_path), context_lines=0)
    assert len(_grep_rows(p1)) == 5
    assert p1.meta["count"] == 12
    assert p1.meta["page_count"] == 3
    assert "You are currently on page 1 from 3 pages" in str(p1.body)

    p3 = _run("grep", "hit", str(tmp_path), page=3, context_lines=0)
    rows3 = _grep_rows(p3)
    assert [r["line"] for r in rows3] == [11, 12]  # the trailing 2 of 12 matched lines


# ── scratchpad cache (/tmp, md5 of query, 10-min TTL) ─────────────


def test_cache_serves_later_pages_without_rewalking(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("".join(f"hit {i}\n" for i in range(12)))
    _run("grep", "hit", str(tmp_path), context_lines=0)  # page 1 → walks, writes cache

    # Gut the tree: a fresh walk would now find ZERO matches. Page 2 must still
    # return the cached rows — proof the second call read the scratchpad.
    (tmp_path / "log.txt").write_text("nothing here now\n")
    p2 = _run("grep", "hit", str(tmp_path), page=2, context_lines=0)
    assert len(_grep_rows(p2)) == 5

    root = Path(str(tmp_path)).resolve()
    assert os.path.exists(search_files._cache_path("grep", "hit", 0, root))


def test_stale_cache_is_discarded_and_rebuilt(tmp_path: Path) -> None:
    (tmp_path / "log.txt").write_text("hit one\n")
    _run("grep", "hit", str(tmp_path), context_lines=0)  # writes cache (1 row)
    cache_file = search_files._cache_path("grep", "hit", 0, Path(str(tmp_path)).resolve())
    assert os.path.exists(cache_file)

    # age the cache past its 10-minute TTL, then grow the tree
    old = time.time() - (search_files._CACHE_TTL_S + 60)
    os.utime(cache_file, (old, old))
    (tmp_path / "log.txt").write_text("hit one\nhit two\n")

    rebuilt = _run("grep", "hit", str(tmp_path), context_lines=0)
    assert len(_grep_rows(rebuilt)) == 2  # stale cache discarded → fresh walk saw both lines


# ── special-file safety (the hang fix) ────────────────────────────


def test_skip_prunes_special_filesystem_prefixes() -> None:
    assert search_files._skip("/proc")
    assert search_files._skip("/proc/123/maps")
    assert search_files._skip("/sys/kernel")
    assert search_files._skip("/dev/null")
    assert not search_files._skip("/home/user/proc_notes")  # substring, not a prefix
    assert not search_files._skip("/etc/hosts")


def test_grep_skips_non_regular_files_without_hanging(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_text("hello world\n")
    os.mkfifo(tmp_path / "pipe")  # a FIFO: open()-for-read blocks forever with no writer

    tr = _run("grep", "hello", str(tmp_path), context_lines=0)  # must NOT park on the fifo
    rows = _grep_rows(tr)
    assert [r["text"] for r in rows] == ["hello world"]  # real file found, pipe skipped


# ── grep content / context ────────────────────────────────────────


def test_grep_returns_rows_with_context(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(1, 12))
    (tmp_path / "a.py").write_text(content)

    tr = _run("grep", "line 6", str(tmp_path), context_lines=2)
    rows = _grep_rows(tr)
    assert len(rows) == 1
    row = rows[0]
    assert cast(str, row["file"]).endswith("a.py")
    assert os.path.isabs(cast(str, row["file"]))
    assert row["line"] == 6
    assert row["text"] == "line 6"
    assert "line 4" in cast(str, row["context"])
    assert "line 6" in cast(str, row["context"])
    assert "line 8" in cast(str, row["context"])


def test_grep_rows_span_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("NEEDLE\n")
    (tmp_path / "b.py").write_text("NEEDLE\n")

    tr = _run("grep", "NEEDLE", str(tmp_path))
    rows = _grep_rows(tr)
    files = {os.path.basename(cast(str, r["file"])) for r in rows}
    assert files == {"a.py", "b.py"}
    assert tr.meta["count"] == 2


def test_grep_no_match_is_success_with_broaden_note(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("nothing here\n")
    tr = _run("grep", "NONEXISTENT", str(tmp_path))
    assert tr.status == "success"
    assert tr.code is None
    assert _grep_rows(tr) == []
    assert tr.meta["count"] == 0
    assert "broaden" in str(tr.body).lower()


def test_grep_context_lines_default_and_override(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(1, 20))
    (tmp_path / "f.py").write_text(content)

    default_tr = _run("grep", "line 10", str(tmp_path))
    default_row = _grep_rows(default_tr)[0]
    assert "line 5" in cast(str, default_row["context"])
    assert "line 10" in cast(str, default_row["context"])
    assert "line 15" in cast(str, default_row["context"])

    override_tr = _run("grep", "line 10", str(tmp_path), context_lines=1)
    override_row = _grep_rows(override_tr)[0]
    ctx_lines = cast(str, override_row["context"]).split("\n")
    assert ctx_lines == ["line 9", "line 10", "line 11"]


def test_grep_emits_one_row_per_match(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("a\nMATCH1\nc\nMATCH2\ne\nf\n")
    tr = _run("grep", "MATCH", str(tmp_path), context_lines=1)
    rows = _grep_rows(tr)
    assert [r["line"] for r in rows] == [2, 4]
    assert [r["text"] for r in rows] == ["MATCH1", "MATCH2"]


def test_grep_supports_regex(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("foo123bar\nhello\nfoo456bar\n")
    tr = _run("grep", r"foo\d+bar", str(tmp_path), context_lines=0)
    rows = _grep_rows(tr)
    texts = sorted(cast(str, r["text"]) for r in rows)
    assert texts == ["foo123bar", "foo456bar"]
    # context_lines=0 → no context field on the rows.
    assert all("context" not in r for r in rows)


def test_grep_invalid_regex_returns_error(tmp_path: Path) -> None:
    tr = _run("grep", "[invalid", str(tmp_path))
    assert tr.status == "error"
    assert tr.code == "invalid-regex"
    assert "Invalid regex" in str(tr.body)


