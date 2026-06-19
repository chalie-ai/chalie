import os
from pathlib import Path

import pytest

from abilities._result import ToolResult
from abilities.search_files import SearchFilesAbility

pytestmark = pytest.mark.unit


def _run(action: str, query: str, directory: str | None = None, **extra) -> ToolResult:
    params: dict = {"action": action, "query": query, **extra}
    if directory is not None:
        params["directory"] = directory
    return SearchFilesAbility().run(params)


def _glob_files(tr: ToolResult) -> list[str]:
    """The list of matched paths from a glob success body."""
    assert tr.status == "success"
    assert isinstance(tr.body, dict)
    return tr.body["files"]


def _grep_rows(tr: ToolResult) -> list[dict]:
    """The match rows from a grep success body — a bare list when untruncated, or
    the ``matches`` field of a dict body when a note/truncation rides along."""
    assert tr.status == "success"
    if isinstance(tr.body, list):
        return tr.body
    assert isinstance(tr.body, dict)
    return tr.body["matches"]


# ── validation ────────────────────────────────────────────────────


def test_invalid_action_returns_error():
    tr = _run("nope", "*.py", "/tmp")
    assert tr.status == "error"
    assert tr.code == "unknown-action"
    assert tr.valid == ("glob", "grep")


def test_missing_query_returns_error():
    tr = _run("glob", "", "/tmp")
    assert tr.status == "error"
    assert tr.code == "empty-query"
    assert "required" in str(tr.body)


def test_max_files_boundary_validation(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    for bad_value in (0, 201):
        tr = _run("glob", "*.py", str(tmp_path), max_files=bad_value)
        assert tr.status == "error"
        assert tr.code == "invalid-param"
        assert "must be between 1 and 200" in str(tr.body)
        assert str(bad_value) in str(tr.body)


def test_context_lines_over_20_returns_error(tmp_path: Path):
    (tmp_path / "a.py").write_text("NEEDLE\n")
    tr = _run("grep", "NEEDLE", str(tmp_path), context_lines=21)
    assert tr.status == "error"
    assert tr.code == "invalid-param"
    assert "must be between 0 and 20" in str(tr.body)
    assert "21" in str(tr.body)


def test_directory_not_found_returns_error():
    tr = _run("glob", "*.py", "/nonexistent_dir_xyz_12345")
    assert tr.status == "error"
    assert tr.code == "directory-not-found"
    assert "not found" in str(tr.body).lower()


def test_not_a_directory_returns_error(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    tr = _run("glob", "*.py", str(f))
    assert tr.status == "error"
    assert tr.code == "not-a-directory"
    assert "Not a directory" in str(tr.body)


# ── glob ──────────────────────────────────────────────────────────


def test_glob_returns_absolute_paths(tmp_path: Path):
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


def test_glob_recursive(tmp_path: Path):
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "found.log").write_text("x")
    (tmp_path / "top.log").write_text("x")

    tr = _run("glob", "**/*.log", str(tmp_path))
    names = sorted(os.path.basename(p) for p in _glob_files(tr))
    assert names == ["found.log", "top.log"]


def test_glob_no_match_is_success_with_broaden_note(tmp_path: Path):
    tr = _run("glob", "*.nonexistent", str(tmp_path))
    assert tr.status == "success"
    assert tr.code is None
    assert _glob_files(tr) == []
    assert tr.meta["count"] == 0
    assert "broaden" in str(tr.body).lower()


def test_glob_max_files_default_and_override(tmp_path: Path):
    for i in range(15):
        (tmp_path / f"f{i:02d}.dat").write_text("x")

    default_tr = _run("glob", "*.dat", str(tmp_path))
    assert len(_glob_files(default_tr)) == 10
    assert default_tr.meta["truncated"] is True

    override_tr = _run("glob", "*.dat", str(tmp_path), max_files=5)
    assert len(_glob_files(override_tr)) == 5
    assert override_tr.meta["truncated"] is True


# ── grep ──────────────────────────────────────────────────────────


def test_grep_returns_rows_with_context(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 12))
    (tmp_path / "a.py").write_text(content)

    tr = _run("grep", "line 6", str(tmp_path), context_lines=2)
    rows = _grep_rows(tr)
    assert len(rows) == 1
    row = rows[0]
    assert row["file"].endswith("a.py")
    assert os.path.isabs(row["file"])
    assert row["line"] == 6
    assert row["text"] == "line 6"
    assert "line 4" in row["context"]
    assert "line 6" in row["context"]
    assert "line 8" in row["context"]


def test_grep_rows_span_multiple_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("NEEDLE\n")
    (tmp_path / "b.py").write_text("NEEDLE\n")

    tr = _run("grep", "NEEDLE", str(tmp_path))
    rows = _grep_rows(tr)
    files = {os.path.basename(r["file"]) for r in rows}
    assert files == {"a.py", "b.py"}
    assert tr.meta["count"] == 2


def test_grep_no_match_is_success_with_broaden_note(tmp_path: Path):
    (tmp_path / "a.py").write_text("nothing here\n")
    tr = _run("grep", "NONEXISTENT", str(tmp_path))
    assert tr.status == "success"
    assert tr.code is None
    assert _grep_rows(tr) == []
    assert tr.meta["count"] == 0
    assert "broaden" in str(tr.body).lower()


def test_grep_context_lines_default_and_override(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 20))
    (tmp_path / "f.py").write_text(content)

    default_tr = _run("grep", "line 10", str(tmp_path))
    default_row = _grep_rows(default_tr)[0]
    assert "line 5" in default_row["context"]
    assert "line 10" in default_row["context"]
    assert "line 15" in default_row["context"]

    override_tr = _run("grep", "line 10", str(tmp_path), context_lines=1)
    override_row = _grep_rows(override_tr)[0]
    ctx_lines = override_row["context"].split("\n")
    assert ctx_lines == ["line 9", "line 10", "line 11"]


def test_grep_max_files_default_and_override(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("NEEDLE\n")

    default_tr = _run("grep", "NEEDLE", str(tmp_path))
    default_files = {os.path.basename(r["file"]) for r in _grep_rows(default_tr)}
    assert len(default_files) == 5
    assert default_tr.meta["truncated"] is True

    override_tr = _run("grep", "NEEDLE", str(tmp_path), max_files=3)
    override_files = {os.path.basename(r["file"]) for r in _grep_rows(override_tr)}
    assert len(override_files) == 3
    assert override_tr.meta["truncated"] is True


def test_grep_emits_one_row_per_match(tmp_path: Path):
    (tmp_path / "f.py").write_text("a\nMATCH1\nc\nMATCH2\ne\nf\n")
    tr = _run("grep", "MATCH", str(tmp_path), context_lines=1)
    rows = _grep_rows(tr)
    assert [r["line"] for r in rows] == [2, 4]
    assert [r["text"] for r in rows] == ["MATCH1", "MATCH2"]


def test_grep_supports_regex(tmp_path: Path):
    (tmp_path / "f.py").write_text("foo123bar\nhello\nfoo456bar\n")
    tr = _run("grep", r"foo\d+bar", str(tmp_path), context_lines=0)
    rows = _grep_rows(tr)
    texts = sorted(r["text"] for r in rows)
    assert texts == ["foo123bar", "foo456bar"]
    # context_lines=0 → no context field on the rows.
    assert all("context" not in r for r in rows)


def test_grep_invalid_regex_returns_error(tmp_path: Path):
    tr = _run("grep", "[invalid", str(tmp_path))
    assert tr.status == "error"
    assert tr.code == "invalid-regex"
    assert "Invalid regex" in str(tr.body)


# ── no restrictions ───────────────────────────────────────────────


def test_no_blocked_paths():
    tr = _run("glob", "*.conf", "/etc")
    assert tr.status == "success"
    assert "blocked" not in str(tr.body).lower()


def test_default_directory_is_root():
    tr = _run("glob", "*.this_extension_should_not_exist_xyz", "/tmp")
    assert tr.status == "success"
    assert _glob_files(tr) == []
    assert tr.meta["count"] == 0
