"""Unit tests for SearchFilesAbility — glob and grep on the real filesystem."""

import os
from pathlib import Path

import pytest

from abilities.search_files import SearchFilesAbility

pytestmark = pytest.mark.unit


def _run(action: str, query: str, directory: str | None = None, **extra) -> str:
    """Execute SearchFilesAbility and return the plain-text result body.

    ``run()`` returns a ``ToolResult`` whose ``body`` is the plain-text payload;
    the dispatcher (not the ability) formats the ``[search_files(...)]`` envelope.
    """
    params: dict = {"action": action, "query": query, **extra}
    if directory is not None:
        params["directory"] = directory
    return SearchFilesAbility().run(params).body


# ── validation ────────────────────────────────────────────────────


def test_invalid_action_returns_error():
    out = _run("nope", "*.py", "/tmp")
    assert "Invalid action" in out


def test_missing_query_returns_error():
    out = _run("glob", "", "/tmp")
    assert "required" in out


def test_max_files_boundary_validation(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    for bad_value in (0, 201):
        out = _run("glob", "*.py", str(tmp_path), max_files=bad_value)
        assert "must be between 1 and 200" in out
        assert str(bad_value) in out


def test_context_lines_over_20_returns_error(tmp_path: Path):
    (tmp_path / "a.py").write_text("NEEDLE\n")
    out = _run("grep", "NEEDLE", str(tmp_path), context_lines=21)
    assert "must be between 0 and 20" in out
    assert "21" in out


def test_directory_not_found_returns_error():
    out = _run("glob", "*.py", "/nonexistent_dir_xyz_12345")
    assert "not found" in out.lower()


def test_not_a_directory_returns_error(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    out = _run("glob", "*.py", str(f))
    assert "Not a directory" in out


# ── glob ──────────────────────────────────────────────────────────


def test_glob_returns_newline_separated_absolute_paths(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("# a")
    (tmp_path / "beta.txt").write_text("# b")
    (tmp_path / "gamma.py").write_text("# c")

    out = _run("glob", "*.py", str(tmp_path))
    paths = out.strip().split("\n")
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["alpha.py", "gamma.py"]
    for p in paths:
        assert os.path.isabs(p)


def test_glob_recursive(tmp_path: Path):
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "found.log").write_text("x")
    (tmp_path / "top.log").write_text("x")

    out = _run("glob", "**/*.log", str(tmp_path))
    names = sorted(os.path.basename(p) for p in out.strip().split("\n"))
    assert names == ["found.log", "top.log"]


def test_glob_no_match_returns_message(tmp_path: Path):
    out = _run("glob", "*.nonexistent", str(tmp_path))
    assert out == "No files matched."


def test_glob_max_files_default_and_override(tmp_path: Path):
    for i in range(15):
        (tmp_path / f"f{i:02d}.dat").write_text("x")

    default_out = _run("glob", "*.dat", str(tmp_path))
    assert len(default_out.strip().split("\n")) == 10

    override_out = _run("glob", "*.dat", str(tmp_path), max_files=5)
    assert len(override_out.strip().split("\n")) == 5


# ── grep ──────────────────────────────────────────────────────────


def test_grep_returns_file_path_with_context(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 12))
    (tmp_path / "a.py").write_text(content)

    out = _run("grep", "line 6", str(tmp_path), context_lines=2)
    lines = out.strip().split("\n")
    assert lines[0].endswith("a.py")
    assert os.path.isabs(lines[0])
    assert any("ln 4:" in ln for ln in lines)
    assert any("ln 6: line 6" in ln for ln in lines)
    assert any("ln 8:" in ln for ln in lines)


def test_grep_files_separated_by_dashes(tmp_path: Path):
    (tmp_path / "a.py").write_text("NEEDLE\n")
    (tmp_path / "b.py").write_text("NEEDLE\n")

    out = _run("grep", "NEEDLE", str(tmp_path))
    assert "----" in out
    blocks = out.split("\n----\n")
    assert len(blocks) == 2


def test_grep_no_match_returns_message(tmp_path: Path):
    (tmp_path / "a.py").write_text("nothing here\n")
    out = _run("grep", "NONEXISTENT", str(tmp_path))
    assert out == "No matches found."


def test_grep_context_lines_default_and_override(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 20))
    (tmp_path / "f.py").write_text(content)

    default_out = _run("grep", "line 10", str(tmp_path))
    default_lines = default_out.strip().split("\n")
    assert any("ln 5:" in ln for ln in default_lines)
    assert any("ln 10: line 10" in ln for ln in default_lines)
    assert any("ln 15:" in ln for ln in default_lines)

    override_out = _run("grep", "line 10", str(tmp_path), context_lines=1)
    override_lines = [ln for ln in override_out.strip().split("\n") if ln.startswith("ln ")]
    assert len(override_lines) == 3
    assert any("ln 9:" in ln for ln in override_lines)
    assert any("ln 10:" in ln for ln in override_lines)
    assert any("ln 11:" in ln for ln in override_lines)


def test_grep_max_files_default_and_override(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("NEEDLE\n")

    default_out = _run("grep", "NEEDLE", str(tmp_path))
    assert len(default_out.split("\n----\n")) == 5

    override_out = _run("grep", "NEEDLE", str(tmp_path), max_files=3)
    assert len(override_out.split("\n----\n")) == 3


def test_grep_merges_overlapping_context(tmp_path: Path):
    (tmp_path / "f.py").write_text("a\nMATCH1\nc\nMATCH2\ne\nf\n")
    out = _run("grep", "MATCH", str(tmp_path), context_lines=1)
    ln_lines = [ln for ln in out.strip().split("\n") if ln.startswith("ln ")]
    assert sum(1 for ln in ln_lines if "ln 3: c" in ln) == 1


def test_grep_supports_regex(tmp_path: Path):
    (tmp_path / "f.py").write_text("foo123bar\nhello\nfoo456bar\n")
    out = _run("grep", r"foo\d+bar", str(tmp_path), context_lines=0)
    assert "ln 1: foo123bar" in out
    assert "ln 3: foo456bar" in out


def test_grep_invalid_regex_returns_error(tmp_path: Path):
    out = _run("grep", "[invalid", str(tmp_path))
    assert "Invalid regex" in out


# ── no restrictions ───────────────────────────────────────────────


def test_no_blocked_paths():
    out = _run("glob", "*.conf", "/etc")
    assert "blocked" not in out.lower()


def test_default_directory_is_root():
    out = _run("glob", "*.this_extension_should_not_exist_xyz", "/tmp")
    assert out == "No files matched."
