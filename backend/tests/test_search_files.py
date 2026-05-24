"""Feature tests for SearchFilesAbility — glob and grep on the real filesystem."""

import json
import os
from pathlib import Path

import pytest

from abilities.search_files import SearchFilesAbility

pytestmark = pytest.mark.integration


def _run(action: str, query: str, directory: str | None = None, **extra) -> dict:
    params: dict = {"action": action, "query": query, **extra}
    if directory is not None:
        params["directory"] = directory
    raw = SearchFilesAbility().execute("user", params, None)
    return json.loads(raw["text"])


def test_glob_finds_files_by_basename_pattern(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("# a")
    (tmp_path / "beta.txt").write_text("# b")
    (tmp_path / "gamma.py").write_text("# c")

    out = _run("glob", "*.py", str(tmp_path))
    names = sorted(os.path.basename(p) for p in out["paths"])
    assert names == ["alpha.py", "gamma.py"]


def test_glob_finds_files_recursively(tmp_path: Path):
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "found.log").write_text("x")
    (tmp_path / "top.log").write_text("x")

    out = _run("glob", "**/*.log", str(tmp_path))
    names = sorted(os.path.basename(p) for p in out["paths"])
    assert names == ["found.log", "top.log"]


def test_glob_truncates_when_exceeding_max_files(tmp_path: Path):
    for i in range(20):
        (tmp_path / f"f{i}.dat").write_text("x")
    out = _run("glob", "*.dat", str(tmp_path), max_files=5)
    assert out["count"] == 5
    assert out["truncated"] is True


def test_grep_returns_matched_line_with_surrounding_context(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "line 1\nline 2\nline 3\nline 4\nTARGET\nline 6\nline 7\nline 8\nline 9\nline 10\n"
    )
    out = _run("grep", "TARGET", str(tmp_path), context_lines=3)
    snippet = out["results"][0]["matches"][0]
    assert "ln 2: line 2" in snippet
    assert "ln 5: TARGET" in snippet
    assert "ln 8: line 8" in snippet


def test_grep_merges_overlapping_context_for_adjacent_matches(tmp_path: Path):
    (tmp_path / "f.py").write_text("a\nMATCH1\nc\nMATCH2\ne\nf\n")
    out = _run("grep", "MATCH", str(tmp_path), context_lines=1)
    result = out["results"][0]
    assert len(result["matches"]) == 1
    snippet = result["matches"][0]
    assert "ln 2: MATCH1" in snippet
    assert "ln 4: MATCH2" in snippet
    assert snippet.count("ln 3: c") == 1


def test_grep_shows_all_matches_across_a_file(tmp_path: Path):
    (tmp_path / "f.py").write_text("AAA\nbbb\nccc\nddd\neee\nfff\nggg\nAAA\nhhh\niii\n")
    out = _run("grep", "AAA", str(tmp_path), context_lines=1)
    result = out["results"][0]
    assert len(result["matches"]) == 2
    assert "ln 1: AAA" in result["matches"][0]
    assert "ln 8: AAA" in result["matches"][1]


def test_grep_skips_files_larger_than_5mb(tmp_path: Path):
    (tmp_path / "big.txt").write_bytes(b"NEEDLE\n" + b"x" * (5 * 1024 * 1024 + 100))
    (tmp_path / "small.txt").write_text("NEEDLE here\n")
    out = _run("grep", "NEEDLE", str(tmp_path))
    files = [os.path.basename(r["file"]) for r in out["results"]]
    assert "small.txt" in files
    assert "big.txt" not in files


def test_symlink_loop_terminates_safely(tmp_path: Path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "real.py").write_text("x")
    try:
        os.symlink(str(inner), str(inner / "loop"))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    out = _run("glob", "*.py", str(tmp_path))
    assert out["count"] == 1
    assert out["paths"][0].endswith("real.py")
