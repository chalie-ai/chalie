"""Tests for abilities/search_files.py (SearchFilesAbility).

Unit tests over deterministic file-system behavior. Each test creates an
isolated tmp tree and exercises the real ability — no mocks.
"""

import json
import os
from pathlib import Path

import pytest

from abilities.search_files import SearchFilesAbility, _HINT

pytestmark = pytest.mark.unit


def _run(action: str, query: str, directory: str | None = None, **extra) -> dict:
    params: dict = {"action": action, "query": query, **extra}
    if directory is not None:
        params["directory"] = directory
    raw = SearchFilesAbility().execute("user", params, None)
    return json.loads(raw["text"])


# ── action / query validation ────────────────────────────────────────────────


def test_invalid_action_returns_error(tmp_path: Path):
    out = _run("scan", "*.py", str(tmp_path))
    assert out["status"] == "error"
    assert out["error"] == "invalid-action"


def test_missing_query_returns_error(tmp_path: Path):
    out = _run("glob", "", str(tmp_path))
    assert out["status"] == "error"
    assert out["error"] == "query-required"


# ── no blocked paths — LLM can search anywhere ──────────────────────────────


def test_etc_is_searchable():
    out = _run("glob", "*.conf", "/etc", max_files=2)
    assert out["status"] == "success"


def test_missing_directory_returns_error(tmp_path: Path):
    out = _run("glob", "*", str(tmp_path / "does_not_exist"))
    assert out["status"] == "error"
    assert out["error"] == "directory-not-found"


def test_file_as_directory_rejected(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    out = _run("glob", "*", str(f))
    assert out["status"] == "error"
    assert out["error"] == "not-a-directory"


# ── glob behaviour ───────────────────────────────────────────────────────────


def test_glob_matches_basename(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("# a")
    (tmp_path / "beta.txt").write_text("# b")
    (tmp_path / "gamma.py").write_text("# c")

    out = _run("glob", "*.py", str(tmp_path))
    assert out["status"] == "success"
    assert out["action"] == "glob"
    assert out["count"] == 2
    assert out["truncated"] is False
    assert out["hint"] == _HINT
    names = sorted(os.path.basename(p) for p in out["paths"])
    assert names == ["alpha.py", "gamma.py"]


def test_glob_recursive_with_double_star(tmp_path: Path):
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "found.log").write_text("x")
    (tmp_path / "top.log").write_text("x")

    out = _run("glob", "**/*.log", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 2
    names = sorted(os.path.basename(p) for p in out["paths"])
    assert names == ["found.log", "top.log"]


def test_glob_returns_hint_on_empty_match(tmp_path: Path):
    out = _run("glob", "*.nope", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 0
    assert out["paths"] == []
    assert out["hint"] == _HINT


def test_glob_max_files_truncates_and_flags(tmp_path: Path):
    for i in range(20):
        (tmp_path / f"f{i}.dat").write_text("x")
    out = _run("glob", "*.dat", str(tmp_path), max_files=5)
    assert out["status"] == "success"
    assert out["count"] == 5
    assert out["truncated"] is True
    assert len(out["paths"]) == 5


def test_glob_default_max_files_is_10(tmp_path: Path):
    for i in range(15):
        (tmp_path / f"f{i}.dat").write_text("x")
    out = _run("glob", "*.dat", str(tmp_path))
    assert out["count"] == 10
    assert out["truncated"] is True


# ── grep behaviour ───────────────────────────────────────────────────────────


def test_grep_finds_literal_match_with_context(tmp_path: Path):
    content = "\n".join(f"line {i}" for i in range(1, 11))
    (tmp_path / "a.py").write_text(content + "\n")
    (tmp_path / "a.py").write_text(
        "line 1\nline 2\nline 3\nline 4\nTARGET\nline 6\nline 7\nline 8\nline 9\nline 10\n"
    )
    (tmp_path / "b.py").write_text("nothing here\n")

    out = _run("grep", "TARGET", str(tmp_path), context_lines=3)
    assert out["status"] == "success"
    assert out["count"] == 1
    assert out["hint"] == _HINT

    result = out["results"][0]
    assert result["file"].endswith("a.py")
    snippet = result["matches"][0]
    assert "ln 5: TARGET" in snippet
    assert "ln 2: line 2" in snippet
    assert "ln 8: line 8" in snippet


def test_grep_context_lines_default_is_3(tmp_path: Path):
    lines = [f"L{i}" for i in range(1, 12)]
    lines[5] = "NEEDLE"
    (tmp_path / "f.txt").write_text("\n".join(lines) + "\n")

    out = _run("grep", "NEEDLE", str(tmp_path))
    snippet = out["results"][0]["matches"][0]
    assert "ln 3: L3" in snippet  # line 6 - 3 = line 3
    assert "ln 9: L9" in snippet  # line 6 + 3 = line 9
    assert "ln 6: NEEDLE" in snippet


def test_grep_context_lines_clamps_at_file_boundaries(tmp_path: Path):
    (tmp_path / "f.txt").write_text("FIRST\nsecond\n")

    out = _run("grep", "FIRST", str(tmp_path), context_lines=5)
    snippet = out["results"][0]["matches"][0]
    assert "ln 1: FIRST" in snippet
    assert "ln 2: second" in snippet
    lines = snippet.strip().split("\n")
    assert len(lines) == 2


def test_grep_supports_regex(tmp_path: Path):
    (tmp_path / "a.py").write_text("foo123\n")
    (tmp_path / "b.py").write_text("bar\n")

    out = _run("grep", r"foo\d+", str(tmp_path))
    assert out["count"] == 1
    assert out["results"][0]["file"].endswith("a.py")


def test_grep_invalid_regex_returns_error(tmp_path: Path):
    out = _run("grep", "[unterminated", str(tmp_path))
    assert out["status"] == "error"
    assert out["error"] == "invalid-regex"


def test_grep_skips_large_files(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"NEEDLE\n" + b"x" * (5 * 1024 * 1024 + 100))
    small = tmp_path / "small.txt"
    small.write_text("NEEDLE here\n")

    out = _run("grep", "NEEDLE", str(tmp_path))
    files = [r["file"] for r in out["results"]]
    basenames = [os.path.basename(f) for f in files]
    assert "small.txt" in basenames
    assert "big.txt" not in basenames


def test_grep_max_files_caps_results(tmp_path: Path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text(f"MATCH {i}\n")
    out = _run("grep", "MATCH", str(tmp_path), max_files=3)
    assert out["count"] == 3
    assert out["truncated"] is True


def test_grep_multiple_matches_in_one_file(tmp_path: Path):
    (tmp_path / "f.py").write_text("AAA\nbbb\nccc\nddd\neee\nfff\nggg\nAAA\nhhh\niii\n")
    out = _run("grep", "AAA", str(tmp_path), context_lines=1)
    result = out["results"][0]
    assert len(result["matches"]) == 2
    assert "ln 1: AAA" in result["matches"][0]
    assert "ln 8: AAA" in result["matches"][1]


def test_grep_adjacent_matches_merge_context(tmp_path: Path):
    (tmp_path / "f.py").write_text("a\nMATCH1\nc\nMATCH2\ne\nf\n")
    out = _run("grep", "MATCH", str(tmp_path), context_lines=1)
    result = out["results"][0]
    assert len(result["matches"]) == 1
    snippet = result["matches"][0]
    assert "ln 2: MATCH1" in snippet
    assert "ln 4: MATCH2" in snippet
    assert snippet.count("ln 3: c") == 1


def test_negative_max_files_clamps_to_1(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("x")
    out = _run("glob", "*.py", str(tmp_path), max_files=-5)
    assert out["status"] == "success"
    assert out["count"] == 1


def test_negative_context_lines_clamps_to_0(tmp_path: Path):
    (tmp_path / "f.txt").write_text("aaa\nTARGET\nzzz\n")
    out = _run("grep", "TARGET", str(tmp_path), context_lines=-3)
    snippet = out["results"][0]["matches"][0]
    assert snippet == "ln 2: TARGET"


# ── default directory ───────────────────────────────────────────────────────


def test_omitted_directory_defaults_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "marker.tmp").write_text("x")

    out = _run("glob", "marker.tmp")
    assert out["status"] == "success"
    assert out["directory"] == str(tmp_path.resolve())
    assert out["count"] == 1


# ── symlink loop guard ──────────────────────────────────────────────────────


def test_symlink_loop_does_not_hang(tmp_path: Path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "real.py").write_text("x")
    loop = inner / "loop"
    try:
        os.symlink(str(inner), str(loop))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    out = _run("glob", "*.py", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 1
    assert out["paths"][0].endswith("real.py")
