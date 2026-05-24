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


def _run(action: str, query: str, directory: str | None = None) -> dict:
    params: dict = {"action": action, "query": query}
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


# ── blocked system paths ─────────────────────────────────────────────────────


def test_blocked_path_etc_rejected():
    out = _run("glob", "*", "/etc")
    assert out["status"] == "error"
    assert out["error"] == "system-path-blocked"


def test_blocked_path_proc_rejected():
    out = _run("grep", "x", "/proc")
    assert out["status"] == "error"
    assert out["error"] == "system-path-blocked"


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


def test_glob_skips_vcs_and_cache_dirs(tmp_path: Path):
    for skip in (".git", "node_modules", "__pycache__", ".venv"):
        d = tmp_path / skip
        d.mkdir()
        (d / "hidden.py").write_text("x")
    (tmp_path / "real.py").write_text("x")

    out = _run("glob", "*.py", str(tmp_path))
    names = sorted(os.path.basename(p) for p in out["paths"])
    assert names == ["real.py"]


def test_glob_returns_hint_on_empty_match(tmp_path: Path):
    out = _run("glob", "*.nope", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 0
    assert out["paths"] == []
    assert out["hint"] == _HINT


def test_glob_result_cap_truncates_and_flags(tmp_path: Path):
    for i in range(250):
        (tmp_path / f"f{i}.dat").write_text("x")
    out = _run("glob", "*.dat", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 200
    assert out["truncated"] is True
    assert len(out["paths"]) == 200


def test_glob_payload_contains_no_excerpts_or_line_numbers(tmp_path: Path):
    (tmp_path / "x.py").write_text("PolicyService\n" * 5)
    out = _run("glob", "*.py", str(tmp_path))
    # Forbidden fields — payload must stay path-list-only
    assert "matches" not in out
    assert "excerpts" not in out
    assert "line" not in out
    assert "text" not in out


# ── grep behaviour ───────────────────────────────────────────────────────────


def test_grep_finds_literal_match(tmp_path: Path):
    (tmp_path / "a.py").write_text("class PolicyService:\n    pass\n")
    (tmp_path / "b.py").write_text("nothing here\n")

    out = _run("grep", "PolicyService", str(tmp_path))
    assert out["status"] == "success"
    assert out["count"] == 1
    assert out["paths"][0].endswith("a.py")
    assert out["hint"] == _HINT


def test_grep_supports_regex(tmp_path: Path):
    (tmp_path / "a.py").write_text("foo123\n")
    (tmp_path / "b.py").write_text("bar\n")

    out = _run("grep", r"foo\d+", str(tmp_path))
    assert out["count"] == 1
    assert out["paths"][0].endswith("a.py")


def test_grep_invalid_regex_returns_error(tmp_path: Path):
    out = _run("grep", "[unterminated", str(tmp_path))
    assert out["status"] == "error"
    assert out["error"] == "invalid-regex"


def test_grep_skips_large_files(tmp_path: Path):
    big = tmp_path / "big.txt"
    # Just over 5 MB
    big.write_bytes(b"NEEDLE\n" + b"x" * (5 * 1024 * 1024 + 100))
    small = tmp_path / "small.txt"
    small.write_text("NEEDLE here\n")

    out = _run("grep", "NEEDLE", str(tmp_path))
    paths = [os.path.basename(p) for p in out["paths"]]
    assert "small.txt" in paths
    assert "big.txt" not in paths


def test_grep_payload_contains_no_excerpts_or_line_numbers(tmp_path: Path):
    (tmp_path / "a.py").write_text("PolicyService\n" * 10)
    (tmp_path / "b.py").write_text("PolicyService line one\nPolicyService line two\n")

    out = _run("grep", "PolicyService", str(tmp_path))
    assert out["status"] == "success"
    # Result must be paths-only — guards against re-introducing per-match excerpts
    assert "matches" not in out
    assert "excerpts" not in out
    assert "line" not in out
    for p in out["paths"]:
        assert isinstance(p, str)
        assert p.startswith("/")


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
    # self-referencing symlink — followlinks=False keeps the walk finite
    loop = inner / "loop"
    try:
        os.symlink(str(inner), str(loop))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this filesystem")

    out = _run("glob", "*.py", str(tmp_path))
    assert out["status"] == "success"
    # Should find the single real file, not recurse into the loop
    assert out["count"] == 1
    assert out["paths"][0].endswith("real.py")
