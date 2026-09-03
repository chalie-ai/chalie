"""Feature tests: the ``read`` ability's line-window contract.

No silent clipping: a whole-file read over the 20 000-character cap is a loud
``too-large`` error telling the model to select a window, and a
``start_line``/``end_line`` window smaller than the file carries
``partial=True`` meta — the field the read-guard uses to refuse stale
full-file writes.
"""

from pathlib import Path

import pytest

from abilities._result import ToolResult
from abilities.read import ReadAbility
from contracts.params.read_params_bag import ReadParamsBag

pytestmark = pytest.mark.unit

_LINES = [f"line {i}\n" for i in range(1, 6)]


def _run(source: str, **extra: object) -> ToolResult:
    bag = ReadParamsBag.from_params({"source": source, **extra})
    return bag if isinstance(bag, ToolResult) else ReadAbility().run(bag)


@pytest.fixture()
def small_file(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text("".join(_LINES), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "params",
    [
        {"start_line": 0},
        {"start_line": "2"},  # strings are rejected, never coerced
        {"end_line": True},
        {"start_line": 4, "end_line": 2},
    ],
)
def test_invalid_windows_rejected(small_file: Path, params: dict[str, object]) -> None:
    result = _run(str(small_file), **params)
    assert (result.status, result.code) == ("error", "invalid-param")


@pytest.mark.parametrize(
    ("window", "body", "partial"),
    [
        ({}, "".join(_LINES), False),
        ({"start_line": 2, "end_line": 3}, "line 2\nline 3\n", True),
        ({"start_line": 4}, "line 4\nline 5\n", True),  # missing bound = file edge
        ({"end_line": 2}, "line 1\nline 2\n", True),
        ({"start_line": 1, "end_line": 5}, "".join(_LINES), False),
        ({"start_line": 1, "end_line": 999}, "".join(_LINES), False),  # clamps to EOF
    ],
)
def test_windows_return_exact_lines(
    small_file: Path, window: dict[str, object], body: str, partial: bool
) -> None:
    result = _run(str(small_file), **window)
    assert result.status == "success"
    assert result.body == body
    assert result.meta.get("partial", False) is partial


def test_start_past_eof_is_out_of_range(small_file: Path) -> None:
    result = _run(str(small_file), start_line=7)
    assert (result.status, result.code) == ("error", "line-out-of-range")
    assert result.body == "Line number 7 exceeds file length (5 lines)."


def test_oversized_file_is_loud_error_and_still_readable_in_slices(tmp_path: Path) -> None:
    """2000 lines × 11 chars = 22 000 — over the cap whole, readable windowed."""
    big = tmp_path / "big.log"
    big.write_text("0123456789\n" * 2000, encoding="utf-8")
    whole = _run(str(big))
    assert (whole.status, whole.code) == ("error", "too-large")
    assert "start_line" in str(whole.body)
    assert whole.meta["total_lines"] == 2000
    window = _run(str(big), start_line=100, end_line=110)
    assert window.status == "success"
    assert window.body == "0123456789\n" * 11
    assert window.meta["partial"] is True


def test_oversized_window_is_loud_error(tmp_path: Path) -> None:
    p = tmp_path / "one_giant_line.txt"
    p.write_text("x" * 25_000, encoding="utf-8")
    result = _run(str(p), start_line=1, end_line=1)
    assert (result.status, result.code) == ("error", "too-large")
    assert "select a narrower range" in str(result.body)
