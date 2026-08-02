"""Feature tests: the ``read`` ability's line-window contract.

``max_chars`` truncation is gone. Oversized content is never silently clipped:
a whole-file read over the 20 000-character cap is a LOUD ``too-large`` error
telling the model to select a window, and a window is requested with 1-indexed
``start_line`` / ``end_line`` params (either alone is valid — the missing bound
defaults to the file edge). A window smaller than the file carries
``partial=True`` meta, which the read-guard uses to refuse stale full-file
writes.
"""

from pathlib import Path

import pytest

from abilities._result import ToolResult
from abilities.read import ReadAbility
from contracts.params.read_params_bag import ReadParamsBag
from tests._tool_result_harness import built

pytestmark = pytest.mark.unit

_LINES = [f"line {i}\n" for i in range(1, 6)]


def _run(source: str, **extra: object) -> ToolResult:
    params: dict[str, object] = {"source": source, **extra}
    # Build the bag exactly as the dispatch seam does — from_params returns the
    # error ToolResult directly, so the error-path tests still see the envelope.
    bag = ReadParamsBag.from_params(params)
    if isinstance(bag, ToolResult):
        return bag
    return ReadAbility().run(bag)


@pytest.fixture()
def small_file(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text("".join(_LINES), encoding="utf-8")
    return p


@pytest.fixture()
def big_file(tmp_path: Path) -> Path:
    """2000 lines × 11 chars = 22 000 chars — over the 20k cap."""
    p = tmp_path / "big.log"
    p.write_text("0123456789\n" * 2000, encoding="utf-8")
    return p


class TestParamValidation:

    def test_zero_start_line_rejected(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=0)
        assert result.status == "error"
        assert result.code == "invalid-param"

    def test_string_line_number_rejected_not_coerced(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line="2")
        assert result.status == "error"
        assert result.code == "invalid-param"

    def test_boolean_rejected(self, small_file: Path) -> None:
        result = _run(str(small_file), end_line=True)
        assert result.status == "error"
        assert result.code == "invalid-param"

    def test_end_before_start_rejected(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=4, end_line=2)
        assert result.status == "error"
        assert result.code == "invalid-param"
        assert "'end_line' must be greater than or equal to 'start_line'." in str(result.body)

    def test_either_bound_alone_is_a_valid_bag(self) -> None:
        start_only = built(ReadParamsBag.from_params({"source": "x", "start_line": 2}))
        assert (start_only.start_line, start_only.end_line) == (2, None)
        end_only = built(ReadParamsBag.from_params({"source": "x", "end_line": 3}))
        assert (end_only.start_line, end_only.end_line) == (None, 3)


class TestWholeFileGate:

    def test_small_file_returned_whole(self, small_file: Path) -> None:
        result = _run(str(small_file))
        assert result.status == "success"
        assert result.body == "".join(_LINES)
        assert "partial" not in result.meta

    def test_oversized_file_is_a_loud_error_not_a_clip(self, big_file: Path) -> None:
        result = _run(str(big_file))
        assert result.status == "error"
        assert result.code == "too-large"
        assert result.body == (
            "File is too large to load into context, select chunks of the "
            "document by supplying `start_line`/`end_line`. The file has "
            "2000 lines."
        )
        assert result.meta["total_lines"] == 2000


class TestLineWindows:

    def test_window_returns_exact_lines_with_endings(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=2, end_line=3)
        assert result.status == "success"
        assert result.body == "line 2\nline 3\n"
        assert result.meta["lines"] == "2-3"
        assert result.meta["total_lines"] == 5
        assert result.meta["partial"] is True

    def test_start_only_reads_to_end_of_file(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=4)
        assert result.status == "success"
        assert result.body == "line 4\nline 5\n"
        assert result.meta["lines"] == "4-5"
        assert result.meta["partial"] is True

    def test_end_only_reads_from_line_one(self, small_file: Path) -> None:
        result = _run(str(small_file), end_line=2)
        assert result.status == "success"
        assert result.body == "line 1\nline 2\n"
        assert result.meta["lines"] == "1-2"
        assert result.meta["partial"] is True

    def test_explicit_full_range_is_not_partial(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=1, end_line=5)
        assert result.status == "success"
        assert result.body == "".join(_LINES)
        assert "partial" not in result.meta

    def test_end_past_eof_clamps_to_last_line(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=1, end_line=999)
        assert result.status == "success"
        assert result.body == "".join(_LINES)
        assert result.meta["lines"] == "1-5"
        assert "partial" not in result.meta

    def test_start_past_eof_is_out_of_range(self, small_file: Path) -> None:
        result = _run(str(small_file), start_line=7)
        assert result.status == "error"
        assert result.code == "line-out-of-range"
        assert result.body == "Line number 7 exceeds file length (5 lines)."

    def test_oversized_window_is_a_loud_error(self, tmp_path: Path) -> None:
        p = tmp_path / "one_giant_line.txt"
        p.write_text("x" * 25_000, encoding="utf-8")
        result = _run(str(p), start_line=1, end_line=1)
        assert result.status == "error"
        assert result.code == "too-large"
        assert "select a narrower range" in str(result.body)

    def test_window_into_oversized_file_works(self, big_file: Path) -> None:
        """The whole point of the redesign: a file too big to read whole is
        still readable in slices."""
        result = _run(str(big_file), start_line=100, end_line=110)
        assert result.status == "success"
        assert result.body == "0123456789\n" * 11
        assert result.meta["partial"] is True
