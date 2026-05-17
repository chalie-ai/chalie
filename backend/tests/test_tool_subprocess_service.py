"""
Unit tests for ToolSubprocessService — subprocess-based tool execution.

Tests run(), run_interactive() with mocked subprocess calls.
"""

import json
import base64
import subprocess
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.unit

from services.tool_subprocess_service import ToolSubprocessService  # noqa: E402


@pytest.fixture
def svc():
    return ToolSubprocessService()


# ── run() tests ──────────────────────────────────────────────────────────────


class TestRun:

    def test_successful_execution(self, svc):
        """Valid JSON on stdout → parsed dict returned."""
        expected = {"text": "hello", "title": "Test"}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(expected).encode()
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result) as mock_run:
            result = svc.run("/fake/runner.py", {"params": {}})

        assert result == expected
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs["timeout"] == 9
        assert call_args.kwargs["capture_output"] is True

    def test_timeout_raises_timeout_error(self, svc):
        """subprocess.TimeoutExpired → TimeoutError."""
        with patch(
            "services.tool_subprocess_service.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=9),
        ):
            with pytest.raises(TimeoutError, match="timed out after 9s"):
                svc.run("/fake/runner.py", {"params": {}})

    def test_nonzero_exit_code_raises_runtime_error(self, svc):
        """Non-zero return code → RuntimeError with stderr excerpt."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"segfault in frobnicator"
        mock_result.stdout = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Tool exited 1.*segfault"):
                svc.run("/fake/runner.py", {"params": {}})

    def test_invalid_json_output_raises_runtime_error(self, svc):
        """Stdout that is not valid JSON → RuntimeError."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"not json at all"
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                svc.run("/fake/runner.py", {"params": {}})

    def test_subprocess_launch_failure(self, svc):
        """FileNotFoundError (missing script) → RuntimeError."""
        with patch(
            "services.tool_subprocess_service.subprocess.run",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to run trusted tool"):
                svc.run("/nonexistent/runner.py", {"params": {}})


# ── run_interactive() tests ──────────────────────────────────────────────────


class TestRunInteractive:

    def test_single_final_result(self, svc):
        """Tool emits one JSON line with output != 'tool' → returned as result."""
        expected = {"text": "Done", "output": "final"}

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = (json.dumps(expected) + "\n").encode()
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            result = svc.run_interactive("/fake/runner.py", {"params": {}})

        assert result == expected

    def test_multi_turn_dialog(self, svc):
        """Tool emits output='tool' → callback invoked → response written to stdin."""
        tool_line_1 = json.dumps({"output": "tool", "question": "Which colour?"}) + "\n"
        tool_line_2 = json.dumps({"text": "Blue noted", "output": "final"}) + "\n"

        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            tool_line_1.encode(),
            tool_line_2.encode(),
        ]
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_proc.stdin = MagicMock()

        callback = MagicMock(return_value="blue")

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            result = svc.run_interactive("/fake/runner.py", {"params": {}}, on_tool_output=callback)

        assert result["text"] == "Blue noted"
        callback.assert_called_once()

        # Verify Chalie's response was written to stdin as JSON
        written = mock_proc.stdin.write.call_args[0][0]
        parsed = json.loads(written.decode())
        assert parsed == {"text": "blue"}

    def test_popen_failure_raises_runtime_error(self, svc):
        """Popen raises → RuntimeError."""
        with patch(
            "services.tool_subprocess_service.subprocess.Popen",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to start interactive"):
                svc.run_interactive("/nonexistent/runner.py", {"params": {}})

    def test_max_turns_enforced(self, svc):
        """Loop stops after _MAX_TURNS and returns last result."""
        # All 10 lines are output='tool' to force turn counting
        tool_line = json.dumps({"output": "tool", "q": "again?"}) + "\n"

        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = tool_line.encode()
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0
        mock_proc.stdin = MagicMock()

        callback = MagicMock(return_value="ok")

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            svc.run_interactive(
                "/fake/runner.py", {"params": {}}, on_tool_output=callback
            )

        # Callback should have been invoked exactly _MAX_TURNS (10) times
        assert callback.call_count == 10

