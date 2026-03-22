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

from services.tool_subprocess_service import ToolSubprocessService


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

    def test_custom_timeout(self, svc):
        """Caller-specified timeout is forwarded to subprocess.run."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b'{"ok": true}'
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result) as mock_run:
            svc.run("/fake/runner.py", {"params": {}}, timeout=30)

        assert mock_run.call_args.kwargs["timeout"] == 30

    def test_payload_encoded_as_base64_json(self, svc):
        """Payload is serialised as base64-encoded JSON and passed as CLI arg."""
        payload = {"params": {"query": "test"}, "settings": {}}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b'{"text": "ok"}'
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result) as mock_run:
            svc.run("/fake/runner.py", payload)

        cmd = mock_run.call_args[0][0]
        # cmd[2] should be the base64-encoded JSON
        decoded = json.loads(base64.b64decode(cmd[2]))
        assert decoded == payload

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

    def test_empty_json_output(self, svc):
        """Empty JSON object is a valid response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"{}"
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result):
            result = svc.run("/fake/runner.py", {"params": {}})
            assert result == {}

    def test_subprocess_launch_failure(self, svc):
        """FileNotFoundError (missing script) → RuntimeError."""
        with patch(
            "services.tool_subprocess_service.subprocess.run",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to run trusted tool"):
                svc.run("/nonexistent/runner.py", {"params": {}})

    def test_stderr_logged_on_success(self, svc):
        """Stderr output is logged even when the tool exits 0."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b'{"ok": true}'
        mock_result.stderr = b"[WARN] some debug info"

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result):
            with patch("services.tool_subprocess_service.logger.info") as mock_log:
                svc.run("/fake/runner.py", {"params": {}})
                mock_log.assert_called_once()
                assert "stderr" in mock_log.call_args[0][0]

    def test_cwd_set_to_runner_parent(self, svc):
        """Working directory is set to the runner script's parent directory."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b'{"ok": true}'
        mock_result.stderr = b""

        with patch("services.tool_subprocess_service.subprocess.run", return_value=mock_result) as mock_run:
            svc.run("/tools/my_tool/runner.py", {"params": {}})

        assert mock_run.call_args.kwargs["cwd"] == "/tools/my_tool"


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

    def test_stdout_closed_returns_last_result(self, svc):
        """Empty readline (stdout closed) → loop exits, returns last parsed result."""
        tool_line = json.dumps({"text": "partial"}) + "\n"

        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = [
            tool_line.encode(),
            b"",  # stdout closed
        ]
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            result = svc.run_interactive("/fake/runner.py", {"params": {}})

        # First line was valid JSON with output != 'tool', so loop breaks on it
        assert result["text"] == "partial"

    def test_popen_failure_raises_runtime_error(self, svc):
        """Popen raises → RuntimeError."""
        with patch(
            "services.tool_subprocess_service.subprocess.Popen",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to start interactive"):
                svc.run_interactive("/nonexistent/runner.py", {"params": {}})

    def test_non_json_lines_skipped(self, svc):
        """Non-JSON stdout lines are skipped, valid JSON is returned."""
        lines = [
            b"this is not json\n",
            (json.dumps({"text": "valid"}) + "\n").encode(),
        ]

        mock_proc = MagicMock()
        mock_proc.stdout.readline.side_effect = lines
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            result = svc.run_interactive("/fake/runner.py", {"params": {}})

        assert result["text"] == "valid"

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
            result = svc.run_interactive(
                "/fake/runner.py", {"params": {}}, on_tool_output=callback
            )

        # Callback should have been invoked exactly _MAX_TURNS (10) times
        assert callback.call_count == 10

    def test_empty_result_when_no_output(self, svc):
        """If stdout closes immediately, result is empty dict."""
        mock_proc = MagicMock()
        mock_proc.stdout.readline.return_value = b""
        mock_proc.stderr.read.return_value = b""
        mock_proc.wait.return_value = 0

        with patch("services.tool_subprocess_service.subprocess.Popen", return_value=mock_proc):
            result = svc.run_interactive("/fake/runner.py", {"params": {}})

        assert result == {}
