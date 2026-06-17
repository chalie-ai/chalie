
import base64
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Interactive protocol limits (same as ToolSubprocessService)
_MAX_LINE_BYTES = 50 * 1024        # 50KB per stdout line
_MAX_TOTAL_BYTES = 200 * 1024      # 200KB cumulative stdout
_MAX_TURNS = 10                    # Max tool↔Chalie dialog turns


def _read_line_with_timeout(proc, timeout: int, turn: int) -> bytes | None:
    line_data = [None]
    read_error = [None]

    def _read_line():
        try:
            line_data[0] = proc.stdout.readline()
        except Exception as e:
            read_error[0] = e

    t = threading.Thread(target=_read_line, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        proc.kill()
        raise TimeoutError(
            f"Interactive tool timed out waiting for output (turn {turn + 1})"
        )

    if read_error[0]:
        raise RuntimeError(str(read_error[0]))

    raw_line = line_data[0]
    if not raw_line:
        return None
    return raw_line


def _apply_size_limits(raw_line: bytes, total_bytes: int) -> tuple:
    if len(raw_line) > _MAX_LINE_BYTES:
        logger.warning(
            f"[SUBPROCESS INTERACTIVE] stdout line exceeds {_MAX_LINE_BYTES}B, truncating"
        )
        raw_line = raw_line[:_MAX_LINE_BYTES]

    total_bytes += len(raw_line)
    if total_bytes > _MAX_TOTAL_BYTES:
        logger.warning(
            f"[SUBPROCESS INTERACTIVE] cumulative stdout exceeds {_MAX_TOTAL_BYTES}B, stopping"
        )
        return raw_line, total_bytes, True

    return raw_line, total_bytes, False


def _parse_json_line(raw_line: bytes, turn: int) -> tuple:
    """Decode and parse one stdout line as JSON.

    Returns (parsed_dict, skip_flag).  skip_flag is True when the line
    should be skipped (empty or non-JSON).
    """
    line_str = raw_line.decode("utf-8", errors="replace").strip()
    if not line_str:
        return {}, True

    try:
        return json.loads(line_str), False
    except ValueError:
        logger.warning(
            f"[SUBPROCESS INTERACTIVE] non-JSON stdout (turn {turn + 1}): "
            f"{line_str[:200]}"
        )
        return {}, True


def _send_chalie_response(proc, result: dict, on_tool_output) -> bool:
    """Call *on_tool_output*, then write the response JSON to proc.stdin.

    Returns True on success, False if an error occurred (caller should break).
    """
    try:
        chalie_response = on_tool_output(result)
        response_json = json.dumps({"text": chalie_response}) + "\n"
        proc.stdin.write(response_json.encode())
        proc.stdin.flush()
        return True
    except Exception as e:
        logger.error(f"[SUBPROCESS INTERACTIVE] on_tool_output error: {e}")
        return False


def _cleanup_proc(proc, stderr_lines: list) -> list:
    """Close stdin, drain stderr, and wait for the subprocess to exit."""
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        stderr_raw = proc.stderr.read(800)
        if stderr_raw:
            stderr_lines.append(stderr_raw.decode("utf-8", errors="replace").strip())
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return stderr_lines


class ToolSubprocessService:
    """Execute trusted tools via subprocess (no Docker required)."""

    def run(self, runner_path: str, payload: dict, timeout: int = 9) -> dict:
        """
        Run a trusted tool via subprocess.

        Args:
            runner_path: Absolute path to the tool's runner.py
            payload: {"params": dict, "settings": dict, "telemetry": dict}
            timeout: Max seconds to wait for process exit

        Returns:
            Parsed JSON dict from stdout.

        Raises:
            RuntimeError: On non-zero exit or invalid JSON output.
            TimeoutError: If process exceeds timeout.
        """
        json_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

        try:
            result = subprocess.run(
                [sys.executable, runner_path, json_b64],
                capture_output=True,
                timeout=timeout,
                cwd=str(Path(runner_path).parent),
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Trusted tool timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"Failed to run trusted tool: {e}")

        # Surface stderr to backend logs
        if result.stderr:
            stderr_text = result.stderr.decode("utf-8", errors="replace")[:800].strip()
            if stderr_text:
                logger.info(f"[SUBPROCESS] {runner_path} stderr: {stderr_text}")

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Tool exited {result.returncode}: {stderr_text}")

        raw = result.stdout

        try:
            return json.loads(raw)
        except ValueError as e:
            raise RuntimeError(f"Tool returned invalid JSON: {e}")

    def run_interactive(
        self,
        runner_path: str,
        payload: dict,
        timeout: int = 120,
        on_tool_output=None,
    ) -> dict:
        """
        Run a trusted tool with bidirectional stdin/stdout for tool↔Chalie dialog.

        Same protocol as ToolSubprocessService.run_interactive().

        Args:
            runner_path: Absolute path to the tool's runner.py
            payload: Standard tool payload dict.
            timeout: Per-turn read timeout in seconds.
            on_tool_output: Callable(result: dict) -> str. Called when tool returns
                output=="tool". Should return Chalie's response text.

        Returns:
            Final result dict.
        """
        json_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

        try:
            proc = subprocess.Popen(
                [sys.executable, runner_path, json_b64],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(runner_path).parent),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start interactive subprocess: {e}")

        result = {}
        total_bytes = 0
        turns = 0
        stderr_lines = []

        try:
            while turns < _MAX_TURNS:
                raw_line = _read_line_with_timeout(proc, timeout, turns)

                if not raw_line:
                    break

                raw_line, total_bytes, should_stop = _apply_size_limits(raw_line, total_bytes)
                if should_stop:
                    break

                result, skip = _parse_json_line(raw_line, turns)
                if skip:
                    continue

                turns += 1

                if result.get("output") != "tool" or on_tool_output is None:
                    break

                if not _send_chalie_response(proc, result, on_tool_output):
                    break

            if turns >= _MAX_TURNS:
                logger.warning(f"[SUBPROCESS INTERACTIVE] max turns ({_MAX_TURNS}) reached")

        finally:
            stderr_lines = _cleanup_proc(proc, stderr_lines)

        if stderr_lines:
            logger.info(f"[SUBPROCESS INTERACTIVE] stderr: {' | '.join(stderr_lines)}")

        return result
