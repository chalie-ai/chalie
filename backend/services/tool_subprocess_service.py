"""
Tool Subprocess Service — Trusted tool execution via Python subprocess.

Mirrors ToolContainerService's API but runs tools directly as subprocesses
instead of Docker containers. Used for tools with "trust": "trusted" in their
manifest. Trusted tools run as the same OS user — no sandboxing.

IPC contract (same as ToolContainerService):
  Input:  base64-encoded JSON as command arg: {"params", "settings", "telemetry"}
  Output: JSON on stdout: {"text"?, "html"?, "title"?, "error"?}
  Error:  non-zero exit code + error text on stderr

Interactive protocol (run_interactive):
  stdout is JSON-protocol ONLY — one JSON object per line.
  stderr is for tool logging and is surfaced to backend logs.
  The framework writes {"text": response} + newline to stdin for each "tool" turn.
"""

import base64
import json
import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

# Interactive protocol limits (same as ToolContainerService)
_MAX_LINE_BYTES = 50 * 1024        # 50KB per stdout line
_MAX_TOTAL_BYTES = 200 * 1024      # 200KB cumulative stdout
_MAX_TURNS = 10                    # Max tool↔Chalie dialog turns


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

        # Cap output at 20KB (same as ToolContainerService)
        raw = result.stdout[:20000]

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
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

        Same protocol as ToolContainerService.run_interactive().

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
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start interactive subprocess: {e}")

        result = {}
        total_bytes = 0
        turns = 0
        stderr_lines = []

        try:
            while turns < _MAX_TURNS:
                # Read one JSON line from stdout with timeout
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
                        f"Interactive tool timed out waiting for output (turn {turns + 1})"
                    )

                if read_error[0]:
                    raise RuntimeError(str(read_error[0]))

                raw_line = line_data[0]
                if not raw_line:
                    # stdout closed — process exited
                    break

                # Enforce per-line size
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
                    break

                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    result = json.loads(line_str)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        f"[SUBPROCESS INTERACTIVE] non-JSON stdout (turn {turns + 1}): "
                        f"{line_str[:200]}"
                    )
                    continue

                turns += 1

                if result.get("output") != "tool" or on_tool_output is None:
                    break

                # Call Chalie, write response back to stdin
                try:
                    chalie_response = on_tool_output(result)
                    response_json = json.dumps({"text": chalie_response}) + "\n"
                    proc.stdin.write(response_json.encode())
                    proc.stdin.flush()
                except Exception as e:
                    logger.error(f"[SUBPROCESS INTERACTIVE] on_tool_output error: {e}")
                    break

            if turns >= _MAX_TURNS:
                logger.warning(f"[SUBPROCESS INTERACTIVE] max turns ({_MAX_TURNS}) reached")

        finally:
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

        if stderr_lines:
            logger.info(f"[SUBPROCESS INTERACTIVE] stderr: {' | '.join(stderr_lines)}")

        return result
