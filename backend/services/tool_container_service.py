"""
Tool Container Service — Docker image lifecycle and sandboxed tool execution.

Builds images from tool Dockerfiles at startup, then runs containers per invocation.
Input: base64-encoded JSON as container command arg.
Output: parsed JSON dict from container stdout.

Interactive protocol (run_interactive):
  stdout is JSON-protocol ONLY — one JSON object per line.
  stderr is for tool logging and is surfaced to backend logs.
  The framework writes {"text": response} + newline to stdin for each "tool" turn.
"""

import base64
import json
import logging
import subprocess
import threading

logger = logging.getLogger(__name__)

# Interactive protocol limits
_MAX_LINE_BYTES = 50 * 1024        # 50KB per stdout line
_MAX_TOTAL_BYTES = 200 * 1024      # 200KB cumulative stdout
_MAX_TURNS = 10                    # Max tool↔Chalie dialog turns


class ToolContainerService:
    def __init__(self):
        import docker
        self.client = docker.from_env()

    def build_image(self, tool_dir: str, image_tag: str, source_hash: str = None) -> bool:
        """
        Build Docker image from tool_dir/Dockerfile.

        Args:
            tool_dir: Absolute path to the tool directory
            image_tag: Docker image tag (e.g. "chalie-tool-telegram:1.0")
            source_hash: MD5 of tool source files, embedded as image label for staleness detection

        Returns:
            True on success, False on build failure.
        """
        import docker
        try:
            labels = {"chalie.source_hash": source_hash} if source_hash else {}
            self.client.images.build(path=tool_dir, tag=image_tag, rm=True, labels=labels)
            logger.info(f"[CONTAINER] Built image {image_tag}")
            return True
        except docker.errors.BuildError as e:
            logger.error(f"[CONTAINER] Build failed for {image_tag}: {e}")
            return False
        except Exception as e:
            logger.error(f"[CONTAINER] Unexpected build error for {image_tag}: {e}")
            return False

    def image_exists(self, image_tag: str) -> bool:
        """Check if image is already built and available locally."""
        import docker
        try:
            self.client.images.get(image_tag)
            return True
        except docker.errors.ImageNotFound:
            return False

    def get_image_source_hash(self, image_tag: str) -> str | None:
        """Return the chalie.source_hash label embedded in an existing image, or None."""
        try:
            img = self.client.images.get(image_tag)
            return img.labels.get("chalie.source_hash")
        except Exception:
            return None

    def run(self, image_tag: str, payload: dict, sandbox_config: dict, timeout: int = 9) -> dict:
        """
        Run tool container with payload as base64-encoded JSON command arg.

        Args:
            image_tag: Docker image to run
            payload: {"topic": str, "params": dict, "config": dict}
            sandbox_config: Sandbox constraints from manifest
            timeout: Max seconds to wait for container exit

        Returns:
            Parsed JSON dict from container stdout.

        Raises:
            RuntimeError: On non-zero exit or invalid JSON output.
            TimeoutError: If container exceeds timeout.
        """
        json_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        mem = sandbox_config.get("memory", "256m")
        network = sandbox_config.get("network", "bridge")

        run_kwargs = dict(
            command=json_b64,
            detach=True,
            remove=False,
            mem_limit=mem,
            network_mode=network,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            pids_limit=64,
            read_only=True,
            tmpfs={"/tmp": "size=64m", "/var/tmp": "size=64m"},
        )

        container = self.client.containers.run(image_tag, **run_kwargs)

        try:
            result = [None]
            err = [None]

            def _wait():
                try:
                    result[0] = container.wait()
                except Exception as e:
                    err[0] = e

            t = threading.Thread(target=_wait, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                container.kill()
                raise TimeoutError(f"Tool timed out after {timeout}s")

            if err[0]:
                raise RuntimeError(str(err[0]))

            if result[0]["StatusCode"] != 0:
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"Container exited {result[0]['StatusCode']}: {stderr}")

            # Cap output at 20KB
            raw = container.logs(stdout=True, stderr=False)[:20000]

            # Always surface container stderr to backend logs for debugging
            stderr_raw = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")[:800]
            if stderr_raw.strip():
                logger.info(f"[CONTAINER] {image_tag} stderr: {stderr_raw.strip()}")

            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"Tool returned invalid JSON: {e}")

        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def run_interactive(
        self,
        image_tag: str,
        payload: dict,
        sandbox_config: dict,
        timeout: int = 120,
        on_tool_output=None,
    ) -> dict:
        """
        Run a tool container with bidirectional stdin/stdout for tool↔Chalie dialog.

        stdout protocol: one JSON object per line (JSON-only, no logging).
        stderr: captured to backend logs as usual.

        Args:
            image_tag: Docker image to run.
            payload: Standard tool payload dict.
            sandbox_config: Sandbox constraints from manifest.
            timeout: Per-turn read timeout in seconds. Full dialog timeout = timeout * MAX_TURNS.
            on_tool_output: Callable(result: dict) -> str. Called when tool returns
                output=="tool". Should return Chalie's response text. If None, the
                loop exits immediately on "tool" output.

        Returns:
            Final result dict (output != "tool", or last result if turns exhausted).

        Raises:
            RuntimeError: On container startup failure or invalid JSON.
            TimeoutError: If a turn exceeds timeout.
        """
        json_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        mem = sandbox_config.get("memory", "256m")
        network = sandbox_config.get("network", "bridge")

        cmd = [
            "docker", "run", "--rm", "-i",
            f"--memory={mem}",
            f"--network={network}",
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "--pids-limit=64",
            "--read-only",
            "--tmpfs=/tmp:size=64m",
            "--tmpfs=/var/tmp:size=64m",
            image_tag,
            json_b64,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start interactive container: {e}")

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
                    raise TimeoutError(f"Interactive tool timed out waiting for output (turn {turns + 1})")

                if read_error[0]:
                    raise RuntimeError(str(read_error[0]))

                raw_line = line_data[0]
                if not raw_line:
                    # stdout closed — container exited
                    break

                # Enforce per-line size
                if len(raw_line) > _MAX_LINE_BYTES:
                    logger.warning(f"[CONTAINER INTERACTIVE] {image_tag}: stdout line exceeds {_MAX_LINE_BYTES}B, truncating")
                    raw_line = raw_line[:_MAX_LINE_BYTES]

                total_bytes += len(raw_line)
                if total_bytes > _MAX_TOTAL_BYTES:
                    logger.warning(f"[CONTAINER INTERACTIVE] {image_tag}: cumulative stdout exceeds {_MAX_TOTAL_BYTES}B, stopping")
                    break

                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                # Parse JSON — non-JSON lines are logged and skipped
                try:
                    result = json.loads(line_str)
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"[CONTAINER INTERACTIVE] {image_tag}: non-JSON stdout (turn {turns + 1}): {line_str[:200]}")
                    continue

                turns += 1

                if result.get("output") != "tool" or on_tool_output is None:
                    # Terminal output — done
                    break

                # Call Chalie, write response back to stdin
                try:
                    chalie_response = on_tool_output(result)
                    response_json = json.dumps({"text": chalie_response}) + "\n"
                    proc.stdin.write(response_json.encode())
                    proc.stdin.flush()
                except Exception as e:
                    logger.error(f"[CONTAINER INTERACTIVE] {image_tag}: on_tool_output error: {e}")
                    break

            if turns >= _MAX_TURNS:
                logger.warning(f"[CONTAINER INTERACTIVE] {image_tag}: max turns ({_MAX_TURNS}) reached")

        finally:
            # Drain stderr for logging
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
            logger.info(f"[CONTAINER INTERACTIVE] {image_tag} stderr: {' | '.join(stderr_lines)}")

        return result

