"""
Tool Container Service — Docker image lifecycle and sandboxed tool execution.

Builds images from tool Dockerfiles at startup, then runs containers per invocation.
Input: base64-encoded JSON as container command arg.
Output: parsed JSON dict from container stdout.
"""

import base64
import json
import logging
import threading

logger = logging.getLogger(__name__)


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
        writable = sandbox_config.get("writable", False)

        run_kwargs = dict(
            command=json_b64,
            detach=True,
            remove=False,
            mem_limit=mem,
            network_mode=network,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            pids_limit=64,
        )
        if not writable:
            run_kwargs["read_only"] = True

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

