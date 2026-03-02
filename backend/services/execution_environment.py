"""
Execution Environment — Unified interface for running sandboxed containers.

Wraps Docker SDK for tool execution, voice service, and any future containers.
Gracefully degrades if Docker is not installed or not running.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Singleton
_instance = None


def get_execution_environment():
    """Get or create the shared ExecutionEnvironment singleton."""
    global _instance
    if _instance is None:
        _instance = ExecutionEnvironment()
    return _instance


class ExecutionEnvironment:
    """Unified interface for running sandboxed containers (tools, voice, etc.)."""

    def __init__(self):
        self._docker = None
        self._docker_available = self._check_docker()
        if self._docker_available:
            logger.info("[ExecEnv] Docker available")
        else:
            logger.warning("[ExecEnv] Docker not available — tool/voice features disabled")

    def _check_docker(self) -> bool:
        """Check if Docker is available and running."""
        try:
            import docker
            self._docker = docker.from_env()
            self._docker.ping()
            return True
        except Exception:
            self._docker = None
            return False

    @property
    def is_available(self) -> bool:
        """Whether Docker is available for container operations."""
        return self._docker_available

    @property
    def client(self):
        """Raw Docker client for advanced operations. Returns None if unavailable."""
        return self._docker

    def run_container(
        self,
        image: str,
        command: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict]] = None,
        network_mode: str = "none",
        mem_limit: str = "512m",
        cpu_period: int = 100000,
        cpu_quota: int = 50000,
        timeout: int = 120,
        remove: bool = True,
        **kwargs,
    ) -> str:
        """
        Run a container and return its stdout output.

        Args:
            image: Docker image name
            command: Command to run
            environment: Environment variables
            volumes: Volume mounts
            network_mode: Network mode (default: none for isolation)
            mem_limit: Memory limit
            cpu_period/cpu_quota: CPU limits
            timeout: Execution timeout in seconds
            remove: Remove container after execution
            **kwargs: Additional docker run arguments

        Returns:
            str: Container stdout output

        Raises:
            RuntimeError: If Docker is not available
        """
        if not self._docker_available:
            raise RuntimeError("Docker is not available — cannot run containers")

        container = self._docker.containers.run(
            image,
            command=command,
            environment=environment or {},
            volumes=volumes or {},
            network_mode=network_mode,
            mem_limit=mem_limit,
            cpu_period=cpu_period,
            cpu_quota=cpu_quota,
            detach=True,
            **kwargs,
        )

        try:
            result = container.wait(timeout=timeout)
            output = container.logs(stdout=True, stderr=True).decode('utf-8', errors='replace')
            exit_code = result.get('StatusCode', -1)
            if exit_code != 0:
                logger.warning(f"[ExecEnv] Container exited with code {exit_code}")
            return output
        finally:
            if remove:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def spawn_service(
        self,
        name: str,
        image: str,
        ports: Optional[Dict[str, int]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Any:
        """
        Spawn a long-running service container.

        Args:
            name: Container name
            image: Docker image
            ports: Port mappings (e.g., {"8000/tcp": 8000})
            environment: Environment variables
            volumes: Volume mounts

        Returns:
            Container object
        """
        if not self._docker_available:
            raise RuntimeError("Docker is not available — cannot spawn services")

        # Remove existing container with same name
        try:
            existing = self._docker.containers.get(name)
            existing.remove(force=True)
        except Exception:
            pass

        container = self._docker.containers.run(
            image,
            name=name,
            ports=ports or {},
            environment=environment or {},
            volumes=volumes or {},
            detach=True,
            restart_policy={"Name": "unless-stopped"},
            **kwargs,
        )
        logger.info(f"[ExecEnv] Spawned service '{name}' (image={image})")
        return container

    def stop_service(self, name: str) -> bool:
        """Stop and remove a named service container."""
        if not self._docker_available:
            return False
        try:
            container = self._docker.containers.get(name)
            container.stop(timeout=10)
            container.remove()
            logger.info(f"[ExecEnv] Stopped service '{name}'")
            return True
        except Exception as e:
            logger.warning(f"[ExecEnv] Failed to stop service '{name}': {e}")
            return False

    def service_running(self, name: str) -> bool:
        """Check if a named service container is running."""
        if not self._docker_available:
            return False
        try:
            container = self._docker.containers.get(name)
            return container.status == 'running'
        except Exception:
            return False

    def build_image(self, path: str, tag: str, **kwargs) -> bool:
        """Build a Docker image from a Dockerfile."""
        if not self._docker_available:
            raise RuntimeError("Docker is not available — cannot build images")
        try:
            self._docker.images.build(path=path, tag=tag, rm=True, **kwargs)
            logger.info(f"[ExecEnv] Built image '{tag}'")
            return True
        except Exception as e:
            logger.error(f"[ExecEnv] Build failed for '{tag}': {e}")
            return False

    def image_exists(self, tag: str) -> bool:
        """Check if a Docker image exists locally."""
        if not self._docker_available:
            return False
        try:
            self._docker.images.get(tag)
            return True
        except Exception:
            return False
