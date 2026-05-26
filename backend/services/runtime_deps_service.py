"""Runtime dependency management — background installation of optional features.

Manages playwright (auto-install on boot) and voice (user-triggered via settings).
All operations run in daemon threads and report status via class methods.
"""

import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import requests

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class RuntimeDepsService:
    """Singleton managing optional dependency installation."""

    _voice_status: str = "unknown"  # unknown | installing | available | unavailable | error
    _voice_error: str | None = None
    _playwright_status: str = "unknown"  # unknown | installing | available | error
    _playwright_error: str | None = None
    _lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def ensure_playwright(cls) -> None:
        """Fire-and-forget: install/update Playwright Chromium in background."""
        t = threading.Thread(
            target=cls._install_playwright,
            name="runtime-deps-playwright",
            daemon=True,
        )
        t.start()

    @classmethod
    def enable_voice(cls) -> dict:
        """Trigger voice dependency installation in background.

        Returns immediate status dict.
        """
        with cls._lock:
            if cls._voice_status == "installing":
                return {"status": "installing", "message": "Voice installation already in progress"}
            cls._voice_status = "installing"
            cls._voice_error = None
        t = threading.Thread(
            target=cls._install_voice,
            name="runtime-deps-voice",
            daemon=True,
        )
        t.start()
        return {"status": "installing", "message": "Voice dependencies installing in background"}

    @classmethod
    def init_voice_from_settings(cls, database_service) -> None:
        """Check DB settings on boot — if voice is enabled, trigger install.

        Migration: if voice_enabled is not yet in the DB but the legacy
        .voice-deps-installed stamp exists, the user had voice before this
        runtime-managed architecture.  Honour that by writing "true" and
        removing the stale stamp (one-time migration).
        """
        try:
            from services.settings_service import SettingsService
            settings = SettingsService(database_service)
            voice_enabled = settings.get("voice_enabled")

            if voice_enabled is None:
                stamp = FileMapperService.get_chalie_root() / ".voice-deps-installed"
                if stamp.exists():
                    voice_enabled = "true"
                    settings.set("voice_enabled", "true")
                    stamp.unlink()
                    logger.info("[RuntimeDeps] Migrated legacy voice stamp → voice_enabled=true")
                else:
                    cls._voice_status = "unavailable"
                    return

            if voice_enabled == "true":
                if cls.check_voice_available():
                    cls._voice_status = "available"
                    logger.info("[RuntimeDeps] Voice already available (deps present)")
                else:
                    cls.enable_voice()
            else:
                cls._voice_status = "unavailable"
        except Exception as e:
            logger.warning("[RuntimeDeps] Could not read voice setting: %s", e)
            cls._voice_status = "unknown"

    @classmethod
    def get_status(cls) -> dict:
        """Return current status of optional dependencies."""
        return {
            "voice": {"status": cls._voice_status, "error": cls._voice_error},
            "playwright": {"status": cls._playwright_status, "error": cls._playwright_error},
        }

    @classmethod
    def check_voice_available(cls) -> bool:
        """Check if voice modules are importable right now."""
        required = (
            "kokoro_onnx",
            "moonshine_onnx",
            "soundfile",
            "numpy",
            "silero_vad_lite",
            "noisereduce",
        )
        return all(importlib.util.find_spec(m) is not None for m in required)

    # ── Private: Playwright ───────────────────────────────────────────────

    @classmethod
    def _install_playwright(cls) -> None:
        with cls._lock:
            cls._playwright_status = "installing"
        try:
            pw_bin = shutil.which("playwright")
            if not pw_bin:
                venv_bin = Path(sys.executable).parent
                candidate = venv_bin / "playwright"
                if candidate.exists():
                    pw_bin = str(candidate)
                else:
                    with cls._lock:
                        cls._playwright_status = "error"
                        cls._playwright_error = "playwright binary not found in venv"
                    logger.warning("[RuntimeDeps] playwright binary not found")
                    return

            cmd = [pw_bin, "install", "chromium"]
            if platform.system() == "Linux":
                cmd.append("--with-deps")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                with cls._lock:
                    cls._playwright_status = "available"
                logger.info("[RuntimeDeps] Playwright Chromium ready")
            else:
                error_snippet = result.stderr[:200] if result.stderr else "install failed"
                with cls._lock:
                    cls._playwright_status = "error"
                    cls._playwright_error = error_snippet
                logger.warning("[RuntimeDeps] Playwright install failed: %s", error_snippet)
        except subprocess.TimeoutExpired:
            with cls._lock:
                cls._playwright_status = "error"
                cls._playwright_error = "install timed out (600s)"
            logger.warning("[RuntimeDeps] Playwright install timed out")
        except Exception as e:
            with cls._lock:
                cls._playwright_status = "error"
                cls._playwright_error = str(e)
            logger.warning("[RuntimeDeps] Playwright install error: %s", e)

    # ── Private: Voice ────────────────────────────────────────────────────

    @classmethod
    def _install_voice(cls) -> None:
        """Full voice installation: deps → GPU detection → ORT wheel → models."""
        try:
            cls._install_voice_deps()
            cls._download_voice_models()
            if cls.check_voice_available():
                with cls._lock:
                    cls._voice_status = "available"
                logger.info("[RuntimeDeps] Voice fully available")
            else:
                with cls._lock:
                    cls._voice_status = "error"
                    cls._voice_error = "deps installed but modules not importable"
                logger.warning("[RuntimeDeps] Voice deps installed but not importable")
        except Exception as e:
            with cls._lock:
                cls._voice_status = "error"
                cls._voice_error = str(e)
            logger.error("[RuntimeDeps] Voice installation failed: %s", e)

    @classmethod
    def _install_voice_deps(cls) -> None:
        """Install voice Python packages with the correct ORT variant."""
        group = cls._detect_voice_group()
        logger.info("[RuntimeDeps] Installing voice group: %s", group)

        backend_dir = str(FileMapperService.get_backend_path())
        install_target = f"{backend_dir}[{group}]"

        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", "--python", sys.executable, "-e", install_target]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "-e", install_target]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Voice dep install failed: {result.stderr[:300]}")
        logger.info("[RuntimeDeps] Voice deps installed (%s)", group)

    @classmethod
    def _detect_voice_group(cls) -> str:
        """Detect GPU and return the appropriate pyproject.toml optional group name."""
        system = platform.system()

        if system == "Darwin":
            return "voice-cpu"

        if system == "Linux":
            if shutil.which("nvidia-smi"):
                try:
                    result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
                    if result.returncode == 0:
                        logger.info("[RuntimeDeps] NVIDIA GPU detected")
                        return "voice-cuda"
                except (subprocess.TimeoutExpired, OSError):
                    pass

            if os.path.exists("/dev/kfd") and os.path.isdir("/sys/module/amdgpu"):
                logger.info("[RuntimeDeps] AMD ROCm GPU detected")
                return "voice-rocm"

        return "voice-cpu"

    @classmethod
    def _download_voice_models(cls) -> None:
        """Download Kokoro TTS + Moonshine STT ONNX models."""
        voice_root = FileMapperService.get_voice_models_path()
        kokoro_dir = voice_root / "kokoro"
        moonshine_dir = voice_root / "moonshine" / "base"
        kokoro_dir.mkdir(parents=True, exist_ok=True)
        moonshine_dir.mkdir(parents=True, exist_ok=True)

        kokoro_release = (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        )
        moonshine_hf = (
            "https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/base/float"
        )

        models = [
            (
                f"{kokoro_release}/kokoro-v1.0.onnx",
                kokoro_dir / "kokoro-v1.0.onnx",
                "Kokoro TTS model",
            ),
            (
                f"{kokoro_release}/voices-v1.0.bin",
                kokoro_dir / "voices-v1.0.bin",
                "Kokoro voices",
            ),
            (
                f"{moonshine_hf}/encoder_model.onnx",
                moonshine_dir / "encoder_model.onnx",
                "Moonshine encoder",
            ),
            (
                f"{moonshine_hf}/decoder_model_merged.onnx",
                moonshine_dir / "decoder_model_merged.onnx",
                "Moonshine decoder",
            ),
        ]

        for url, dest, label in models:
            if dest.exists() and dest.stat().st_size > 1024:
                continue
            logger.info("[RuntimeDeps] Downloading %s...", label)
            try:
                resp = requests.get(url, stream=True, timeout=600)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("[RuntimeDeps] %s downloaded", label)
            except Exception as e:
                logger.warning("[RuntimeDeps] %s download failed: %s", label, e)
                if dest.exists():
                    dest.unlink()
