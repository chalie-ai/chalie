"""Canonical path layout for Chalie. Hard-coded, no env overrides.

All persistent runtime state lives under ``data/`` at the repo root.
Git-tracked classifier heads live under ``backend/pre-trained/``.
Runtime-only install artifacts (voice models) live under ``resources/``.

Adding a new path? Define it here, import it everywhere else. No
``os.environ.get(...)`` fallbacks, no CLI flags, no per-service path
constants. One layout, everywhere.
"""

from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
APP_ROOT = _BACKEND_DIR.parent

# ── Persistent runtime state (gitignored, owned by the running process) ──
DATA_DIR = APP_ROOT / "data"
DB_PATH = DATA_DIR / "chalie.db"
DASHBOARD_DB_PATH = DATA_DIR / "dashboard.db"
SESSION_SECRET_PATH = DATA_DIR / ".session_secret"
DOCUMENTS_DIR = DATA_DIR / "documents"
INTERFACES_DIR = DATA_DIR / "interfaces"
MODELS_DIR = DATA_DIR / "models"

# ── Git-tracked, ship-with-Chalie assets ─────────────────────────────────
PRETRAINED_DIR = _BACKEND_DIR / "pre-trained"

# ── Runtime-only install artifact (voice models, created by install.sh) ───
RESOURCES_DIR = APP_ROOT / "resources"
