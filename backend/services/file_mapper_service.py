"""Canonical file-system layout for Chalie. Hard-coded, no env overrides.

Single chokepoint for every path in the codebase. Use the class-method helpers
instead of constructing paths manually.

Adding a new path? Add a classmethod here — never use Path(__file__) or
os.path.join outside of this module.
"""

from pathlib import Path

# Dotted scratch directories under data/ used by the snapshot Time-Machine.
_PENDING_RESTORE_DIR = ".pending-restore"
_SNAPSHOT_STAGING_DIR = ".snapshot-staging"


class FileMapperService:
    """Resolves all file-system paths used by Chalie.

    All constants are resolved once at class-load time. All methods are
    classmethods — no instances are created.
    """

    _CHALIE_ROOT: Path = Path(__file__).resolve().parents[2]
    _BACKEND_DIR: Path = _CHALIE_ROOT / "backend"
    _FRONTEND_DIR: Path = _CHALIE_ROOT / "frontend"
    _DATA_DIR: Path = _CHALIE_ROOT / "data"
    _SECURE_DIR: Path = _DATA_DIR / "secure"
    _RESOURCES_DIR: Path = _CHALIE_ROOT / "resources"
    _ABILITIES_DIR: Path = _BACKEND_DIR / "abilities"
    _CONFIGS_DIR: Path = _BACKEND_DIR / "configs"
    _PRETRAINED_DIR: Path = _BACKEND_DIR / "pre-trained"
    _CAPABILITIES_DIR: Path = _BACKEND_DIR / "capabilities"
    _MODELS_DIR: Path = _DATA_DIR / "models"
    _DOCUMENTS_DIR: Path = _DATA_DIR / "documents"
    _USER_SKILLS_DIR: Path = _DATA_DIR / "skills" / "user"
    _VOICE_MODELS_DIR: Path = _RESOURCES_DIR / "voice-models"
    _VOICE_DIR: Path = _DATA_DIR / "generated" / "voice"
    _ABILITIES_SKILLS_DIR: Path = _ABILITIES_DIR / "skills"
    _CODE_AGENT_WORKSPACE_DIR: Path = _DATA_DIR / "code_agent_workspace"

    # ── Well-known file paths ────────────────────────────────────────────────

    @classmethod
    def get_db_path(cls) -> Path:
        """Return path to the main SQLite database."""
        return cls._DATA_DIR / "chalie.db"

    @classmethod
    def get_secure_dir(cls) -> Path:
        """Return the vault key-material backups directory."""
        return cls._SECURE_DIR

    @classmethod
    def get_vault_backup_path(cls, stamp: str) -> Path:
        """Return the vault key-material backup path for a given stamp."""
        return cls._SECURE_DIR / f"vault_backup_{stamp}.json"

    @classmethod
    def list_vault_backups(cls) -> list[Path]:
        """Return all vault backup files, newest first."""
        if not cls._SECURE_DIR.is_dir():
            return []
        return sorted(cls._SECURE_DIR.glob("vault_backup_*.json"), reverse=True)

    @classmethod
    def get_ssl_cert_path(cls) -> Path:
        """TLS certificate (PEM) uploaded via the System page; stored 0600 in the secure dir."""
        return cls._SECURE_DIR / "ssl_cert.pem"

    @classmethod
    def get_ssl_key_path(cls) -> Path:
        """TLS private key (PEM) uploaded via the System page; stored 0600 in the secure dir."""
        return cls._SECURE_DIR / "ssl_key.pem"

    @classmethod
    def get_schema_path(cls) -> Path:
        """Return path to the declarative schema SQL file."""
        return cls._BACKEND_DIR / "schema.sql"

    @classmethod
    def get_version_path(cls) -> Path:
        """Return path to the VERSION file at the repo root."""
        return cls._CHALIE_ROOT / "VERSION"

    @classmethod
    def get_dev_credentials_path(cls) -> Path:
        """Return path to the optional dev-only auto-login credentials file.

        Local convenience only: if present at the install root, its username/
        password are tried first to unlock the vault, bypassing the login
        page. Gitignored — never bundled into a release or committed.
        """
        return cls._CHALIE_ROOT / "credentials.json"

    @classmethod
    def get_skills_db_path(cls) -> Path:
        """Return path to the skills vector+FTS5 search index."""
        return cls._ABILITIES_DIR / "assets" / "skills.sqlite"

    @classmethod
    def get_policy_defaults_path(cls) -> Path:
        """Return path to the static hand-authored policy seed (flat triples)."""
        return cls._ABILITIES_DIR / "assets" / "policy_defaults.json"

    @classmethod
    def get_mcp_tools_db_path(cls) -> Path:
        """Return path to the runtime MCP-tools index (gitignored, data/).

        Managed exclusively by McpClientService — never rebuilt by the
        abilities pipeline.
        """
        return cls._DATA_DIR / "mcp_tools.sqlite"

    @classmethod
    def get_file_index_db_path(cls) -> Path:
        """Return path to the file-index FTS5 search database (gitignored, data/).

        Managed exclusively by FileIndexService — never touched by any
        other service.
        """
        return cls._DATA_DIR / "file_index.sqlite"

    @classmethod
    def get_search_providers_db_path(cls) -> Path:
        """Return path to the search-provider routing database."""
        return cls._BACKEND_DIR / "tools" / "search" / "assets" / "search_tool_providers.sqlite"

    @classmethod
    def get_abilities_sha_path(cls) -> Path:
        """Return path to the abilities drift-sidecar SHA file."""
        return cls._PRETRAINED_DIR / "abilities_sha.json"

    @classmethod
    def get_skills_sha_path(cls) -> Path:
        """Return path to the skills drift-sidecar SHA file."""
        return cls._PRETRAINED_DIR / "skills_sha.json"

    # ── Directory helpers ────────────────────────────────────────────────────

    @classmethod
    def get_chalie_root(cls, *parts: str) -> Path:
        """Return the repo root joined with any additional path parts."""
        return cls._CHALIE_ROOT.joinpath(*parts) if parts else cls._CHALIE_ROOT

    @classmethod
    def get_backend_path(cls, *parts: str) -> Path:
        """Return backend/ joined with any additional path parts."""
        return cls._BACKEND_DIR.joinpath(*parts) if parts else cls._BACKEND_DIR

    @classmethod
    def get_frontend_path(cls, *parts: str) -> Path:
        """Return frontend/ joined with any additional path parts."""
        return cls._FRONTEND_DIR.joinpath(*parts) if parts else cls._FRONTEND_DIR

    @classmethod
    def get_data_path(cls, *parts: str) -> Path:
        """Return data/ joined with any additional path parts."""
        return cls._DATA_DIR.joinpath(*parts) if parts else cls._DATA_DIR

    @classmethod
    def get_pending_restore_path(cls, *parts: str) -> Path:
        """Return the staged-restore directory (``data/.pending-restore``)."""
        return cls.get_data_path(_PENDING_RESTORE_DIR, *parts)

    @classmethod
    def get_snapshot_staging_path(cls, *parts: str) -> Path:
        """Return the data-level scratch directory for assembling a snapshot."""
        return cls.get_data_path(_SNAPSHOT_STAGING_DIR, *parts)

    @classmethod
    def get_resources_path(cls, *parts: str) -> Path:
        """Return resources/ joined with any additional path parts."""
        return cls._RESOURCES_DIR.joinpath(*parts) if parts else cls._RESOURCES_DIR

    @classmethod
    def get_abilities_path(cls, *parts: str) -> Path:
        """Return backend/abilities/ joined with any additional path parts."""
        return cls._ABILITIES_DIR.joinpath(*parts) if parts else cls._ABILITIES_DIR

    @classmethod
    def get_configs_path(cls, *parts: str) -> Path:
        """Return backend/configs/ joined with any additional path parts."""
        return cls._CONFIGS_DIR.joinpath(*parts) if parts else cls._CONFIGS_DIR

    @classmethod
    def get_pretrained_path(cls, *parts: str) -> Path:
        """Return backend/pre-trained/ joined with any additional path parts."""
        return cls._PRETRAINED_DIR.joinpath(*parts) if parts else cls._PRETRAINED_DIR

    @classmethod
    def get_models_path(cls, *parts: str) -> Path:
        """Return data/models/ joined with any additional path parts."""
        return cls._MODELS_DIR.joinpath(*parts) if parts else cls._MODELS_DIR

    @classmethod
    def get_documents_path(cls, *parts: str) -> Path:
        """Return data/documents/ joined with any additional path parts."""
        return cls._DOCUMENTS_DIR.joinpath(*parts) if parts else cls._DOCUMENTS_DIR

    @classmethod
    def get_user_skills_path(cls, *parts: str) -> Path:
        """Return data/skills/user/ joined with any additional path parts."""
        return cls._USER_SKILLS_DIR.joinpath(*parts) if parts else cls._USER_SKILLS_DIR

    @classmethod
    def get_voice_models_path(cls, *parts: str) -> Path:
        """Return resources/voice-models/ joined with any additional path parts."""
        return cls._VOICE_MODELS_DIR.joinpath(*parts) if parts else cls._VOICE_MODELS_DIR

    @classmethod
    def get_voice_path(cls, *parts: str) -> Path:
        """Return data/generated/voice/ joined with any additional path parts."""
        return cls._VOICE_DIR.joinpath(*parts) if parts else cls._VOICE_DIR

    @classmethod
    def get_abilities_skills_path(cls, *parts: str) -> Path:
        """Return backend/abilities/skills/ joined with any additional path parts."""
        return cls._ABILITIES_SKILLS_DIR.joinpath(*parts) if parts else cls._ABILITIES_SKILLS_DIR

    @classmethod
    def get_capabilities_path(cls, *parts: str) -> Path:
        """Return backend/capabilities/ joined with any additional path parts."""
        return cls._CAPABILITIES_DIR.joinpath(*parts) if parts else cls._CAPABILITIES_DIR

    @classmethod
    def get_code_agent_workspace_path(cls, *parts: str) -> Path:
        """Return data/code_agent_workspace/ joined with any additional path parts.

        The default scratch location for code_agent work — a convention, not
        an enforced boundary; created lazily on first use, not at import time.
        """
        return cls._CODE_AGENT_WORKSPACE_DIR.joinpath(*parts) if parts else cls._CODE_AGENT_WORKSPACE_DIR

    @classmethod
    def get_web_pages_path(cls, *parts: str) -> Path:
        """Return data/web/pages/ joined with any additional path parts."""
        base = cls._DATA_DIR / "web" / "pages"
        return base.joinpath(*parts) if parts else base

    @classmethod
    def get_downloads_path(cls, *parts: str) -> Path:
        """Return data/downloads/ joined with any additional path parts."""
        base = cls._DATA_DIR / "downloads"
        return base.joinpath(*parts) if parts else base

    @classmethod
    def validate_document_path(cls, full_path: str) -> bool:
        """Check if *full_path* resolves inside the documents root."""
        import os
        real = os.path.realpath(full_path)
        root = os.path.realpath(str(cls._DOCUMENTS_DIR))
        return real.startswith(root + os.sep) or real == root
