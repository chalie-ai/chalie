"""Canonical file-system layout for Chalie. Hard-coded, no env overrides.

Single chokepoint for every path in the codebase. Use the class-method helpers
instead of constructing paths manually.

Adding a new path? Add a classmethod here — never use Path(__file__) or
os.path.join outside of this module.
"""

from pathlib import Path


class FileMapperService:
    """Resolves all file-system paths used by Chalie.

    All constants are resolved once at class-load time. All methods are
    classmethods — no instances are created.
    """

    _CHALIE_ROOT: Path = Path(__file__).resolve().parents[2]
    _BACKEND_DIR: Path = _CHALIE_ROOT / "backend"
    _FRONTEND_DIR: Path = _CHALIE_ROOT / "frontend"
    _DATA_DIR: Path = _CHALIE_ROOT / "data"
    _RESOURCES_DIR: Path = _CHALIE_ROOT / "resources"
    _ABILITIES_DIR: Path = _BACKEND_DIR / "abilities"
    _CONFIGS_DIR: Path = _BACKEND_DIR / "configs"
    _PRETRAINED_DIR: Path = _BACKEND_DIR / "pre-trained"
    _CAPABILITIES_DIR: Path = _BACKEND_DIR / "capabilities"
    _LOGS_DIR: Path = _CHALIE_ROOT / "logs"
    _MODELS_DIR: Path = _DATA_DIR / "models"
    _DOCUMENTS_DIR: Path = _DATA_DIR / "documents"
    _USER_SKILLS_DIR: Path = _DATA_DIR / "skills" / "user"
    _VOICE_MODELS_DIR: Path = _RESOURCES_DIR / "voice-models"
    _ABILITIES_SKILLS_DIR: Path = _ABILITIES_DIR / "skills"

    # ── Well-known file paths ────────────────────────────────────────────────

    @classmethod
    def get_db_path(cls) -> Path:
        """Return path to the main SQLite database."""
        return cls._DATA_DIR / "chalie.db"

    @classmethod
    def get_session_secret_path(cls) -> Path:
        """Return path to the persisted Flask session secret."""
        return cls._DATA_DIR / ".session_secret"

    @classmethod
    def get_schema_path(cls) -> Path:
        """Return path to the declarative schema SQL file."""
        return cls._BACKEND_DIR / "schema.sql"

    @classmethod
    def get_version_path(cls) -> Path:
        """Return path to the VERSION file at the repo root."""
        return cls._CHALIE_ROOT / "VERSION"

    @classmethod
    def get_abilities_db_path(cls) -> Path:
        """Return path to the abilities vector+FTS5 search index."""
        return cls._ABILITIES_DIR / "assets" / "abilities.sqlite"

    @classmethod
    def get_skills_db_path(cls) -> Path:
        """Return path to the skills vector+FTS5 search index."""
        return cls._ABILITIES_DIR / "assets" / "skills.sqlite"

    @classmethod
    def get_search_providers_db_path(cls) -> Path:
        """Return path to the search-provider routing database."""
        return cls._BACKEND_DIR / "tools" / "search" / "assets" / "search_tool_providers.sqlite"

    @classmethod
    def get_concept_lut_db_path(cls) -> Path:
        """Return path to the concept LUT sqlite database."""
        return cls._BACKEND_DIR / "services" / "data_graph" / "assets" / "concept_lut.sqlite"

    @classmethod
    def get_concept_lut_yaml_path(cls) -> Path:
        """Return path to the concept LUT YAML source file."""
        return cls._BACKEND_DIR / "services" / "data_graph" / "assets" / "concept_lut.yaml"

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
    def get_logs_path(cls, *parts: str) -> Path:
        """Return logs/ joined with any additional path parts."""
        return cls._LOGS_DIR.joinpath(*parts) if parts else cls._LOGS_DIR

    @classmethod
    def get_user_skills_path(cls, *parts: str) -> Path:
        """Return data/skills/user/ joined with any additional path parts."""
        return cls._USER_SKILLS_DIR.joinpath(*parts) if parts else cls._USER_SKILLS_DIR

    @classmethod
    def get_voice_models_path(cls, *parts: str) -> Path:
        """Return resources/voice-models/ joined with any additional path parts."""
        return cls._VOICE_MODELS_DIR.joinpath(*parts) if parts else cls._VOICE_MODELS_DIR

    @classmethod
    def get_abilities_skills_path(cls, *parts: str) -> Path:
        """Return backend/abilities/skills/ joined with any additional path parts."""
        return cls._ABILITIES_SKILLS_DIR.joinpath(*parts) if parts else cls._ABILITIES_SKILLS_DIR

    @classmethod
    def get_capabilities_path(cls, *parts: str) -> Path:
        """Return backend/capabilities/ joined with any additional path parts."""
        return cls._CAPABILITIES_DIR.joinpath(*parts) if parts else cls._CAPABILITIES_DIR
