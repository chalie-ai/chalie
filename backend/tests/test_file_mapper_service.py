"""Feature tests for FileMapperService path resolution.

FileMapperService is a pure classmethod service — all methods are deterministic
transforms over class-level Path constants resolved once at import time.
These qualify as pure-function unit tests (no IO collaborators, no state).

Each test asks: does this method resolve to the CORRECT real-world location?
"""

import pytest

from services.file_mapper_service import FileMapperService


@pytest.mark.unit
class TestFileMapperService:

    def test_chalie_root_contains_backend_and_frontend(self):
        """_CHALIE_ROOT must point at the repo root — it must contain backend/ and frontend/."""
        root = FileMapperService._CHALIE_ROOT
        assert root.is_dir(), f"_CHALIE_ROOT does not exist: {root}"
        assert (root / "backend").is_dir(), f"backend/ missing under {root}"
        assert (root / "frontend").is_dir(), f"frontend/ missing under {root}"

    def test_get_db_path_ends_with_data_chalie_db(self):
        """get_db_path() must resolve under data/chalie.db relative to repo root."""
        db_path = FileMapperService.get_db_path()
        assert db_path.parts[-1] == "chalie.db"
        assert db_path.parts[-2] == "data"

    def test_get_schema_path_ends_correctly_and_file_exists(self):
        """get_schema_path() must resolve to the real backend/schema.sql that exists on disk."""
        schema = FileMapperService.get_schema_path()
        assert schema.parts[-1] == "schema.sql"
        assert schema.parts[-2] == "backend"
        assert schema.is_file(), f"schema.sql not found at {schema}"

    def test_get_version_path_ends_with_VERSION(self):
        """get_version_path() must point at the VERSION file at repo root."""
        version = FileMapperService.get_version_path()
        assert version.parts[-1] == "VERSION"
        assert version.is_file(), f"VERSION file not found at {version}"

    def test_get_backend_path_is_existing_dir_with_services(self):
        """get_backend_path() must return the existing backend/ directory containing services/."""
        backend = FileMapperService.get_backend_path()
        assert backend.is_dir(), f"backend path does not exist: {backend}"
        assert (backend / "services").is_dir(), f"services/ missing under backend at {backend}"

    def test_get_frontend_path_with_parts_joins_correctly(self):
        """get_frontend_path('brain') must return a path ending with frontend/brain."""
        brain = FileMapperService.get_frontend_path("brain")
        assert brain.parts[-1] == "brain"
        assert brain.parts[-2] == "frontend"

    def test_all_well_known_paths_are_under_chalie_root(self):
        """No well-known path may escape the repo root (guards against path traversal)."""
        root = FileMapperService._CHALIE_ROOT
        candidates = [
            FileMapperService.get_db_path(),
            FileMapperService.get_schema_path(),
            FileMapperService.get_version_path(),
            FileMapperService.get_abilities_db_path(),
            FileMapperService.get_skills_db_path(),
            FileMapperService.get_search_providers_db_path(),
            FileMapperService.get_concept_lut_db_path(),
            FileMapperService.get_abilities_sha_path(),
            FileMapperService.get_skills_sha_path(),
        ]
        for path in candidates:
            assert str(path).startswith(str(root) + "/"), (
                f"{path} escapes _CHALIE_ROOT {root}"
            )
