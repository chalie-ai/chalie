"""Tests that FileMapperService resolves paths to correct locations.

The main database is versioned (``data/chalie-<VERSION>.sqlite``); the
naming convention lives in FileMapperService, and these tests exercise it
through the builder (version → path) and its inverse (path → version).
"""

import pytest

from services.file_mapper_service import FileMapperService


def _version() -> str:
    """The repo-root VERSION content, exactly as the running build reads it."""
    return FileMapperService.get_version_path().read_text().strip()


@pytest.mark.unit
class TestFileMapperService:

    def test_get_db_path_is_versioned(self) -> None:
        """The running db is data/chalie-<VERSION file content>.sqlite under data/."""
        db_path = FileMapperService.get_db_path()
        assert db_path.parts[-1] == f"chalie-{_version()}.sqlite"
        assert db_path.parts[-2] == "data"

    def test_get_legacy_db_path_is_data_chalie_db(self) -> None:
        """The pre-versioning file name is preserved as the legacy path."""
        legacy = FileMapperService.get_legacy_db_path()
        assert legacy.parts[-1] == "chalie.db"
        assert legacy.parts[-2] == "data"

    def test_versioned_db_path_round_trips_through_its_inverse(self) -> None:
        """version → path → version is the identity for versioned names."""
        for version in ("1.3.0-beta", "1.10.0", "2.0.0"):
            path = FileMapperService.get_versioned_db_path(version)
            assert path.name == f"chalie-{version}.sqlite"
            assert path.parent.name == "data"
            assert FileMapperService.version_from_db_path(path) == version
        # The running build's db path round-trips with the running version.
        assert FileMapperService.version_from_db_path(FileMapperService.get_db_path()) == _version()

    def test_version_from_db_path_is_none_for_unversioned_names(self) -> None:
        """Only chalie-<version>.sqlite names encode a version — everything else is None."""
        for path in (
            FileMapperService.get_legacy_db_path(),
            FileMapperService.get_mcp_tools_db_path(),
            FileMapperService.get_file_index_db_path(),
            FileMapperService.get_skills_db_path(),
            "chalie-.sqlite",
            "chalie-1.3.0",
            "db-chalie-1.3.0-beta.sqlite",
        ):
            assert FileMapperService.version_from_db_path(path) is None, str(path)

    def test_get_schema_path_ends_correctly_and_file_exists(self) -> None:
        schema = FileMapperService.get_schema_path()
        assert schema.parts[-1] == "schema.sql"
        assert schema.parts[-2] == "backend"
        assert schema.is_file(), f"schema.sql not found at {schema}"

    def test_get_version_path_ends_with_VERSION(self) -> None:
        version = FileMapperService.get_version_path()
        assert version.parts[-1] == "VERSION"
        assert version.is_file(), f"VERSION file not found at {version}"

    def test_get_backend_path_is_existing_dir_with_services(self) -> None:
        backend = FileMapperService.get_backend_path()
        assert backend.is_dir(), f"backend path does not exist: {backend}"
        assert (backend / "services").is_dir(), f"services/ missing under backend at {backend}"

    def test_all_well_known_paths_are_under_chalie_root(self) -> None:
        """No well-known path may escape the repo root (guards against path traversal)."""
        root = FileMapperService._CHALIE_ROOT
        candidates = [
            FileMapperService.get_db_path(),
            FileMapperService.get_legacy_db_path(),
            FileMapperService.get_schema_path(),
            FileMapperService.get_version_path(),
            FileMapperService.get_skills_db_path(),
            FileMapperService.get_search_providers_db_path(),
            FileMapperService.get_abilities_sha_path(),
            FileMapperService.get_skills_sha_path(),
        ]
        for path in candidates:
            assert str(path).startswith(str(root) + "/"), (
                f"{path} escapes _CHALIE_ROOT {root}"
            )
