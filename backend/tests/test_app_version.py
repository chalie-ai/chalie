"""Tests for services.app_version — the single place that reads and compares the app version."""

from pathlib import Path

import pytest

from services import app_version
from services.file_mapper_service import FileMapperService


@pytest.mark.unit
class TestAppVersion:

    def test_get_version_reads_the_version_file(self) -> None:
        """get_version returns exactly the stripped content of the repo-root VERSION file."""
        expected = FileMapperService.get_version_path().read_text().strip()
        assert app_version.get_version() == expected

    def test_missing_version_file_raises_naming_the_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing VERSION file raises (no "0.0.0" fallback) and names the path."""
        missing = tmp_path / "VERSION"
        monkeypatch.setattr(FileMapperService, "get_version_path", lambda *_: missing)
        with pytest.raises(FileNotFoundError, match=str(missing)):
            app_version.get_version()

    def test_empty_version_file_raises_naming_the_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An empty VERSION file raises (no fallback) and names the path."""
        empty = tmp_path / "VERSION"
        empty.write_text("   \n", encoding="utf-8")
        monkeypatch.setattr(FileMapperService, "get_version_path", lambda *_: empty)
        with pytest.raises(ValueError, match=str(empty)):
            app_version.get_version()

    def test_sort_key_orders_numerically_per_dotted_part(self) -> None:
        """1.2.0 < 1.10.0 — each dotted part compares as a number, not a string."""
        versions = ("1.10.0", "1.2.0", "1.2.10", "1.10.2")
        assert sorted(versions, key=app_version.version_sort_key) == [
            "1.2.0", "1.2.10", "1.10.0", "1.10.2",
        ]

    def test_sort_key_puts_pre_release_before_its_final(self) -> None:
        """1.3.0-beta < 1.3.0 < 1.3.1 — a pre-release sorts before its final release."""
        versions = ("1.3.1", "1.3.0-beta", "1.3.0")
        assert sorted(versions, key=app_version.version_sort_key) == [
            "1.3.0-beta", "1.3.0", "1.3.1",
        ]
