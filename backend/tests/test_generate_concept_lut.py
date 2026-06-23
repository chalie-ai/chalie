"""Tests for generate_concept_lut — idempotency and schema correctness."""

import sqlite3
from pathlib import Path
from typing import cast

import pytest
import yaml

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.integration

_YAML_PATH = FileMapperService.get_concept_lut_yaml_path()


def _canonical_keys_from_yaml() -> set[str]:
    with open(_YAML_PATH) as f:
        data = yaml.safe_load(f)
    return {c["canonical_key"] for c in data.get("concepts", [])}


def _encoder_available() -> bool:
    try:
        from services.embedding_service import EmbeddingService

        EmbeddingService().generate_embeddings_batch(["probe"])
        return True
    except Exception:
        return False


def _run_generator(db_path: str) -> None:
    import utils.generate_concept_lut as gen
    from pathlib import Path

    original_db = gen._DB_PATH
    try:
        gen._DB_PATH = Path(db_path)
        gen.main()
    finally:
        gen._DB_PATH = original_db

def _open_lut(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension('vec0')
    return conn


def _require_generator_prereqs() -> None:
    if not _YAML_PATH.exists():
        pytest.skip(f"YAML not found at {_YAML_PATH}")
    if not _encoder_available():
        pytest.skip("ONNX embedding encoder unavailable")


class TestGenerateConceptLut:

    def test_row_count_covers_every_canonical_key(self, tmp_path: Path) -> None:
        _require_generator_prereqs()

        expected_canonical = len(_canonical_keys_from_yaml())
        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            total = cast(int, conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0])
            distinct = cast(int, conn.execute(
                "SELECT count(DISTINCT canonical_key) FROM lut_concepts"
            ).fetchone()[0])
        finally:
            conn.close()

        assert distinct == expected_canonical, (
            f"Expected {expected_canonical} distinct canonical keys, got {distinct}"
        )
        assert total >= expected_canonical, (
            f"Expected at least {expected_canonical} label rows, got {total}"
        )

    def test_embeddings_count_matches_concepts(self, tmp_path: Path) -> None:
        _require_generator_prereqs()

        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            n_concepts = cast(int, conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0])
            n_embeddings = cast(int, conn.execute("SELECT count(*) FROM lut_embeddings").fetchone()[0])
        finally:
            conn.close()

        assert n_embeddings == n_concepts

    def test_idempotent_second_run_replaces_cleanly(self, tmp_path: Path) -> None:
        """Running the generator twice produces the same result — no duplicates."""
        _require_generator_prereqs()

        expected_canonical = len(_canonical_keys_from_yaml())
        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)
        first = _row_count(db_path)
        _run_generator(db_path)  # second run — must not duplicate rows
        second = _row_count(db_path)

        assert first == second, f"Row count changed between runs: {first} -> {second}"
        assert second >= expected_canonical, (
            f"After 2 runs: expected at least {expected_canonical}, got {second}"
        )

    def test_rules_are_valid_values(self, tmp_path: Path) -> None:
        """Every rule column value must be one of: temporal, coexist, immutable."""
        _require_generator_prereqs()

        valid_rules = {'temporal', 'coexist', 'immutable'}
        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            rows = conn.execute("SELECT rule FROM lut_concepts").fetchall()
        finally:
            conn.close()

        invalid = [r[0] for r in rows if r[0] not in valid_rules]
        assert not invalid, f"Invalid rule values found: {invalid}"

    def test_canonical_keys_match_yaml(self, tmp_path: Path) -> None:
        """The distinct canonical keys in the DB match the source YAML set.

        Each alias gets its own row pointing to the same canonical_key, so the
        invariant is on the *distinct* canonical keys, not total row count.
        """
        _require_generator_prereqs()

        expected_keys = _canonical_keys_from_yaml()
        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            rows = conn.execute("SELECT DISTINCT canonical_key FROM lut_concepts").fetchall()
        finally:
            conn.close()

        db_keys = {r[0] for r in rows}
        assert db_keys == expected_keys, (
            f"Canonical keys diverge from YAML: "
            f"missing={expected_keys - db_keys}, extra={db_keys - expected_keys}"
        )


def _row_count(db_path: str) -> int:
    conn = _open_lut(db_path)
    try:
        return cast(int, conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0])
    finally:
        conn.close()
