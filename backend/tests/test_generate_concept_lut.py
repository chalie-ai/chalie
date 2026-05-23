"""Tests for the generate_concept_lut script — idempotency and schema correctness.

Runs the generator against the real YAML and a temp DB so the production asset
is never touched. Marked as integration because the embedding model is required.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.integration

_YAML_PATH = FileMapperService.get_concept_lut_yaml_path()


def _run_generator(db_path: str) -> None:
    """Invoke the generator with an overridden output path."""
    import utils.generate_concept_lut as gen
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


class TestGenerateConceptLut:

    def test_produces_27_concepts(self, tmp_path):
        """Generator embeds all 27 canonical keys from the v3 YAML."""
        if not _YAML_PATH.exists():
            pytest.skip(f"YAML not found at {_YAML_PATH}")

        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            count = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
        finally:
            conn.close()

        assert count == 27, f"Expected 27 concepts (v3 LUT), got {count}"

    def test_embeddings_count_matches_concepts(self, tmp_path):
        """One embedding row per concept — no orphan or missing embeddings."""
        if not _YAML_PATH.exists():
            pytest.skip(f"YAML not found at {_YAML_PATH}")

        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            n_concepts = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
            n_embeddings = conn.execute("SELECT count(*) FROM lut_embeddings").fetchone()[0]
        finally:
            conn.close()

        assert n_embeddings == n_concepts

    def test_idempotent_second_run_replaces_cleanly(self, tmp_path):
        """Running the generator twice produces the same 27-row result — no duplicates."""
        if not _YAML_PATH.exists():
            pytest.skip(f"YAML not found at {_YAML_PATH}")

        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)
        _run_generator(db_path)  # second run — must not duplicate rows

        conn = _open_lut(db_path)
        try:
            count = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
        finally:
            conn.close()

        assert count == 27, f"After 2 runs: expected 27, got {count}"

    def test_rules_are_valid_values(self, tmp_path):
        """Every rule column value must be one of: temporal, coexist, immutable."""
        if not _YAML_PATH.exists():
            pytest.skip(f"YAML not found at {_YAML_PATH}")

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

    def test_canonical_keys_are_unique(self, tmp_path):
        """No duplicate canonical_key values in lut_concepts."""
        if not _YAML_PATH.exists():
            pytest.skip(f"YAML not found at {_YAML_PATH}")

        db_path = str(tmp_path / "concept_lut_test.sqlite")
        _run_generator(db_path)

        conn = _open_lut(db_path)
        try:
            total = conn.execute("SELECT count(*) FROM lut_concepts").fetchone()[0]
            distinct = conn.execute(
                "SELECT count(DISTINCT canonical_key) FROM lut_concepts"
            ).fetchone()[0]
        finally:
            conn.close()

        assert total == distinct, f"Duplicate canonical keys found: total={total}, distinct={distinct}"
