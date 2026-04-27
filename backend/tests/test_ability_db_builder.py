"""Feature tests for build_ability_db.py (Phase 0 scope).

All tests use tmp_path so they never touch the production
abilities.sqlite or abilities_sha.json.

EmbeddingService note: the empty-registry _build() path skips EmbeddingService
construction entirely (guarded by `if abilities`), so the empty round-trip tests
are fast and require no model files.  The sha-computation test for a synthetic
subclass only calls _compute_sha() which is pure — no embedding service needed.
The populated build round-trip (with real embeddings) is deferred to the Phase 1
integration test when WeatherAbility is added.
"""

import gc
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from abilities._base import Ability
from abilities._registry import _reset_for_tests
from utils.build_ability_db import _build, _compute_sha

pytestmark = pytest.mark.unit

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(autouse=True)
def clean_registry():
    """Isolate registry state around each test.

    gc.collect() before reset ensures subclasses defined in other test files
    (test_ability_base.py) are removed from Ability.__subclasses__() before
    the registry lazy-loads into a fresh _registry dict.
    """
    gc.collect()
    _reset_for_tests()
    yield
    gc.collect()
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Empty-registry round-trip (Phase 0 canonical pass)
# ---------------------------------------------------------------------------


def test_build_empty_registry_creates_all_five_schema_objects(tmp_path):
    """_build() with zero abilities creates abilities.sqlite with all 5 expected schema objects."""
    db_path = tmp_path / "abilities.sqlite"
    sha_path = tmp_path / "abilities_sha.json"

    _build(db_path, sha_path)

    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception:
        conn.load_extension("vec0")

    schema_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        ).fetchall()
    }
    conn.close()

    assert "abilities" in schema_names
    assert "ability_search_entries" in schema_names
    assert "ability_search_vec" in schema_names
    assert "ability_search_fts" in schema_names
    assert "idx_search_entries_ability" in schema_names


def test_build_empty_registry_has_zero_rows(tmp_path):
    """_build() with zero abilities produces zero rows in abilities and ability_search_entries."""
    db_path = tmp_path / "abilities.sqlite"
    sha_path = tmp_path / "abilities_sha.json"

    _build(db_path, sha_path)

    conn = sqlite3.connect(str(db_path))
    abilities_count = conn.execute("SELECT count(*) FROM abilities").fetchone()[0]
    entries_count = conn.execute("SELECT count(*) FROM ability_search_entries").fetchone()[0]
    conn.close()

    assert abilities_count == 0
    assert entries_count == 0


def test_build_empty_registry_writes_empty_sidecar(tmp_path):
    """_build() with zero abilities writes sidecar JSON of exactly {}."""
    db_path = tmp_path / "abilities.sqlite"
    sha_path = tmp_path / "abilities_sha.json"

    _build(db_path, sha_path)

    assert sha_path.exists()
    sidecar = json.loads(sha_path.read_text())
    assert sidecar == {}


# ---------------------------------------------------------------------------
# --check matches state
# ---------------------------------------------------------------------------


def test_check_passes_when_sidecar_matches_empty_registry(tmp_path):
    """_check() exits 0 when sidecar is {} and the registry has no subclasses."""
    sha_path = tmp_path / "abilities_sha.json"
    sha_path.write_text("{}\n")

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '.')
from utils.build_ability_db import _check
from pathlib import Path
_check(Path(r"{sha_path}"))
""",
        ],
        cwd=_BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}: {result.stdout}"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_check_exits_one_when_sidecar_has_ghost_entry(tmp_path):
    """_check() exits 1 and names the ghost ability when sidecar has a key the registry does not."""
    sha_path = tmp_path / "abilities_sha.json"
    sha_path.write_text(json.dumps({"ghost": "deadbeef" * 8}))

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '.')
from utils.build_ability_db import _check
from pathlib import Path
_check(Path(r"{sha_path}"))
""",
        ],
        cwd=_BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "ghost" in result.stdout, f"Expected 'ghost' in output: {result.stdout!r}"


def test_check_exits_one_when_sidecar_has_stale_sha_for_existing_ability(tmp_path):
    """_check() exits 1 when a known ability's sha in the sidecar is outdated."""

    class _DriftAbility(Ability):
        NAME = "drift_check"
        SUMMARY = "drift test ability"
        EXAMPLES = ["check drift", "detect change", "hash mismatch", "stale sidecar", "sha wrong", "outdated"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    _reset_for_tests()

    sha_path = tmp_path / "abilities_sha.json"
    sha_path.write_text(json.dumps({"drift_check": "0000000000000000000000000000000000000000000000000000000000000000"}))

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '.')
# define the same ability so registry finds it
from abilities._base import Ability
class _DriftAbility(Ability):
    NAME = 'drift_check'
    SUMMARY = 'drift test ability'
    EXAMPLES = ['check drift', 'detect change', 'hash mismatch', 'stale sidecar', 'sha wrong', 'outdated']
    INPUT_SCHEMA = {{}}
    def execute(self, channel, params, telemetry): return {{}}

from utils.build_ability_db import _check
from pathlib import Path
_check(Path(r"{sha_path}"))
""",
        ],
        cwd=_BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "drift_check" in result.stdout

    del _DriftAbility
    gc.collect()


# ---------------------------------------------------------------------------
# SHA computation — pure function, no embedding service
# ---------------------------------------------------------------------------


def test_compute_sha_matches_expected_formula():
    """_compute_sha() produces sha256(json([SUMMARY, *EXAMPLES]))."""

    class _ShaAbility(Ability):
        NAME = "sha_ability"
        SUMMARY = "Checks the current weather conditions"
        EXAMPLES = [
            "what is the weather",
            "will it rain today",
            "temperature outside",
            "do I need an umbrella",
            "is it sunny",
            "weather forecast",
        ]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    instance = _ShaAbility()
    computed = _compute_sha(instance)

    raw = json.dumps([instance.SUMMARY, *instance.EXAMPLES], ensure_ascii=False)
    expected = hashlib.sha256(raw.encode()).hexdigest()

    assert computed == expected, f"SHA mismatch: {computed!r} != {expected!r}"

    del _ShaAbility
    gc.collect()


def test_compute_sha_is_deterministic():
    """_compute_sha() returns the same value on repeated calls for the same ability."""

    class _DetAbility(Ability):
        NAME = "det_ability"
        SUMMARY = "Deterministic sha check"
        EXAMPLES = ["one", "two", "three", "four", "five", "six"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    instance = _DetAbility()
    sha1 = _compute_sha(instance)
    sha2 = _compute_sha(instance)

    assert sha1 == sha2

    del _DetAbility
    gc.collect()


def test_compute_sha_differs_when_summary_changes():
    """Two abilities identical except SUMMARY produce different SHAs."""

    class _ShaA(Ability):
        NAME = "sha_a"
        SUMMARY = "Summary version A"
        EXAMPLES = ["one", "two", "three", "four", "five", "six"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    class _ShaB(Ability):
        NAME = "sha_b"
        SUMMARY = "Summary version B"
        EXAMPLES = ["one", "two", "three", "four", "five", "six"]
        INPUT_SCHEMA = {}

        def execute(self, channel, params, telemetry):
            return {}

    assert _compute_sha(_ShaA()) != _compute_sha(_ShaB())

    del _ShaA, _ShaB
    gc.collect()
