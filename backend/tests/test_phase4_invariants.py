"""Phase 4 dispatch-cutover invariant tests.

These tests assert that the legacy dispatch paths (innate_skills submodules,
legacy tools/*.py files, tool_capability_profiles table, ToolProfileService)
have been fully removed and cannot silently reappear via future commits.

They complement test_ability_phase2_contracts.py (ABC contract checks) and
test_ability_registry.py (registry mechanics) — there is no overlap.

Tests run under pytest -m unit (no external deps).

NOTE: Tests in this file deliberately avoid calling AbilityRegistry.all() at
module level for the registry-count and walk-exclusion checks, because other
test files (test_pattern_match_processor.py) import SavePatternAbility and
SaveGraphAbility directly, which contaminates Ability.__subclasses__() for
the rest of the test session. Instead, those tests verify the disk layout and
registry walk mechanism — the source of truth that is always accurate
regardless of what other tests have imported.
"""

import ast
import sqlite3
from pathlib import Path

import pytest

import abilities._registry as _reg_module

pytestmark = pytest.mark.unit

# Root of the backend source tree — all path calculations derive from here.
_BACKEND_DIR = Path(__file__).resolve().parent.parent

# Production directories to scan for legacy imports.
# tests/ is excluded deliberately: test files may legitimately reference
# deleted names to verify removal (like this file does).
_PROD_DIRS = [
    _BACKEND_DIR / "abilities",
    _BACKEND_DIR / "api",
    _BACKEND_DIR / "capabilities",
    _BACKEND_DIR / "services",
    _BACKEND_DIR / "tools",
    _BACKEND_DIR / "utils",
    _BACKEND_DIR / "workers",
]

# Patterns that must not appear in any import statement in production code.
# Each entry is a tuple of (description, set_of_banned_import_module_prefixes).
# The check is AST-based to avoid false positives from comments or strings.
_BANNED_IMPORT_PREFIXES = [
    # Legacy innate-skill submodules other than the kept _tag helper.
    # e.g. "from services.innate_skills.find_tools_skill import ..."
    #      "from services.innate_skills.memory_skill import ..."
    # The _tag submodule is explicitly allowed — it is the shared response-tag
    # helper still used by ability wrappers.
    "services.innate_skills.find_tools_skill",
    "services.innate_skills.memory_skill",
    "services.innate_skills.weather_skill",
    "services.innate_skills.news_skill",
    "services.innate_skills.code_eval_skill",
    "services.innate_skills.search_skill",
    "services.innate_skills.browser_skill",
    "services.innate_skills.schedule_skill",
    "services.innate_skills.list_skill",
    "services.innate_skills.document_skill",
    "services.innate_skills.goal_pursuit_skill",
    "services.innate_skills.read_skill",
    "services.innate_skills.review_tool_calls_skill",
    "services.innate_skills.rich_render_skill",
    # Legacy top-level tool module imports (pre-Phase-4 dispatch path).
    "tools.weather",
    "tools.code_eval",
    "tools.programming_docs_search",
    "tools.news",
    "tools.search.search",
    "tools.browser.browser",
]

# Names that must not appear as attribute references or standalone identifiers
# in production import statements.
_BANNED_IMPORT_NAMES = [
    "ToolProfileService",
]


def _collect_python_files(dirs: list[Path]) -> list[Path]:
    """Return all .py files under *dirs*, skipping __pycache__."""
    files: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            files.append(f)
    return files


def _extract_imports(source: str) -> list[ast.Import | ast.ImportFrom]:
    """Parse *source* and return all Import / ImportFrom nodes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


# ---------------------------------------------------------------------------
# Test 1 — No legacy import paths in production code
# ---------------------------------------------------------------------------


def test_no_legacy_import_paths_in_production_code():
    """No production file imports a deleted legacy dispatch path.

    Walks abilities/, api/, capabilities/, services/, tools/, utils/, workers/
    and asserts that no import statement references:
      - Any services.innate_skills submodule other than _tag
      - tools.weather, tools.code_eval, tools.programming_docs_search,
        tools.news, tools.search.search, tools.browser.browser
      - ToolProfileService

    This is the regression gate: if a future commit reintroduces one of these
    paths, this test catches it before it reaches production.
    """
    violations: list[str] = []
    files = _collect_python_files(_PROD_DIRS)

    for fpath in files:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        for node in _extract_imports(source):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Check banned module prefixes.
                for banned in _BANNED_IMPORT_PREFIXES:
                    if module == banned or module.startswith(banned + "."):
                        violations.append(
                            f"{fpath.relative_to(_BACKEND_DIR)}:{node.lineno} "
                            f"— banned import 'from {module}'"
                        )
                # Check banned names appearing as imported symbols.
                for alias in node.names:
                    if alias.name in _BANNED_IMPORT_NAMES:
                        violations.append(
                            f"{fpath.relative_to(_BACKEND_DIR)}:{node.lineno} "
                            f"— banned symbol '{alias.name}' imported from '{module}'"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    for banned in _BANNED_IMPORT_PREFIXES:
                        if name == banned or name.startswith(banned + "."):
                            violations.append(
                                f"{fpath.relative_to(_BACKEND_DIR)}:{node.lineno} "
                                f"— banned import '{name}'"
                            )
                    if name in _BANNED_IMPORT_NAMES:
                        violations.append(
                            f"{fpath.relative_to(_BACKEND_DIR)}:{node.lineno} "
                            f"— banned import '{name}'"
                        )

    assert not violations, (
        "Legacy dispatch paths found in production code — Phase 4 cutover incomplete:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# Test 2 — INTERNAL=True excludes pattern_match abilities from the registry
# ---------------------------------------------------------------------------


def test_pattern_match_abilities_excluded_via_internal_flag():
    """SavePatternAbility / SaveGraphAbility are reachable via direct import
    but never appear in AbilityRegistry — the INTERNAL=True ClassVar is the
    exclusion mechanism, robust to test-import contamination.

    Why this matters: pattern_match abilities are processor-internal — only
    PatternMatchProcessor may invoke them, never the dispatcher / find_tools /
    UMP NATIVE_TOOLS list. Without INTERNAL, any test (or production code) that
    imports these classes would put them into ``Ability.__subclasses__()``,
    and they would surface as dispatchable abilities the next time the registry
    rebuilt.
    """
    # Force pattern_match abilities into Ability.__subclasses__().
    from abilities.pattern_match.save_pattern import SavePatternAbility
    from abilities.pattern_match.save_graph import SaveGraphAbility

    assert SavePatternAbility.INTERNAL is True, (
        "SavePatternAbility must declare INTERNAL=True to stay out of the registry"
    )
    assert SaveGraphAbility.INTERNAL is True, (
        "SaveGraphAbility must declare INTERNAL=True to stay out of the registry"
    )

    # Rebuild the registry from scratch with both classes already in
    # Ability.__subclasses__(). The INTERNAL filter must keep them out.
    _reg_module._reset_for_tests()
    try:
        names = {a.NAME for a in _reg_module.AbilityRegistry.all()}
        assert "save_pattern" not in names, (
            "save_pattern surfaced in AbilityRegistry.all() — INTERNAL filter is broken"
        )
        assert "save_graph" not in names, (
            "save_graph surfaced in AbilityRegistry.all() — INTERNAL filter is broken"
        )

        with pytest.raises(KeyError):
            _reg_module.AbilityRegistry.get("save_pattern")
        with pytest.raises(KeyError):
            _reg_module.AbilityRegistry.get("save_graph")
    finally:
        # Leave the registry rebuilt so other tests aren't disturbed.
        _reg_module._reset_for_tests()

    # Defense-in-depth: pattern_match must remain a subdirectory so a top-level
    # glob does not pick the modules up if INTERNAL is ever forgotten.
    abilities_dir = Path(_reg_module.__file__).resolve().parent
    shallow_py_files = {p.name for p in abilities_dir.glob("*.py")}
    assert "save_pattern.py" not in shallow_py_files
    assert "save_graph.py" not in shallow_py_files
    assert (abilities_dir / "pattern_match").is_dir()


# ---------------------------------------------------------------------------
# Test 3 — abilities.sqlite excludes pattern_match
# ---------------------------------------------------------------------------


def test_abilities_sqlite_excludes_pattern_match():
    """abilities.sqlite contains no row for save_pattern or save_graph.

    The SQLite ability index is the discovery layer for find_tools. If a
    regenerated DB accidentally includes pattern_match entries, those names
    would surface in tool discovery even though the registry excludes them —
    this test catches that inconsistency.
    """
    db_path = _BACKEND_DIR / "abilities" / "assets" / "abilities.sqlite"
    assert db_path.exists(), f"abilities.sqlite not found at {db_path}"

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM abilities WHERE name IN ('save_pattern', 'save_graph')"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [], (
        f"abilities.sqlite contains pattern_match entries that must not exist: "
        f"{[r[0] for r in rows]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — abilities/ disk layout has exactly 15 non-underscore top-level modules
# ---------------------------------------------------------------------------

_EXPECTED_ABILITY_MODULE_STEMS = frozenset({
    "browser",
    "code_eval",
    "document",
    "find_tools",
    "goal_pursuit",
    "list",
    "memory",
    "news",
    "programming_docs_search",
    "read",
    "review_tool_calls",
    "rich_render",
    "schedule",
    "search",
    "weather",
})


def test_abilities_directory_has_exactly_15_non_underscore_modules():
    """abilities/ contains exactly the 15 Phase 4 top-level modules — no more, no less.

    This mirrors what AbilityRegistry._load() walks: a shallow glob("*.py")
    over abilities/, skipping files that start with "_". The test asserts the
    disk layout, not the runtime registry state, so it is immune to test-
    session contamination from files that import pattern_match abilities.

    If a new ability is intentionally added, update _EXPECTED_ABILITY_MODULE_STEMS
    in this file at the same time.
    """
    abilities_dir = Path(_reg_module.__file__).resolve().parent

    walked = {
        p.stem
        for p in abilities_dir.glob("*.py")
        if not p.name.startswith("_")
    }

    added = walked - _EXPECTED_ABILITY_MODULE_STEMS
    removed = _EXPECTED_ABILITY_MODULE_STEMS - walked

    assert not added, (
        f"Unexpected .py files appeared at abilities/ top level: {sorted(added)}. "
        "Add them to _EXPECTED_ABILITY_MODULE_STEMS if intentional."
    )
    assert not removed, (
        f"Expected ability modules are missing from abilities/: {sorted(removed)}. "
        "Remove them from _EXPECTED_ABILITY_MODULE_STEMS if intentional."
    )
    assert len(walked) == 15
