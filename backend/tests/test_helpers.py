"""
Shared test helpers — utilities used across multiple test files.
"""

import re
from pathlib import Path


_SCHEMA_PATH = Path(__file__).parent.parent / "schema.sql"

# Regex strips CREATE VIRTUAL TABLE ... USING vec0|fts5 (...); statements
# that require SQLite extensions not available in the test environment.
_EXT_TABLE_RE = re.compile(
    r'CREATE\s+VIRTUAL\s+TABLE\s+[^;]+USING\s+(?:vec0|fts5)\s*\([^)]*\)\s*;',
    re.IGNORECASE | re.DOTALL,
)


def load_schema_sql() -> str:
    """Load schema.sql stripped of extension-dependent virtual tables (vec0, fts5).

    Use this instead of reading schema.sql directly when setting up in-memory
    test databases so tests work without sqlite-vec or FTS5 extensions.
    """
    return _EXT_TABLE_RE.sub('', _SCHEMA_PATH.read_text())
