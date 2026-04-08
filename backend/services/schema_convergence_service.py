"""
Schema Convergence Service — declarative SQLite schema management.

Compares the desired state (schema.sql executed into :memory:) against the
live database and applies the minimum set of changes needed to bring the live
schema up to date.  Replaces SchemaService.initialize_schema(),
DatabaseService.run_pending_migrations(), and SchemaService._create_vec_tables().
"""

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Idempotent error substrings: these mean the statement was already applied.
_IDEMPOTENT_ERRORS = (
    "no such column",
    "no such table",
    "duplicate column name",
    "no such module",
    "already exists",
    "sql logic error",
)


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into a connection. No-op if unavailable."""
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception as exc:
        logger.debug(f"[convergence] sqlite-vec not available: {exc}")


class SchemaConvergenceService:
    """
    SilverStripe-inspired declarative schema convergence for SQLite.

    Single entry point: ``converge()``.  Idempotent — safe to call on every
    startup.
    """

    def __init__(self, db_service, embedding_dimensions: int = 768):
        self.db_service = db_service
        self._embedding_dimensions = embedding_dimensions
        self._schema_path = Path(__file__).resolve().parent.parent / "schema.sql"
        self._migrations_dir = Path(__file__).resolve().parent.parent / "migrations"

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def converge(self) -> None:
        """Single entry point — replaces initialize_schema + run_pending_migrations + ensure_vec_tables."""
        if not self._schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self._schema_path}")

        schema_sql = self._schema_path.read_text()

        desired_conn = self._load_desired_state(schema_sql)
        try:
            desired_tables = self._introspect_tables(desired_conn)
            desired_indexes = self._introspect_indexes(desired_conn)
            desired_virtual = self._introspect_virtual_tables(desired_conn)
        finally:
            desired_conn.close()

        tables_created = 0
        columns_added = 0
        indexes_synced = 0
        virtual_tables_created = 0

        with self.db_service.connection() as conn:
            _load_sqlite_vec(conn)

            is_fresh = self._is_fresh_db(conn)

            live_tables = self._introspect_tables(conn)
            live_indexes = self._introspect_indexes(conn)
            live_virtual = self._introspect_virtual_tables(conn)

            tc, ca = self._converge_tables(desired_tables, live_tables, conn, schema_sql)
            tables_created += tc
            columns_added += ca

            indexes_synced += self._converge_indexes(desired_indexes, live_indexes, conn)
            virtual_tables_created += self._converge_virtual_tables(desired_virtual, live_virtual, conn, schema_sql)

            self._run_drop_statements(schema_sql, conn)

            if is_fresh:
                self._run_seed_data(schema_sql, conn)
                self._stamp_migrations(conn)
            else:
                self._run_pending_migrations(conn)

        logger.info(
            f"Schema converged: {tables_created} tables created, "
            f"{columns_added} columns added, "
            f"{indexes_synced} indexes synced, "
            f"{virtual_tables_created} virtual tables created"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Desired state
    # ──────────────────────────────────────────────────────────────────────────

    def _load_desired_state(self, schema_sql: str) -> sqlite3.Connection:
        """Execute schema.sql into :memory: and return the open connection.

        Uses statement-by-statement execution instead of executescript() so that
        a single failing statement (e.g. vec0 when sqlite-vec is unavailable)
        does not prevent all subsequent tables from being created in the desired
        state.
        """
        conn = sqlite3.connect(":memory:")
        _load_sqlite_vec(conn)
        # Replace hardcoded vec0 dimension with configured value
        sql = schema_sql
        if self._embedding_dimensions != 768:
            sql = sql.replace("float[768]", f"float[{self._embedding_dimensions}]")
        # Strip comments, then split on semicolons and execute one at a time.
        sql_no_comments = re.sub(r"--[^\n]*", "", sql)
        statements = [s.strip() for s in sql_no_comments.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(stmt)
            except Exception as exc:
                logger.debug(f"[convergence] Skipping desired-state statement: {exc}")
        return conn

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _introspect_tables(self, conn: sqlite3.Connection) -> dict:
        """Return {table_name: {col_name: (cid, name, type, notnull, dflt_value, pk)}}
        for all non-virtual, non-shadow, non-system tables."""
        virtual_names = set(self._introspect_virtual_tables(conn).keys())
        # Build shadow table prefixes (FTS5/vec0 create shadow tables like
        # episodes_fts_data, episodes_vec_info, etc.)
        shadow_prefixes = tuple(f"{vn}_" for vn in virtual_names)

        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        result = {}
        for (name,) in rows:
            if name in virtual_names:
                continue
            if shadow_prefixes and name.startswith(shadow_prefixes):
                continue
            cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
            result[name] = {row[1]: row for row in cols}
        return result

    def _introspect_indexes(self, conn: sqlite3.Connection) -> dict:
        """Return {index_name: normalized_ddl} for all user-defined indexes."""
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ).fetchall()
        return {name: self._normalize_ddl(ddl) for name, ddl in rows}

    def _introspect_virtual_tables(self, conn: sqlite3.Connection) -> dict:
        """Return {table_name: normalized_ddl} for all virtual tables (FTS5, vec0)."""
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        result = {}
        for name, ddl in rows:
            if ddl and ddl.strip().upper().startswith("CREATE VIRTUAL TABLE"):
                result[name] = self._normalize_ddl(ddl)
        return result

    def _normalize_ddl(self, ddl: str) -> str:
        """Lowercase, collapse whitespace, strip IF NOT EXISTS for comparison."""
        if not ddl:
            return ""
        normalized = ddl.lower()
        normalized = re.sub(r"\bif\s+not\s+exists\s+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    # ──────────────────────────────────────────────────────────────────────────
    # Convergence steps
    # ──────────────────────────────────────────────────────────────────────────

    def _converge_tables(self, desired: dict, actual: dict, live_conn: sqlite3.Connection, schema_sql: str):
        """Create missing tables; add missing columns to existing tables."""
        tables_created = 0
        columns_added = 0

        for table_name, desired_cols in desired.items():
            if table_name not in actual:
                ddl = self._extract_table_ddl(schema_sql, table_name)
                if ddl:
                    try:
                        live_conn.execute(ddl)
                        logger.info(f"[convergence] Created table: {table_name}")
                        tables_created += 1
                    except Exception as exc:
                        logger.error(f"[convergence] Failed to create table {table_name}: {exc}")
                else:
                    logger.warning(f"[convergence] No DDL found for missing table: {table_name}")
                continue

            # Table exists — check columns
            live_cols = actual[table_name]
            for col_name, col_info in desired_cols.items():
                if col_name not in live_cols:
                    # col_info tuple: (cid, name, type, notnull, dflt_value, pk)
                    col_type = col_info[2] if col_info[2] else "TEXT"
                    dflt_value = col_info[4]
                    notnull = col_info[3]

                    col_def = col_type
                    if dflt_value is not None:
                        col_def += f" DEFAULT {dflt_value}"
                        if notnull:
                            col_def += " NOT NULL"
                    # Don't add NOT NULL without a default — SQLite rejects it for ADD COLUMN

                    try:
                        live_conn.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"
                        )
                        logger.info(f"[convergence] Added column: {table_name}.{col_name}")
                        columns_added += 1
                    except Exception as exc:
                        logger.error(
                            f"[convergence] Failed to add column {table_name}.{col_name}: {exc}"
                        )

            # Log columns in live but not desired (candidates for removal — do not drop)
            # TODO v0.5.0: auto-drop stale columns
            for col_name in live_cols:
                if col_name not in desired_cols:
                    logger.debug(
                        f"[convergence] Stale column (not in desired schema): "
                        f"{table_name}.{col_name}"
                    )

            # Log type/constraint mismatches — do not auto-fix
            # TODO v0.5.0: auto-alter columns with type/constraint mismatches
            for col_name, desired_info in desired_cols.items():
                if col_name in live_cols:
                    live_info = live_cols[col_name]
                    desired_type = (desired_info[2] or "").lower()
                    live_type = (live_info[2] or "").lower()
                    if desired_type != live_type:
                        logger.debug(
                            f"[convergence] Type mismatch {table_name}.{col_name}: "
                            f"desired={desired_type!r}, live={live_type!r} (not auto-fixed)"
                        )

        return tables_created, columns_added

    def _converge_indexes(self, desired: dict, actual: dict, live_conn: sqlite3.Connection) -> int:
        """Create missing indexes; drop and recreate indexes whose DDL has changed."""
        synced = 0

        for idx_name, desired_ddl in desired.items():
            if idx_name not in actual:
                # Reconstruct CREATE INDEX with IF NOT EXISTS
                raw_ddl = self._restore_if_not_exists(desired_ddl, "index")
                try:
                    live_conn.execute(raw_ddl)
                    logger.info(f"[convergence] Created index: {idx_name}")
                    synced += 1
                except Exception as exc:
                    logger.error(f"[convergence] Failed to create index {idx_name}: {exc}")
            elif actual[idx_name] != desired_ddl:
                # DDL changed — drop and recreate
                try:
                    live_conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
                    raw_ddl = self._restore_if_not_exists(desired_ddl, "index")
                    live_conn.execute(raw_ddl)
                    logger.info(f"[convergence] Recreated index (DDL changed): {idx_name}")
                    synced += 1
                except Exception as exc:
                    logger.error(f"[convergence] Failed to recreate index {idx_name}: {exc}")

        return synced

    def _converge_virtual_tables(self, desired: dict, actual: dict, live_conn: sqlite3.Connection, schema_sql: str) -> int:
        """Create missing virtual tables; warn on DDL mismatch (no auto-rebuild)."""
        created = 0

        for table_name, desired_ddl in desired.items():
            if table_name not in actual:
                raw_ddl = self._extract_virtual_table_ddl(schema_sql, table_name)
                if not raw_ddl:
                    logger.warning(
                        f"[convergence] No DDL found for missing virtual table: {table_name}"
                    )
                    continue

                ok = self._create_virtual_table(live_conn, table_name, raw_ddl)
                if ok:
                    created += 1
                    # For FTS5 content tables: rebuild to populate from the content table
                    if "fts5" in desired_ddl and "content=" in desired_ddl:
                        try:
                            live_conn.execute(
                                f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                            )
                            logger.info(f"[convergence] Rebuilt FTS5 index: {table_name}")
                        except Exception as exc:
                            logger.warning(
                                f"[convergence] FTS5 rebuild failed for {table_name}: {exc}"
                            )
            elif actual[table_name] != desired_ddl:
                # FTS5/vec0 recreation is destructive — log only
                logger.warning(
                    f"[convergence] Virtual table DDL mismatch (not auto-fixed): {table_name}"
                )

        return created

    def _create_virtual_table(self, conn: sqlite3.Connection, table_name: str, ddl: str) -> bool:
        """Attempt to CREATE VIRTUAL TABLE; handle orphaned shadow tables for vec0."""
        try:
            conn.execute(ddl)
            logger.info(f"[convergence] Created virtual table: {table_name}")
            return True
        except Exception as exc:
            err = str(exc).lower()
            # Orphaned shadow tables from a prior ALTER TABLE RENAME leave behind
            # rows in sqlite_master that block re-creation.
            if "already exists" in err:
                logger.warning(
                    f"[convergence] Virtual table {table_name} creation failed, "
                    f"attempting shadow table cleanup: {exc}"
                )
                try:
                    conn.execute("PRAGMA writable_schema=ON")
                    shadow_rows = conn.execute(
                        "SELECT name FROM sqlite_master WHERE name LIKE ?",
                        (f"{table_name}_%",),
                    ).fetchall()
                    for (shadow_name,) in shadow_rows:
                        conn.execute(
                            "DELETE FROM sqlite_master WHERE name = ?", (shadow_name,)
                        )
                    conn.execute("PRAGMA writable_schema=OFF")
                    # Retry (VACUUM skipped — cannot run inside a transaction)
                    conn.execute(ddl)
                    logger.info(
                        f"[convergence] Created virtual table after cleanup: {table_name}"
                    )
                    return True
                except Exception as retry_exc:
                    logger.error(
                        f"[convergence] Failed to create virtual table {table_name} "
                        f"after cleanup: {retry_exc}"
                    )
                    return False
            else:
                logger.error(
                    f"[convergence] Failed to create virtual table {table_name}: {exc}"
                )
                return False

    # ──────────────────────────────────────────────────────────────────────────
    # DROP / seed / migration helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_drop_statements(self, schema_sql: str, live_conn: sqlite3.Connection) -> None:
        """Execute DROP TABLE IF EXISTS statements found in schema.sql."""
        for match in re.finditer(
            r"DROP\s+TABLE\s+IF\s+EXISTS\s+(\w+)\s*;", schema_sql, re.IGNORECASE
        ):
            table_name = match.group(1)
            try:
                live_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.debug(f"[convergence] Executed drop: {table_name}")
            except Exception as exc:
                logger.warning(f"[convergence] DROP TABLE {table_name} failed: {exc}")

    def _run_seed_data(self, schema_sql: str, live_conn: sqlite3.Connection) -> None:
        """Execute INSERT OR IGNORE seed statements on a fresh database."""
        stripped = re.sub(r"--[^\n]*", "", schema_sql)
        for match in re.finditer(
            r"(INSERT\s+OR\s+IGNORE\s+INTO\s+\w+[^;]+;)", stripped, re.IGNORECASE | re.DOTALL
        ):
            stmt = match.group(1).strip()
            try:
                live_conn.execute(stmt)
            except Exception as exc:
                logger.warning(f"[convergence] Seed insert failed: {exc}")

    def _stamp_migrations(self, live_conn: sqlite3.Connection) -> None:
        """On a fresh DB, mark all *.sql migration files as already applied."""
        if not self._migrations_dir.exists():
            return
        for migration_file in sorted(self._migrations_dir.glob("*.sql")):
            try:
                live_conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (filename) VALUES (?)",
                    (migration_file.name,),
                )
            except Exception as exc:
                logger.warning(f"[convergence] Could not stamp migration {migration_file.name}: {exc}")
        logger.info("[convergence] Fresh install — stamped all migrations as applied")

    def _run_pending_migrations(self, live_conn: sqlite3.Connection) -> None:
        """Apply unapplied *.sql migration files from the migrations directory."""
        if not self._migrations_dir.exists():
            return

        applied_rows = live_conn.execute(
            "SELECT filename FROM schema_migrations"
        ).fetchall()
        applied = {row[0] for row in applied_rows}

        pending_count = 0
        for migration_file in sorted(self._migrations_dir.glob("*.sql")):
            filename = migration_file.name
            if filename in applied:
                continue

            logger.info(f"[convergence] Applying migration: {filename}")
            sql = migration_file.read_text()

            # Execute statement-by-statement with idempotent error handling.
            # Never use executescript() for migrations — it auto-commits and can
            # produce FK violations.
            sql_no_comments = re.sub(r"--[^\n]*", "", sql)
            statements = [s.strip() for s in sql_no_comments.split(";") if s.strip()]

            for stmt in statements:
                try:
                    live_conn.execute(stmt)
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if any(pat in err_msg for pat in _IDEMPOTENT_ERRORS):
                        logger.debug(
                            f"[convergence] Skipping already-applied step in {filename}: {exc}"
                        )
                    else:
                        raise

            live_conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (?)", (filename,)
            )
            pending_count += 1
            logger.info(f"[convergence] Migration applied: {filename}")

        if pending_count == 0:
            logger.info("[convergence] No pending migrations")
        else:
            logger.info(f"[convergence] Applied {pending_count} migration(s)")

    # ──────────────────────────────────────────────────────────────────────────
    # DDL extraction from schema.sql
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_table_ddl(self, schema_sql: str, table_name: str) -> str | None:
        """Extract the CREATE TABLE ... ; block for a normal table from schema.sql."""
        # Strip single-line SQL comments first — they can contain semicolons
        # (e.g. "-- dropped by migration 026; kept here") which break [^;]+.
        stripped = re.sub(r"--[^\n]*", "", schema_sql)
        pattern = re.compile(
            r"(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?" + re.escape(table_name) + r"\s*\([^;]+;)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(stripped)
        return match.group(1).strip() if match else None

    def _extract_virtual_table_ddl(self, schema_sql: str, table_name: str) -> str | None:
        """Extract the CREATE VIRTUAL TABLE ... ; block for a virtual table from schema.sql.

        For vec0 tables, replaces the hardcoded dimension with the configured
        ``embedding_dimensions`` (defaults to 768; tests use 256).
        """
        # Strip SQL comments before regex matching (same as _extract_table_ddl)
        clean_sql = re.sub(r"--[^\n]*", "", schema_sql)
        pattern = re.compile(
            r"(CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?" + re.escape(table_name) + r"[^;]+;)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(clean_sql)
        if not match:
            return None
        ddl = match.group(1).strip()
        # Replace hardcoded vec0 dimension with configured value
        if self._embedding_dimensions != 768 and "vec0" in ddl.lower():
            ddl = re.sub(r"float\[768\]", f"float[{self._embedding_dimensions}]", ddl)
        return ddl

    def _restore_if_not_exists(self, normalized_ddl: str, obj_type: str) -> str:
        """Re-insert IF NOT EXISTS into a normalized DDL string."""
        # Handle UNIQUE indexes: "create unique index foo" → "create unique index if not exists foo"
        if obj_type == "index":
            for kw in ("unique index", "index"):
                prefix = f"create {kw} "
                if normalized_ddl.startswith(prefix):
                    return f"create {kw} if not exists " + normalized_ddl[len(prefix):]
            return normalized_ddl
        prefix = f"create {obj_type} "
        if normalized_ddl.startswith(prefix):
            return f"create {obj_type} if not exists " + normalized_ddl[len(prefix):]
        return normalized_ddl

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def _is_fresh_db(self, conn: sqlite3.Connection) -> bool:
        """Return True if the live database has no user tables yet."""
        row = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        return (row[0] if row else 0) == 0
