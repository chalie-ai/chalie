"""Declarative SQLite schema management — converges live DB to match schema.sql
by adding missing objects and dropping stale ones."""

import logging
import os
import re
import sqlite3
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_RE_SQL_COMMENTS = r"--[^\n]*"

# Episodic-memory redesign backfill constants.
_SUPERSEDED_BY_EDGE = "superseded_by"  # data_graph_edges.edge_type linking old → superseding fact

# Tables whose names we never touch even if they appear stale.  SQLite system
# tables, FTS5/vec0 shadow tables, and our own bookkeeping live here.  Shadow
# tables are usually filtered out by ``_introspect_tables`` via virtual-table
# prefix matching, but this is a defence-in-depth guard for orphans.
_PROTECTED_TABLE_PREFIXES = ("sqlite_",)
_PROTECTED_TABLE_NAMES = frozenset({"sqlite_sequence"})

# Shadow-table suffixes used by FTS5 and sqlite-vec — defence in depth so an
# orphaned shadow (e.g. left over from a manual drop) is never blindly dropped
# as a stale table.
_SHADOW_SUFFIXES = (
    "_data", "_idx", "_docsize", "_config", "_content",  # FTS5
    "_chunks", "_rowids", "_info",                       # sqlite-vec
)


def _destructive_allowed() -> bool:
    """Return True unless the operator has explicitly opted out via env var."""
    return os.environ.get("CHALIE_SCHEMA_ALLOW_DESTRUCTIVE", "1").lower() not in (
        "0", "false", "no", "off",
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

    def __init__(self, db_service, embedding_dimensions: int = 768):
        self.db_service = db_service
        self._embedding_dimensions = embedding_dimensions
        self._schema_path = FileMapperService.get_schema_path()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def converge(self) -> None:
        """Single entry point — idempotent, safe to call on every startup."""
        if not self._schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self._schema_path}")

        schema_sql = self._schema_path.read_text()

        desired_conn = self._load_desired_state(schema_sql)
        try:
            desired_tables = self._introspect_tables(desired_conn)
            desired_indexes = self._introspect_indexes(desired_conn)
            desired_virtual = self._introspect_virtual_tables(desired_conn)
            desired_triggers = self._introspect_triggers(desired_conn)
        finally:
            desired_conn.close()

        with self.db_service.connection() as conn:
            _load_sqlite_vec(conn)

            is_fresh = self._is_fresh_db(conn)

            live_tables = self._introspect_tables(conn)
            live_indexes = self._introspect_indexes(conn)
            live_virtual = self._introspect_virtual_tables(conn)
            live_triggers = self._introspect_triggers(conn)

            # Additive pass first — ensures new shape is in place before any
            # destructive change runs.  If a destructive op fails we still end
            # up with a usable schema.
            tables_created, columns_added = self._converge_tables(
                desired_tables, live_tables, conn, schema_sql
            )
            indexes_synced = self._converge_indexes(desired_indexes, live_indexes, conn)
            virtual_tables_created = self._converge_virtual_tables(
                desired_virtual, live_virtual, conn, schema_sql
            )
            triggers_synced = self._converge_triggers(desired_triggers, live_triggers, conn, schema_sql)

            # Destructive pass (gated).  Safety: if the desired state looks
            # suspiciously empty or truncated, refuse to drop anything — the
            # live DB is more trustworthy than a corrupt schema.sql.
            destructive_safe = self._destructive_safety_check(
                desired_tables, live_tables
            )
            if _destructive_allowed() and destructive_safe:
                # Order: virtual tables first (cascades shadow tables) → indexes
                # (some auto-drop with their owning table) → columns within
                # surviving tables → stale regular tables last.
                virtual_tables_dropped = self._drop_stale_virtual_tables(
                    desired_virtual, live_virtual, conn
                )
                indexes_dropped = self._drop_stale_indexes(
                    desired_indexes, live_indexes, conn
                )
                columns_dropped = self._drop_stale_columns(
                    desired_tables, live_tables, conn
                )
                tables_dropped = self._drop_stale_tables(
                    desired_tables, live_tables, live_virtual, conn
                )
                triggers_dropped = self._drop_stale_triggers(desired_triggers, live_triggers, conn)
            else:
                virtual_tables_dropped = indexes_dropped = columns_dropped = tables_dropped = 0
                triggers_dropped = 0
                self._log_stale(desired_tables, live_tables, desired_indexes, live_indexes,
                                desired_virtual, live_virtual)

            self._strip_data_graph_check_constraint(conn)

            if is_fresh:
                self._run_seed_data(schema_sql, conn)

        logger.info(
            f"Schema converged: "
            f"+{tables_created} tables / -{tables_dropped}, "
            f"+{columns_added} columns / -{columns_dropped}, "
            f"+{indexes_synced} indexes / -{indexes_dropped}, "
            f"+{virtual_tables_created} virtual tables / -{virtual_tables_dropped}, "
            f"+{triggers_synced} triggers / -{triggers_dropped}"
        )

    def backfill_redesign_columns(self) -> None:
        with self.db_service.connection() as conn:
            self._backfill_episode_columns(conn)
            self._backfill_data_graph_columns(conn)
            conn.commit()
        logger.info("[convergence] Redesign-column backfill complete")

    def _backfill_episode_columns(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE episodes SET last_relevant_at = COALESCE(last_accessed_at, created_at) "
            "WHERE last_relevant_at IS NULL"
        )

    def _backfill_data_graph_columns(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE data_graph SET valid_from = first_seen_at WHERE valid_from IS NULL"
        )
        conn.execute(
            "UPDATE data_graph AS old SET valid_to = ("
            "    SELECT new.first_seen_at FROM data_graph_edges AS e "
            "    JOIN data_graph AS new ON new.id = e.to_id "
            # Earliest superseder = the moment the fact stopped being true.
            # The supersession writer demotes a row to active=0 and never
            # re-matches it, so at most one superseded_by edge exists per row
            # today — ASC makes the choice explicit if that ever changes.
            f"    WHERE e.from_id = old.id AND e.edge_type = '{_SUPERSEDED_BY_EDGE}' "
            "    ORDER BY new.first_seen_at ASC LIMIT 1"
            ") "
            "WHERE old.valid_to IS NULL AND old.active = 0 AND EXISTS ("
            "    SELECT 1 FROM data_graph_edges AS e2 "
            f"    WHERE e2.from_id = old.id AND e2.edge_type = '{_SUPERSEDED_BY_EDGE}'"
            ")"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Desired state
    # ──────────────────────────────────────────────────────────────────────────

    def _load_desired_state(self, schema_sql: str) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        _load_sqlite_vec(conn)
        # Replace hardcoded vec0 dimension with configured value
        sql = schema_sql
        if self._embedding_dimensions != 768:
            sql = sql.replace("float[768]", f"float[{self._embedding_dimensions}]")
        # Strip single-line comments, then split respecting BEGIN...END blocks.
        sql_no_comments = re.sub(_RE_SQL_COMMENTS, "",sql)
        for stmt in self._split_statements(sql_no_comments):
            try:
                conn.execute(stmt)
            except Exception as exc:
                logger.debug(f"[convergence] Skipping desired-state statement: {exc}")
        return conn

    def _split_statements(self, sql: str) -> list:
        statements = []
        current: list = []
        depth = 0

        for token in re.split(r"(\bBEGIN\b|\bEND\b|;)", sql, flags=re.IGNORECASE):
            upper = token.strip().upper()
            if upper == "BEGIN":
                depth += 1
                current.append(token)
            elif upper == "END":
                depth = max(0, depth - 1)
                current.append(token)
            elif token == ";":
                if depth == 0:
                    stmt = "".join(current).strip()
                    if stmt:
                        statements.append(stmt)
                    current = []
                else:
                    current.append(token)
            else:
                current.append(token)

        # Handle any trailing content without a final semicolon
        remainder = "".join(current).strip()
        if remainder:
            statements.append(remainder)

        return statements

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection helpers
    # ──────────────────────────────────────────────────────────────────────────

    def column_set(self, conn: sqlite3.Connection) -> dict:
        return {
            table: set(cols) for table, cols in self._introspect_tables(conn).items()
        }

    def _introspect_tables(self, conn: sqlite3.Connection) -> dict:
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

    def _introspect_triggers(self, conn: sqlite3.Connection) -> dict:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL"
        ).fetchall()
        return {name: self._normalize_ddl(ddl) for name, ddl in rows}

    def _normalize_ddl(self, ddl) -> str:
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

            # Table exists — add any missing columns.
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

            # Log type/constraint mismatches — SQLite cannot ALTER column types
            # in place; rebuilding the table is destructive enough that we
            # require an explicit migration helper rather than auto-fix.
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

    # ──────────────────────────────────────────────────────────────────────────
    # Destructive convergence — drop what is no longer declared in schema.sql
    # ──────────────────────────────────────────────────────────────────────────

    def _drop_stale_tables(
        self,
        desired: dict,
        actual: dict,
        live_virtual: dict,
        live_conn: sqlite3.Connection,
    ) -> int:
        dropped = 0
        virtual_names = set(live_virtual.keys())
        shadow_prefixes = tuple(f"{vn}_" for vn in virtual_names)

        for table_name in actual:
            if table_name in desired:
                continue
            if not self._is_droppable_table(table_name, virtual_names, shadow_prefixes):
                continue
            try:
                live_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.warning(f"[convergence] DROPPED table: {table_name}")
                dropped += 1
            except Exception as exc:
                logger.error(f"[convergence] Failed to drop stale table {table_name}: {exc}")
        return dropped

    def _drop_stale_columns(
        self,
        desired: dict,
        actual: dict,
        live_conn: sqlite3.Connection,
    ) -> int:
        dropped = 0
        for table_name, desired_cols in desired.items():
            if table_name not in actual:
                continue
            live_cols = actual[table_name]
            for col_name in live_cols:
                if col_name in desired_cols:
                    continue
                try:
                    live_conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {col_name}")
                    logger.warning(f"[convergence] DROPPED column: {table_name}.{col_name}")
                    dropped += 1
                except Exception as exc:
                    logger.error(
                        f"[convergence] Failed to drop stale column {table_name}.{col_name}: {exc}"
                    )
        return dropped

    def _drop_stale_indexes(
        self,
        desired: dict,
        actual: dict,
        live_conn: sqlite3.Connection,
    ) -> int:
        dropped = 0
        for idx_name in actual:
            if idx_name in desired:
                continue
            try:
                live_conn.execute(f"DROP INDEX IF EXISTS {idx_name}")
                logger.warning(f"[convergence] DROPPED index: {idx_name}")
                dropped += 1
            except Exception as exc:
                logger.error(f"[convergence] Failed to drop stale index {idx_name}: {exc}")
        return dropped

    def _drop_stale_virtual_tables(
        self,
        desired: dict,
        actual: dict,
        live_conn: sqlite3.Connection,
    ) -> int:
        dropped = 0
        for table_name in actual:
            if table_name in desired:
                continue
            try:
                live_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.warning(f"[convergence] DROPPED virtual table: {table_name}")
                dropped += 1
            except Exception as exc:
                logger.error(
                    f"[convergence] Failed to drop stale virtual table {table_name}: {exc}"
                )
        return dropped

    def _destructive_safety_check(self, desired_tables: dict, live_tables: dict) -> bool:
        if not desired_tables:
            logger.error(
                "[convergence] SAFETY: desired schema has zero tables — "
                "refusing destructive ops.  Check schema.sql integrity."
            )
            return False
        if len(live_tables) >= 10 and len(desired_tables) < len(live_tables) / 2:
            logger.error(
                f"[convergence] SAFETY: desired={len(desired_tables)} tables vs "
                f"live={len(live_tables)} — too large a shrink, refusing destructive "
                f"ops.  Check schema.sql integrity."
            )
            return False
        return True

    def _is_droppable_table(
        self,
        name: str,
        virtual_names: set,
        shadow_prefixes: tuple,
    ) -> bool:
        if name in _PROTECTED_TABLE_NAMES:
            return False
        if name.startswith(_PROTECTED_TABLE_PREFIXES):
            return False
        if name in virtual_names:
            return False
        # If this name looks like a shadow table for an existing virtual table,
        # leave it alone — it will be cleaned up when the virtual table is.
        if shadow_prefixes and name.startswith(shadow_prefixes):
            return False
        # Extra defence in depth: bare suffix match for orphaned shadow names
        # (e.g. "foo_data" with no virtual "foo"). Prefer to leak rather than
        # silently corrupt a virtual table.
        for suffix in _SHADOW_SUFFIXES:
            if name.endswith(suffix):
                return False
        return True

    def _log_stale(
        self,
        desired_tables: dict,
        live_tables: dict,
        desired_indexes: dict,
        live_indexes: dict,
        desired_virtual: dict,
        live_virtual: dict,
    ) -> None:
        for t in live_tables.keys() - desired_tables.keys():
            logger.warning(f"[convergence] STALE table (would drop): {t}")
        for t, cols in live_tables.items():
            if t not in desired_tables:
                continue
            for c in cols.keys() - desired_tables[t].keys():
                logger.warning(f"[convergence] STALE column (would drop): {t}.{c}")
        for i in live_indexes.keys() - desired_indexes.keys():
            logger.warning(f"[convergence] STALE index (would drop): {i}")
        for v in live_virtual.keys() - desired_virtual.keys():
            logger.warning(f"[convergence] STALE virtual table (would drop): {v}")

    def _converge_indexes(self, desired: dict, actual: dict, live_conn: sqlite3.Connection) -> int:
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

    def _extract_fts5_columns(self, normalized_ddl: str) -> list:
        """Parse the column list from a normalized FTS5 CREATE VIRTUAL TABLE DDL.

        Strips key=value options (e.g. content='episodes', content_rowid='rowid')
        and returns only bare column names.

        Returns an empty list if the DDL is not an FTS5 table or cannot be parsed.
        """
        if "fts5" not in normalized_ddl:
            return []
        # Extract the argument list between the outer parentheses after USING fts5(
        match = re.search(r"using\s+fts5\s*\((.+)\)", normalized_ddl)
        if not match:
            return []
        args_str = match.group(1)
        # Split by comma, then discard entries that contain '=' (key=value options)
        columns = []
        for part in args_str.split(","):
            part = part.strip().strip("'\"")
            if "=" not in part and part:
                columns.append(part)
        return columns

    def _converge_virtual_tables(self, desired: dict, actual: dict, live_conn: sqlite3.Connection, schema_sql: str) -> int:
        """Create missing virtual tables; rebuild FTS5 tables whose column list changed."""
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
                # For FTS5 content tables, check if only the column list changed.
                # If so, rebuild automatically (DROP + CREATE + rebuild).
                live_ddl = actual[table_name]
                if "fts5" in desired_ddl and "content=" in desired_ddl:
                    desired_cols = self._extract_fts5_columns(desired_ddl)
                    live_cols = self._extract_fts5_columns(live_ddl)
                    if desired_cols != live_cols:
                        logger.warning(
                            f"[convergence] FTS5 column list changed for {table_name}: "
                            f"{live_cols} -> {desired_cols}. Rebuilding."
                        )
                        raw_ddl = self._extract_virtual_table_ddl(schema_sql, table_name)
                        if raw_ddl:
                            try:
                                live_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                                live_conn.execute(raw_ddl)
                                live_conn.execute(
                                    f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                                )
                                logger.info(f"[convergence] Rebuilt FTS5 table (column change): {table_name}")
                                created += 1
                            except Exception as exc:
                                logger.error(
                                    f"[convergence] Failed to rebuild FTS5 table {table_name}: {exc}"
                                )
                        continue
                # Non-FTS5 or non-column-list change — log only
                logger.warning(
                    f"[convergence] Virtual table DDL mismatch (not auto-fixed): {table_name}"
                )

        return created

    def _converge_triggers(
        self,
        desired: dict,
        actual: dict,
        live_conn: sqlite3.Connection,
        schema_sql: str,
    ) -> int:
        """Create missing triggers; drop and recreate triggers whose DDL has changed.

        Mirrors ``_converge_indexes``: compare normalized DDL, CREATE if absent,
        DROP+CREATE if the body changed.
        """
        synced = 0

        for trigger_name, desired_ddl in desired.items():
            if trigger_name not in actual:
                raw_ddl = self._extract_trigger_ddl(schema_sql, trigger_name)
                if not raw_ddl:
                    logger.warning(
                        f"[convergence] No DDL found for missing trigger: {trigger_name}"
                    )
                    continue
                try:
                    live_conn.execute(raw_ddl)
                    logger.info(f"[convergence] Created trigger: {trigger_name}")
                    synced += 1
                except Exception as exc:
                    logger.error(f"[convergence] Failed to create trigger {trigger_name}: {exc}")
            elif actual[trigger_name] != desired_ddl:
                raw_ddl = self._extract_trigger_ddl(schema_sql, trigger_name)
                if not raw_ddl:
                    logger.warning(
                        f"[convergence] No DDL found for changed trigger: {trigger_name}"
                    )
                    continue
                try:
                    live_conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                    live_conn.execute(raw_ddl)
                    logger.info(f"[convergence] Recreated trigger (DDL changed): {trigger_name}")
                    synced += 1
                except Exception as exc:
                    logger.error(f"[convergence] Failed to recreate trigger {trigger_name}: {exc}")

        return synced

    def _drop_stale_triggers(
        self,
        desired: dict,
        actual: dict,
        live_conn: sqlite3.Connection,
    ) -> int:
        dropped = 0
        for trigger_name in actual:
            if trigger_name in desired:
                continue
            try:
                live_conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                logger.warning(f"[convergence] DROPPED trigger: {trigger_name}")
                dropped += 1
            except Exception as exc:
                logger.error(f"[convergence] Failed to drop stale trigger {trigger_name}: {exc}")
        return dropped

    def _create_virtual_table(self, conn: sqlite3.Connection, table_name: str, ddl: str) -> bool:
        """Handle orphaned vec0 shadow tables that block re-creation."""
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
    # Seed / one-shot helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_data_graph_check_constraint(self, live_conn: sqlite3.Connection) -> None:
        """Strip CHECK constraint from data_graph.kind on existing databases.

        Python validates kind via VALID_KINDS in data_graph_service.py.
        To be removed when SchemaConvergence handles constraint changes fully.

        The embedded ``data_graph_new`` DDL below must stay in lockstep with the
        ``data_graph`` table in schema.sql. The copy matches columns BY NAME
        (shared columns only), so a legacy table that predates newer columns
        (e.g. valid_from/valid_to) survives the rebuild: missing columns are
        NULL-filled and populated later by ``backfill_redesign_columns()``.
        """
        row = live_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_graph'"
        ).fetchone()
        if not row or not row[0] or 'CHECK' not in row[0].upper():
            return

        live_conn.execute("PRAGMA foreign_keys=OFF")
        try:
            live_conn.execute("""
                CREATE TABLE data_graph_new (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind              TEXT NOT NULL,
                    key               TEXT NOT NULL,
                    value             TEXT,
                    storage_strength  REAL NOT NULL DEFAULT 0.5,
                    retrieval_weight  REAL NOT NULL DEFAULT 1.0,
                    salience_score    REAL NOT NULL DEFAULT 0.0,
                    evidence_count    INTEGER NOT NULL DEFAULT 1,
                    first_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
                    last_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_accessed_at  TEXT,
                    source            TEXT,
                    deleted_at        TEXT,
                    active            INTEGER NOT NULL DEFAULT 1,
                    search_queries    TEXT DEFAULT NULL,
                    valid_from        TEXT,
                    valid_to          TEXT
                )
            """)
            # Copy by explicit shared column names — positional SELECT * corrupts
            # the copy whenever the legacy table's column count drifts from the
            # DDL above (e.g. a pre-redesign table without valid_from/valid_to).
            live_cols = [r[1] for r in live_conn.execute("PRAGMA table_info(data_graph)")]
            new_cols = {r[1] for r in live_conn.execute("PRAGMA table_info(data_graph_new)")}
            shared = ", ".join(c for c in live_cols if c in new_cols)
            live_conn.execute(
                f"INSERT INTO data_graph_new ({shared}) SELECT {shared} FROM data_graph"
            )
            live_conn.execute("DROP TABLE data_graph")
            live_conn.execute("ALTER TABLE data_graph_new RENAME TO data_graph")
            live_conn.execute("INSERT INTO data_graph_fts(data_graph_fts) VALUES('rebuild')")
            # Recreate indexes destroyed when the old table was dropped.
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS idx_data_graph_kind      ON data_graph(kind)",
                "CREATE INDEX IF NOT EXISTS idx_data_graph_key       ON data_graph(key)",
                "CREATE INDEX IF NOT EXISTS idx_data_graph_retrieval ON data_graph(retrieval_weight DESC)",
                "CREATE INDEX IF NOT EXISTS idx_data_graph_active    ON data_graph(kind, active) WHERE deleted_at IS NULL",
                "CREATE INDEX IF NOT EXISTS idx_data_graph_confirmed ON data_graph(last_confirmed_at)",
                "CREATE INDEX IF NOT EXISTS idx_data_graph_live      ON data_graph(kind) WHERE active = 1 AND valid_to IS NULL AND deleted_at IS NULL",
            ]:
                live_conn.execute(idx_sql)
            logger.info("[convergence] Stripped CHECK constraint from data_graph.kind")
        except Exception as exc:
            logger.error(f"[convergence] Failed to strip data_graph CHECK constraint: {exc}")
            raise
        finally:
            live_conn.execute("PRAGMA foreign_keys=ON")

    def _run_seed_data(self, schema_sql: str, live_conn: sqlite3.Connection) -> None:
        stripped = re.sub(_RE_SQL_COMMENTS, "",schema_sql)
        for match in re.finditer(
            r"(INSERT\s+OR\s+IGNORE\s+INTO\s+\w+[^;]+;)", stripped, re.IGNORECASE | re.DOTALL
        ):
            stmt = match.group(1).strip()
            try:
                live_conn.execute(stmt)
            except Exception as exc:
                logger.warning(f"[convergence] Seed insert failed: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # DDL extraction from schema.sql
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_table_ddl(self, schema_sql: str, table_name: str) -> str | None:
        """Extract the CREATE TABLE ... ; block for a normal table from schema.sql."""
        # Strip single-line SQL comments first — they can contain semicolons
        # (e.g. "-- dropped by migration 026; kept here") which break [^;]+.
        stripped = re.sub(_RE_SQL_COMMENTS, "",schema_sql)
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
        clean_sql = re.sub(_RE_SQL_COMMENTS, "",schema_sql)
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
            ddl = ddl.replace("float[768]", f"float[{self._embedding_dimensions}]")
        return ddl

    def _extract_trigger_ddl(self, schema_sql: str, trigger_name: str) -> str | None:
        """Extract the CREATE TRIGGER ... END ; block for a trigger from schema.sql.

        SQLite trigger bodies contain semicolons inside the BEGIN...END block, so
        we cannot rely on the simple split-on-semicolon approach used elsewhere.
        Instead we match from CREATE TRIGGER <name> through the END keyword that
        closes the outermost block, then consume the trailing semicolon.
        """
        stripped = re.sub(_RE_SQL_COMMENTS, "",schema_sql)
        pattern = re.compile(
            r"(CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?" + re.escape(trigger_name) + r"\b.+?END\s*;)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(stripped)
        return match.group(1).strip() if match else None

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
