"""
Folder Watcher Service — CRUD and scanning for watched filesystem directories.

Monitors user-selected folders for new, modified, renamed, and deleted files.
Automatically processes changes through the document pipeline.

Design notes:
- Files are referenced in-place (absolute paths), never copied to DOCUMENTS_ROOT.
- Watched folder documents auto-confirm (source_type='watched_folder' skips awaiting_confirmation).
- Missing-file tolerance: files must be absent for MISSING_THRESHOLD consecutive scans before soft-delete.
- Ingestion rate limiter: max MAX_ENQUEUE_PER_SCAN new documents per scan cycle.
- Environment tags derived from folder label + subfolder structure.
"""

import fnmatch
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import time
from datetime import datetime, timezone
from services.time_utils import utc_now, parse_utc
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Scan limits
MAX_ENQUEUE_PER_SCAN = 50
MISSING_THRESHOLD = 3
MIN_SCAN_INTERVAL = 60

# Allowed extensions (matches api/documents.py)
ALLOWED_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.html', '.htm', '.txt', '.md',
    '.css', '.csv', '.xml', '.json',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.rb',
    '.jpg', '.jpeg', '.png', '.webp', '.gif',
}


class FolderWatcherService:
    """Manages watched folder CRUD, directory browsing, and file scanning."""

    def __init__(self, db_service):
        self.db = db_service

    # ─────────────────────────────────────────────
    # CRUD
    # ─────────────────────────────────────────────

    def create_folder(
        self,
        folder_path: str,
        label: str = None,
        file_patterns: list = None,
        ignore_patterns: list = None,
        recursive: bool = True,
        scan_interval: int = 300,
        source_type: str = 'filesystem',
        source_config: dict = None,
    ) -> Dict[str, Any]:
        """Create a new watched folder. Validates path exists and is readable."""
        real_path = os.path.realpath(folder_path)
        self._validate_folder_path(real_path)

        folder_id = secrets.token_hex(4)
        scan_interval = max(scan_interval, MIN_SCAN_INTERVAL)

        default_ignores = [".git", "node_modules", "__pycache__", "build", "dist", ".DS_Store", "Thumbs.db"]

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO watched_folders
                    (id, folder_path, label, source_type, enabled,
                     file_patterns, ignore_patterns, recursive,
                     scan_interval, source_config, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """, (
                folder_id, real_path, label or os.path.basename(real_path),
                source_type,
                json.dumps(file_patterns or ["*"]),
                json.dumps(ignore_patterns if ignore_patterns is not None else default_ignores),
                1 if recursive else 0,
                scan_interval,
                json.dumps(source_config or {}),
            ))
            cursor.close()

        logger.info(f"[WATCHER] Created watched folder '{real_path}' (id={folder_id})")
        return self.get_folder(folder_id)

    def get_folder(self, folder_id: str) -> Optional[Dict[str, Any]]:
        """Get a single watched folder by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watched_folders WHERE id = ?", (folder_id,))
            row = cursor.fetchone()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            cursor.close()
        if not row:
            return None
        return self._row_to_dict(row, cols)

    def get_all_folders(self) -> List[Dict[str, Any]]:
        """Get all watched folders."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watched_folders ORDER BY created_at DESC")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            cursor.close()
        return [self._row_to_dict(row, cols) for row in rows]

    def get_enabled_folders(self) -> List[Dict[str, Any]]:
        """Get only enabled watched folders."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM watched_folders WHERE enabled = 1 ORDER BY created_at DESC")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            cursor.close()
        return [self._row_to_dict(row, cols) for row in rows]

    def update_folder(self, folder_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update mutable fields of a watched folder."""
        allowed_fields = {
            'folder_path', 'label', 'enabled', 'file_patterns', 'ignore_patterns',
            'recursive', 'scan_interval', 'source_config',
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_folder(folder_id)

        if 'folder_path' in updates:
            real_path = os.path.realpath(updates['folder_path'])
            self._validate_folder_path(real_path)
            updates['folder_path'] = real_path

        if 'scan_interval' in updates:
            updates['scan_interval'] = max(int(updates['scan_interval']), MIN_SCAN_INTERVAL)

        # JSON-encode list/dict fields
        for field in ('file_patterns', 'ignore_patterns', 'source_config'):
            if field in updates and isinstance(updates[field], (list, dict)):
                updates[field] = json.dumps(updates[field])

        if isinstance(updates.get('recursive'), bool):
            updates['recursive'] = 1 if updates['recursive'] else 0

        set_parts = [f"{k} = ?" for k in updates]
        set_parts.append("updated_at = datetime('now')")
        params = list(updates.values()) + [folder_id]

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE watched_folders SET {', '.join(set_parts)} WHERE id = ?",
                params,
            )
            cursor.close()

        return self.get_folder(folder_id)

    def delete_folder(self, folder_id: str, delete_documents: bool = False) -> bool:
        """Delete a watched folder. Optionally soft-delete associated documents."""
        if delete_documents:
            from services.document_service import DocumentService
            doc_svc = DocumentService(self.db)
            docs = doc_svc.get_documents_by_watched_folder(folder_id)
            for doc in docs:
                if not doc.get('deleted_at'):
                    doc_svc.soft_delete(doc['id'])

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM watched_folders WHERE id = ?", (folder_id,))
            deleted = cursor.rowcount > 0
            cursor.close()

        if deleted:
            # Clear scan state cache
            self._clear_scan_cache(folder_id)
            logger.info(f"[WATCHER] Deleted watched folder {folder_id}")
        return deleted

    def trigger_scan(self, folder_id: str) -> None:
        """Request an immediate scan for a folder."""
        from services.memory_store import MemoryStore
        store = MemoryStore()
        store.set(f"watcher:scan_now:{folder_id}", "1", ex=600)

    # ─────────────────────────────────────────────
    # Directory browsing
    # ─────────────────────────────────────────────

    def browse_directory(self, path: str = None) -> Dict[str, Any]:
        """List directories at the given path. Returns {current, parent, directories}."""
        if not path:
            path = os.path.expanduser("~")

        real_path = os.path.realpath(path)

        if not os.path.isdir(real_path):
            raise ValueError(f"Not a directory: {path}")
        if not os.access(real_path, os.R_OK):
            raise PermissionError(f"Cannot read directory: {path}")

        directories = []
        try:
            for entry in sorted(os.scandir(real_path), key=lambda e: e.name.lower()):
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                    try:
                        # Check readability
                        os.listdir(entry.path)
                        directories.append(entry.name)
                    except PermissionError:
                        pass
        except PermissionError:
            raise PermissionError(f"Cannot read directory: {path}")

        parent = os.path.dirname(real_path) if real_path != '/' else None

        return {
            'current': real_path,
            'parent': parent,
            'directories': directories,
        }

    # ─────────────────────────────────────────────
    # Scanning
    # ─────────────────────────────────────────────

    def is_scan_due(self, folder: Dict) -> bool:
        """Check if enough time has passed since the last scan."""
        last_scan = folder.get('last_scan_at')
        if not last_scan:
            return True
        try:
            last_dt = parse_utc(last_scan)
            elapsed = (utc_now() - last_dt).total_seconds()
            return elapsed >= folder.get('scan_interval', 300)
        except (ValueError, TypeError):
            return True

    def is_scan_requested(self, folder_id: str) -> bool:
        """Check if an immediate scan was requested via trigger_scan()."""
        from services.memory_store import MemoryStore
        store = MemoryStore()
        val = store.get(f"watcher:scan_now:{folder_id}")
        if val:
            store.delete(f"watcher:scan_now:{folder_id}")
            return True
        return False

    def scan_folder(self, folder: Dict) -> Dict[str, int]:
        """
        Scan a watched folder for changes. Returns summary dict.

        Algorithm:
        1. Walk folder, collect {path: mtime} for matching files
        2. Compare against existing documents in DB
        3. Detect: new, modified, renamed, deleted
        4. Enqueue processing for new/modified (capped at MAX_ENQUEUE_PER_SCAN)
        5. Soft-delete files missing for MISSING_THRESHOLD consecutive scans
        """
        from services.memory_store import MemoryStore
        store = MemoryStore()
        lock_key = f"watcher:scanning:{folder['id']}"

        # Skip if already scanning
        if store.get(lock_key):
            logger.debug(f"[WATCHER] Scan already in progress for {folder['id']}")
            return {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

        store.set(lock_key, "1", ex=3600)  # 1h max lock

        try:
            return self._do_scan(folder, store)
        except Exception as e:
            self._update_scan_error(folder['id'], str(e)[:500])
            raise
        finally:
            store.delete(lock_key)

    def _do_scan(self, folder: Dict, store) -> Dict[str, int]:
        """Internal scan implementation."""
        from services.document_service import DocumentService
        from services.document_queue import enqueue_document_processing

        folder_path = folder['folder_path']
        folder_id = folder['id']
        result = {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

        # Validate folder still exists
        if not os.path.isdir(folder_path):
            msg = f"Folder no longer accessible: {folder_path}"
            logger.warning(f"[WATCHER] {msg}")
            self._update_scan_error(folder_id, msg)
            return result

        # Parse patterns
        file_patterns = self._parse_json_list(folder.get('file_patterns', '["*"]'))
        ignore_patterns = self._parse_json_list(folder.get('ignore_patterns', '[]'))
        recursive = bool(folder.get('recursive', 1))

        # 1. Walk and collect discovered files
        discovered = {}  # {abs_path: mtime}
        for abs_path, mtime in self._walk_folder(folder_path, recursive, file_patterns, ignore_patterns):
            discovered[abs_path] = mtime

        # 2. Get existing documents for this folder
        doc_svc = DocumentService(self.db)
        existing_docs = doc_svc.get_documents_by_watched_folder(folder_id)

        # Build lookups (skip failed docs so they get re-processed)
        existing_by_path = {}  # {file_path: doc_dict}
        existing_by_hash = {}  # {file_hash: doc_dict} (for rename detection)
        failed_by_path = {}    # {file_path: doc_dict} (for cleanup before retry)
        for doc in existing_docs:
            if doc.get('deleted_at'):
                continue
            if doc.get('status') == 'failed':
                failed_by_path[doc['file_path']] = doc
                continue
            existing_by_path[doc['file_path']] = doc
            if doc.get('file_hash'):
                existing_by_hash[doc['file_hash']] = doc

        # Load scan state cache (for missing_count tracking)
        scan_cache = self._load_scan_cache(store, folder_id)

        enqueued = 0

        # 3. Check discovered files for new/modified
        for abs_path, mtime in discovered.items():
            try:
                cached = scan_cache.get(abs_path, {})
                existing = existing_by_path.get(abs_path)

                if existing:
                    # File exists in DB — check if modified
                    cached_mtime = cached.get('mtime')
                    if cached_mtime and abs(mtime - cached_mtime) < 1:
                        # mtime unchanged — skip
                        result['skipped'] += 1
                        # Clear any missing count
                        if 'missing_count' in cached:
                            del cached['missing_count']
                        scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
                        continue

                    # mtime changed — check hash
                    file_hash = self._compute_hash(abs_path)
                    if file_hash == existing.get('file_hash'):
                        # Content unchanged (touch only) — update cache, skip
                        result['skipped'] += 1
                        scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
                        continue

                    # Content changed — supersede
                    if enqueued < MAX_ENQUEUE_PER_SCAN:
                        new_doc_id = self._create_watched_document(
                            doc_svc, folder, abs_path, file_hash, mtime)
                        doc_svc.set_supersedes(new_doc_id, existing['id'])
                        doc_svc.soft_delete(existing['id'])
                        enqueue_document_processing(new_doc_id)
                        scan_cache[abs_path] = {'mtime': mtime, 'doc_id': new_doc_id}
                        result['updated'] += 1
                        enqueued += 1
                    else:
                        result['skipped'] += 1

                else:
                    # File not in DB (or previously failed) — new or renamed?
                    file_hash = self._compute_hash(abs_path)

                    # Check for rename (same hash, different path)
                    renamed_doc = existing_by_hash.get(file_hash)
                    if renamed_doc and renamed_doc['file_path'] not in discovered:
                        # Rename detected — update path, no reprocessing
                        doc_svc.update_file_path(renamed_doc['id'], abs_path)
                        old_path = renamed_doc['file_path']
                        scan_cache.pop(old_path, None)
                        scan_cache[abs_path] = {'mtime': mtime, 'doc_id': renamed_doc['id']}
                        result['renamed'] += 1
                        continue

                    # Clean up previously failed doc before retry
                    failed_doc = failed_by_path.get(abs_path)
                    if failed_doc:
                        doc_svc.soft_delete(failed_doc['id'])

                    # New file (or retry of previously failed)
                    if enqueued < MAX_ENQUEUE_PER_SCAN:
                        new_doc_id = self._create_watched_document(
                            doc_svc, folder, abs_path, file_hash, mtime)
                        enqueue_document_processing(new_doc_id)
                        scan_cache[abs_path] = {'mtime': mtime, 'doc_id': new_doc_id}
                        result['new'] += 1
                        enqueued += 1
                    else:
                        result['skipped'] += 1

            except Exception as e:
                logger.warning(f"[WATCHER] Error processing {abs_path}: {e}")
                result['errors'].append(f"{os.path.basename(abs_path)}: {e}")

        # 4. Check for deleted files (in DB but not on disk)
        for abs_path, doc in existing_by_path.items():
            if abs_path not in discovered:
                cached = scan_cache.get(abs_path, {})
                missing_count = cached.get('missing_count', 0) + 1

                if missing_count >= MISSING_THRESHOLD:
                    # File confirmed missing — soft-delete
                    doc_svc.soft_delete(doc['id'])
                    scan_cache.pop(abs_path, None)
                    result['deleted'] += 1
                    logger.info(f"[WATCHER] Soft-deleted missing file: {os.path.basename(abs_path)}")
                else:
                    # Tolerate temporary absence
                    cached['missing_count'] = missing_count
                    scan_cache[abs_path] = cached

        # 5. Save scan state
        self._save_scan_cache(store, folder_id, scan_cache)
        self._update_scan_stats(folder_id, len(discovered))

        return result

    # ─────────────────────────────────────────────
    # Scan helpers
    # ─────────────────────────────────────────────

    def _walk_folder(self, folder_path, recursive, file_patterns, ignore_patterns):
        """Yield (abs_path, mtime) for matching files in the folder."""
        real_root = os.path.realpath(folder_path)

        if recursive:
            walker = os.walk(folder_path, followlinks=False)
        else:
            # Non-recursive: just the top directory
            walker = [(folder_path, [], [e.name for e in os.scandir(folder_path) if e.is_file()])]

        for dirpath, dirnames, filenames in walker:
            # Filter out ignored directories (in-place for os.walk pruning)
            dirnames[:] = [
                d for d in dirnames
                if not any(fnmatch.fnmatch(d, pat) for pat in ignore_patterns)
                and not d.startswith('.')
            ]

            for filename in filenames:
                # Check ignore patterns
                if any(fnmatch.fnmatch(filename, pat) for pat in ignore_patterns):
                    continue

                # Check file patterns
                if not any(fnmatch.fnmatch(filename, pat) for pat in file_patterns):
                    continue

                # Check extension
                ext = os.path.splitext(filename)[1].lower()
                if ext and ext not in ALLOWED_EXTENSIONS:
                    continue

                abs_path = os.path.join(dirpath, filename)

                # Symlink safety: skip if target is outside watched folder
                real_file = os.path.realpath(abs_path)
                if not real_file.startswith(real_root):
                    continue

                try:
                    stat = os.stat(abs_path)
                    yield abs_path, stat.st_mtime
                except (PermissionError, OSError) as e:
                    logger.debug(f"[WATCHER] Cannot stat {abs_path}: {e}")

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _create_watched_document(self, doc_svc, folder, abs_path, file_hash, mtime):
        """Create a document record for a watched folder file."""
        original_name = os.path.basename(abs_path)
        mime_type = mimetypes.guess_type(abs_path)[0] or 'application/octet-stream'
        file_size = os.path.getsize(abs_path)

        doc_id = doc_svc.create_document(
            original_name=original_name,
            mime_type=mime_type,
            file_size=file_size,
            file_path=abs_path,
            file_hash=file_hash,
            source_type='watched_folder',
            watched_folder_id=folder['id'],
        )

        # Derive and set environment tags
        tags = self._derive_environment_tags(folder, abs_path)
        if tags:
            doc_svc.update_tags(doc_id, tags)

        return doc_id

    def _derive_environment_tags(self, folder: Dict, abs_path: str) -> list:
        """Derive semantic environment tags from folder structure."""
        tags = []

        # Folder label is the primary environment
        if folder.get('label'):
            tags.append(folder['label'])

        # Relative subfolder segments become secondary tags
        rel_path = os.path.relpath(os.path.dirname(abs_path), folder['folder_path'])
        if rel_path != '.':
            segments = [s for s in rel_path.split(os.sep) if s and not s.startswith('.')]
            tags.extend(segments)

        return tags

    # ─────────────────────────────────────────────
    # Scan state cache (MemoryStore)
    # ─────────────────────────────────────────────

    def _load_scan_cache(self, store, folder_id: str) -> dict:
        """Load per-folder scan state from MemoryStore. Rebuild from DB on cold start."""
        cache_key = f"watcher:state:{folder_id}"
        raw = store.get(cache_key)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        # Cold start: rebuild from DB
        from services.document_service import DocumentService
        doc_svc = DocumentService(self.db)
        docs = doc_svc.get_documents_by_watched_folder(folder_id)
        cache = {}
        for doc in docs:
            if not doc.get('deleted_at') and doc.get('file_path'):
                cache[doc['file_path']] = {'doc_id': doc['id']}
        return cache

    def _save_scan_cache(self, store, folder_id: str, cache: dict) -> None:
        """Save scan state to MemoryStore. TTL = 48h (survives missed scans)."""
        cache_key = f"watcher:state:{folder_id}"
        store.set(cache_key, json.dumps(cache), ex=172800)

    def _clear_scan_cache(self, folder_id: str) -> None:
        """Clear scan state cache for a folder."""
        from services.memory_store import MemoryStore
        store = MemoryStore()
        store.delete(f"watcher:state:{folder_id}")
        store.delete(f"watcher:scan_now:{folder_id}")

    # ─────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────

    def _update_scan_stats(self, folder_id: str, file_count: int) -> None:
        """Update last_scan_at and last_scan_files for a folder."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE watched_folders
                SET last_scan_at = datetime('now'), last_scan_files = ?,
                    last_scan_error = NULL, updated_at = datetime('now')
                WHERE id = ?
            """, (file_count, folder_id))
            cursor.close()

    def _update_scan_error(self, folder_id: str, error: str) -> None:
        """Record a scan error."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE watched_folders
                SET last_scan_error = ?, updated_at = datetime('now')
                WHERE id = ?
            """, (error, folder_id))
            cursor.close()

    def _validate_folder_path(self, real_path: str) -> None:
        """Validate a folder path is safe to watch."""
        if not os.path.isdir(real_path):
            raise ValueError(f"Path is not a directory: {real_path}")
        if not os.access(real_path, os.R_OK):
            raise PermissionError(f"Directory is not readable: {real_path}")

    def _parse_json_list(self, val) -> list:
        """Parse a JSON string or return list as-is."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    def _row_to_dict(self, row, cols) -> Dict[str, Any]:
        """Convert a database row to dict using column names."""
        d = dict(zip(cols, row))
        # Parse JSON fields
        for field in ('file_patterns', 'ignore_patterns', 'source_config'):
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
