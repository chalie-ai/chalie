"""
Folder Watcher Service — CRUD and scanning for watched filesystem directories.

Monitors user-selected folders for new, modified, renamed, and deleted files.
Automatically processes changes through the document pipeline.

Design notes:
- Files are referenced in-place (absolute paths), never copied to the documents root.
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
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, cast

from services.log_utils import safe
from services.time_utils import utc_now, parse_utc

if TYPE_CHECKING:
    from services.document_service import DocumentService
    from services.memory_store import MemoryStore

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

    def __init__(self) -> None:
        """Initialize the service."""

    # CRUD
    # ─────────────────────────────────────────────

    def create_folder(
        self,
        folder_path: str,
        label: Optional[str] = None,
        file_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        recursive: bool = True,
        scan_interval: int = 60,
        source_type: str = 'filesystem',
        source_config: Optional[dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Create and persist a new watched folder record."""
        real_path = os.path.realpath(folder_path)
        self._validate_folder_path(real_path)

        scan_interval = max(scan_interval, MIN_SCAN_INTERVAL)

        default_ignores = [".git", "node_modules", "__pycache__", "build", "dist", ".DS_Store", "Thumbs.db"]

        from models.watched_folder import WatchedFolder

        folder = WatchedFolder(
            folder_path=real_path,
            label=label or os.path.basename(real_path),
            source_type=source_type,
            enabled=1,
            file_patterns=json.dumps(file_patterns or ["*"]),
            ignore_patterns=json.dumps(ignore_patterns if ignore_patterns is not None else default_ignores),
            recursive=1 if recursive else 0,
            scan_interval=scan_interval,
            source_config=json.dumps(source_config or {}),
            created_at=utc_now().isoformat(),
            updated_at=utc_now().isoformat(),
        )
        folder.save()
        folder_id = folder.id

        logger.info(f"[WATCHER] Created watched folder '{real_path}' (id={folder_id})")
        return cast(Dict[str, object], folder.to_dict())

    def get_folder(self, folder_id: str) -> Optional[Dict[str, object]]:
        """Retrieve a single watched folder record by its ID."""
        from models.watched_folder import WatchedFolder
        folder = WatchedFolder.get(folder_id)
        return folder.to_dict() if folder else None

    def get_all_folders(self) -> List[Dict[str, object]]:
        """Retrieve all watched folder records, newest first."""
        from models.watched_folder import WatchedFolder
        return [folder.to_dict() for folder in WatchedFolder.all_ordered()]

    def get_enabled_folders(self) -> List[Dict[str, object]]:
        """Retrieve only enabled watched folder records."""
        from models.watched_folder import WatchedFolder
        return [folder.to_dict() for folder in WatchedFolder.enabled_ordered()]

    def update_folder(self, folder_id: str, **kwargs: object) -> Optional[Dict[str, object]]:
        """Update mutable fields of a watched folder."""
        allowed_fields = {
            'folder_path', 'label', 'enabled', 'file_patterns', 'ignore_patterns',
            'recursive', 'scan_interval', 'source_config',
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return self.get_folder(folder_id)

        if 'folder_path' in updates:
            real_path = os.path.realpath(cast(str, updates['folder_path']))
            self._validate_folder_path(real_path)
            updates['folder_path'] = real_path

        if 'scan_interval' in updates:
            updates['scan_interval'] = max(int(cast(int, updates['scan_interval'])), MIN_SCAN_INTERVAL)

        # JSON-encode list/dict fields
        for field in ('file_patterns', 'ignore_patterns', 'source_config'):
            if field in updates and isinstance(updates[field], (list, dict)):
                updates[field] = json.dumps(updates[field])

        if isinstance(updates.get('recursive'), bool):
            updates['recursive'] = 1 if updates['recursive'] else 0

        from models.watched_folder import WatchedFolder
        WatchedFolder.update_fields(folder_id, updates)

        return self.get_folder(folder_id)

    def delete_folder(self, folder_id: str, delete_documents: bool = False) -> bool:
        """Delete a watched folder record."""
        if delete_documents:
            from services.document_service import DocumentService
            doc_svc = DocumentService()
            docs = doc_svc.get_documents_by_watched_folder(folder_id)
            for doc in docs:
                if not doc.get('deleted_at'):
                    doc_svc.soft_delete(cast(str, doc['id']))

        from models.watched_folder import WatchedFolder
        folder = WatchedFolder.get(folder_id)
        deleted = False
        if folder:
            folder.delete()
            deleted = True

        if deleted:
            # Clear scan state cache
            self._clear_scan_cache(folder_id)
            logger.info("[WATCHER] Deleted watched folder %s", safe(folder_id))
        return deleted

    def trigger_scan(self, folder_id: str) -> None:
        """Request an out-of-schedule immediate scan for a watched folder."""
        from services.memory_client import MemoryClientService
        MemoryClientService.create_connection().set(f"watcher:scan_now:{folder_id}", "1", ex=600)

    # ─────────────────────────────────────────────
    # Directory browsing
    # ─────────────────────────────────────────────

    def browse_directory(self, path: Optional[str] = None) -> Dict[str, object]:
        """List readable sub-directories at the given filesystem path."""
        if not path:
            path = os.path.expanduser("~")

        if "\x00" in path:
            raise ValueError("Path contains null bytes")
        if not os.path.isabs(path):
            raise ValueError(f"Path must be absolute: {path}")
        path = os.path.normpath(path)
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

    # Scanning
    # ─────────────────────────────────────────────

    def is_scan_due(self, folder: Dict[str, object]) -> bool:
        """Check if enough time has passed since the last scan."""
        last_scan = folder.get('last_scan_at')
        if not last_scan:
            return True
        try:
            last_dt = parse_utc(cast(str, last_scan))
            elapsed = (utc_now() - last_dt).total_seconds()
            return elapsed >= cast(float, folder.get('scan_interval', 300))
        except (ValueError, TypeError):
            return True

    def is_scan_requested(self, folder_id: str) -> bool:
        """Check if an immediate scan was requested via trigger_scan()."""
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        val = store.get(f"watcher:scan_now:{folder_id}")
        if val:
            store.delete(f"watcher:scan_now:{folder_id}")
            return True
        return False

    def scan_folder(self, folder: Dict[str, object]) -> Dict[str, object]:
        """Scan a watched folder for changes."""
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        lock_key = f"watcher:scanning:{folder['id']}"

        # Skip if already scanning
        if store.get(lock_key):
            logger.debug(f"[WATCHER] Scan already in progress for {folder['id']}")
            return {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

        store.set(lock_key, "1", ex=3600)  # 1h max lock

        try:
            return self._do_scan(folder, store)
        except Exception as e:
            self._update_scan_error(cast(str, folder['id']), str(e)[:500])
            raise
        finally:
            store.delete(lock_key)

    def _categorize_existing_file(self, doc_svc: "DocumentService", folder: Dict[str, object], abs_path: str, mtime: float,
                                   existing: Dict[str, object], cached: Dict[str, object], scan_cache: Dict[str, Dict[str, object]],
                                   result: Dict[str, object], enqueued: int) -> int:
        """Handle a discovered file that already has a document record."""
        cached_mtime = cached.get('mtime')

        if existing.get('status') == 'failed':
            if cached_mtime is None or abs(mtime - cast(float, cached_mtime)) < 1:
                scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
                result['skipped'] = cast(int, result['skipped']) + 1
                return enqueued

        elif existing.get('status') in ('pending', 'processing'):
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
            result['skipped'] = cast(int, result['skipped']) + 1
            return enqueued

        elif cached_mtime and abs(mtime - cast(float, cached_mtime)) < 1:
            result['skipped'] = cast(int, result['skipped']) + 1
            cached.pop('missing_count', None)
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
            return enqueued

        file_hash = self._compute_hash(abs_path)
        if file_hash == existing.get('file_hash'):
            result['skipped'] = cast(int, result['skipped']) + 1
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': existing['id']}
            return enqueued

        if enqueued < MAX_ENQUEUE_PER_SCAN:
            new_doc_id = self._create_watched_document(doc_svc, folder, abs_path, file_hash)
            doc_svc.set_supersedes(new_doc_id, cast(str, existing['id']))
            try:
                from models.document import DocumentRow
                DocumentRow.purge_by_document_id(cast(str, existing['id']))
            except Exception as exc:
                logger.warning("[WATCHER] Failed to cascade-delete old artifacts for %s: %s", existing['id'], exc)
            doc_svc.soft_delete(cast(str, existing['id']))
            self._process_watched_document(new_doc_id, abs_path)
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': new_doc_id}
            result['updated'] = cast(int, result['updated']) + 1
            enqueued += 1
        else:
            result['skipped'] = cast(int, result['skipped']) + 1
        return enqueued

    def _categorize_new_file(self, doc_svc: "DocumentService", folder: Dict[str, object], abs_path: str, mtime: float,
                              discovered: Dict[str, float], existing_by_hash: Dict[str, Dict[str, object]],
                              scan_cache: Dict[str, Dict[str, object]], result: Dict[str, object], enqueued: int) -> int:
        """Handle a discovered file with no existing document record."""
        file_hash = self._compute_hash(abs_path)

        renamed_doc = existing_by_hash.get(file_hash)
        if renamed_doc and renamed_doc['file_path'] not in discovered:
            doc_svc.update_file_path(cast(str, renamed_doc['id']), abs_path)
            old_path = renamed_doc['file_path']
            scan_cache.pop(cast(str, old_path), None)
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': renamed_doc['id']}
            result['renamed'] = cast(int, result['renamed']) + 1
            return enqueued

        if enqueued < MAX_ENQUEUE_PER_SCAN:
            new_doc_id = self._create_watched_document(doc_svc, folder, abs_path, file_hash)
            self._process_watched_document(new_doc_id, abs_path)
            scan_cache[abs_path] = {'mtime': mtime, 'doc_id': new_doc_id}
            result['new'] = cast(int, result['new']) + 1
            enqueued += 1
        else:
            result['skipped'] = cast(int, result['skipped']) + 1
        return enqueued

    def _do_scan(self, folder: Dict[str, object], store: "MemoryStore") -> Dict[str, object]:
        """Execute the folder scan against the database."""
        from services.document_service import DocumentService

        folder_path = cast(str, folder['folder_path'])
        folder_id = cast(str, folder['id'])
        result: Dict[str, object] = {'new': 0, 'updated': 0, 'deleted': 0, 'renamed': 0, 'skipped': 0, 'errors': []}

        if not os.path.isdir(folder_path):
            msg = f"Folder no longer accessible: {folder_path}"
            logger.warning(f"[WATCHER] {msg}")
            self._update_scan_error(folder_id, msg)
            return result

        file_patterns = self._parse_json_list(folder.get('file_patterns', '["*"]'))
        ignore_patterns = self._parse_json_list(folder.get('ignore_patterns', '[]'))
        recursive = bool(folder.get('recursive', 1))

        discovered: Dict[str, float] = {}
        for abs_path, mtime in self._walk_folder(folder_path, recursive, file_patterns, ignore_patterns):
            discovered[abs_path] = mtime

        doc_svc = DocumentService()
        existing_docs = doc_svc.get_documents_by_watched_folder(folder_id)

        existing_by_path: Dict[str, Dict[str, object]] = {}
        existing_by_hash: Dict[str, Dict[str, object]] = {}
        for doc in existing_docs:
            if doc.get('deleted_at'):
                continue
            existing_by_path[cast(str, doc['file_path'])] = doc
            if doc.get('file_hash'):
                existing_by_hash[cast(str, doc['file_hash'])] = doc

        scan_cache = self._load_scan_cache(store, folder_id)
        enqueued = 0

        for abs_path, mtime in discovered.items():
            try:
                cached = scan_cache.get(abs_path, {})
                existing = existing_by_path.get(abs_path)

                if existing:
                    enqueued = self._categorize_existing_file(
                        doc_svc, folder, abs_path, mtime, existing,
                        cached, scan_cache, result, enqueued,
                    )
                else:
                    enqueued = self._categorize_new_file(
                        doc_svc, folder, abs_path, mtime, discovered,
                        existing_by_hash, scan_cache, result, enqueued,
                    )
            except Exception as e:
                logger.warning(f"[WATCHER] Error processing {abs_path}: {e}")
                cast(list[str], result['errors']).append(f"{os.path.basename(abs_path)}: {e}")

        # Check for deleted files
        for abs_path, doc in existing_by_path.items():
            if abs_path not in discovered:
                cached = scan_cache.get(abs_path, {})
                missing_count = cast(int, cached.get('missing_count', 0)) + 1
                if missing_count >= MISSING_THRESHOLD:
                    doc_svc.soft_delete(cast(str, doc['id']))
                    scan_cache.pop(abs_path, None)
                    result['deleted'] = cast(int, result['deleted']) + 1
                    logger.info(f"[WATCHER] Soft-deleted missing file: {os.path.basename(abs_path)}")
                else:
                    cached['missing_count'] = missing_count
                    scan_cache[abs_path] = cached

        self._save_scan_cache(store, folder_id, scan_cache)
        self._update_scan_stats(folder_id, len(discovered))

        return result

    # ─────────────────────────────────────────────
    # Scan helpers
    # ─────────────────────────────────────────────

    def _walk_folder(self, folder_path: str, recursive: bool, file_patterns: list[str], ignore_patterns: list[str]) -> Iterator[tuple[str, float]]:
        """Yield matching files from a folder tree as (abs_path, mtime) tuples."""
        real_root = os.path.realpath(folder_path)

        walker: Iterable[tuple[str, list[str], list[str]]]
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
                if not real_file.startswith(real_root + os.sep):
                    continue

                try:
                    stat = os.stat(abs_path)
                    yield abs_path, stat.st_mtime
                except OSError as e:
                    logger.debug(f"[WATCHER] Cannot stat {abs_path}: {e}")

    def _compute_hash(self, file_path: str) -> str:
        """Compute the SHA-256 content hash of a file."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def _create_watched_document(self, doc_svc: "DocumentService", folder: Dict[str, object], abs_path: str, file_hash: str) -> str:
        """Create a document record for a file discovered in a watched folder."""
        original_name = os.path.basename(abs_path)
        mime_type = mimetypes.guess_type(abs_path)[0] or 'application/octet-stream'
        file_size = os.path.getsize(abs_path)

        doc_id: str = doc_svc.create_document(
            original_name=original_name,
            mime_type=mime_type,
            file_size=file_size,
            file_path=abs_path,
            file_hash=file_hash,
            source_type='watched_folder',
            watched_folder_id=cast(str, folder['id']),
        )

        # Derive and set environment tags
        tags = self._derive_environment_tags(folder, abs_path)
        if tags:
            doc_svc.update_tags(doc_id, tags)

        return doc_id

    def _process_watched_document(self, doc_id: str, abs_path: str) -> None:
        """Extract text from a watched file and create data_graph artifacts."""
        import threading

        def _run() -> None:
            try:
                from services.text_extractor import extract_text
                from services.document_chunking import create_document_artifacts
                from services.document_service import DocumentService

                text = extract_text(abs_path)
                if not text:
                    DocumentService().update_status(
                        doc_id, 'failed', 'Text extraction empty'
                    )
                    return

                artifact_count = create_document_artifacts(doc_id, text)
                svc = DocumentService()

                summary = text[:500]
                dot_pos = summary.rfind('. ')
                if dot_pos > 200:
                    summary = summary[:dot_pos + 1]
                svc.update_summary(doc_id, summary)
                svc.update_status(doc_id, 'ready', chunk_count=artifact_count)
            except Exception as e:
                logger.error(f"[FOLDER WATCHER] Failed to process {doc_id}: {e}")
                try:
                    from services.document_service import DocumentService
                    DocumentService().update_status(
                        doc_id, 'failed', str(e)[:500]
                    )
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True, name=f"doc-watch-{doc_id[:8]}").start()

    def _derive_environment_tags(self, folder: Dict[str, object], abs_path: str) -> list[str]:
        """Derive semantic environment tags from the folder label and subfolder path."""
        tags = []

        # Folder label is the primary environment
        if folder.get('label'):
            tags.append(cast(str, folder['label']))

        # Relative subfolder segments become secondary tags
        rel_path = os.path.relpath(os.path.dirname(abs_path), cast(str, folder['folder_path']))
        if rel_path != '.':
            segments = [s for s in rel_path.split(os.sep) if s and not s.startswith('.')]
            tags.extend(segments)

        return tags

    # ─────────────────────────────────────────────
    # Scan state cache (MemoryStore)
    # ─────────────────────────────────────────────

    def _load_scan_cache(self, store: "MemoryStore", folder_id: str) -> Dict[str, Dict[str, object]]:
        """Load the per-folder scan state from MemoryStore."""
        cache_key = f"watcher:state:{folder_id}"
        raw = store.get(cache_key)
        if raw:
            try:
                return cast(Dict[str, Dict[str, object]], json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

        # Cold start: rebuild from DB
        from services.document_service import DocumentService
        doc_svc = DocumentService()
        docs = doc_svc.get_documents_by_watched_folder(folder_id)
        cache: Dict[str, Dict[str, object]] = {}
        for doc in docs:
            if not doc.get('deleted_at') and doc.get('file_path'):
                cache[cast(str, doc['file_path'])] = {'doc_id': doc['id']}
        return cache

    def _save_scan_cache(self, store: "MemoryStore", folder_id: str, cache: Dict[str, Dict[str, object]]) -> None:
        """Persist the per-folder scan state to MemoryStore with a 48-hour TTL."""
        cache_key = f"watcher:state:{folder_id}"
        store.set(cache_key, json.dumps(cache), ex=172800)

    def _clear_scan_cache(self, folder_id: str) -> None:
        """Clear the MemoryStore scan-state cache for a folder."""
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        store.delete(f"watcher:state:{folder_id}")
        store.delete(f"watcher:scan_now:{folder_id}")

    # ─────────────────────────────────────────────
    # DB helpers
    # ─────────────────────────────────────────────

    def _update_scan_stats(self, folder_id: str, file_count: int) -> None:
        """Persist scan completion statistics to the database."""
        from models.watched_folder import WatchedFolder
        WatchedFolder.update_scan_stats(folder_id, file_count)

    def _update_scan_error(self, folder_id: str, error: str) -> None:
        """Record a scan error message in the database."""
        from models.watched_folder import WatchedFolder
        WatchedFolder.update_scan_error(folder_id, error)

    def _validate_folder_path(self, real_path: str) -> None:
        """Validate that a resolved folder path is a readable directory."""
        if not os.path.isdir(real_path):
            raise ValueError(f"Path is not a directory: {real_path}")
        if not os.access(real_path, os.R_OK):
            raise PermissionError(f"Directory is not readable: {real_path}")

    def _parse_json_list(self, val: object) -> list[str]:
        """Parse a JSON-encoded list string or pass a list through unchanged."""
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []

