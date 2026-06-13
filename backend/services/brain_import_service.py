"""Brain Import Service — decrypt and restore a .chalie-backup archive.

Reverses BrainExportService.export_brain():
  1. Validate magic + version
  2. Derive key from passphrase, AES-256-GCM decrypt
  3. Validate checksums from export.json
  4. Create a pre-restore backup of the current brain
  5. Replace brain.db via sqlite3 backup API
  6. Restore files/ to data/documents/
  7. Return summary
"""

import hashlib
import json
import logging
import os
import shutil
import struct
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_MAGIC = b"CHALIEBACKUP"
_FORMAT_VERSION = 1
_HEADER_STRUCT = struct.Struct("<12sH16sI12s")


class BrainImportService:

    def import_brain(
        self,
        backup_path: Path,
        password: str | None = None,
    ) -> dict:
        """Import a .chalie-backup file, replacing the current brain.

        Returns:
            Summary dict with keys: exported_at, db_size, files_restored, success.
        """
        raw = backup_path.read_bytes()
        magic, version, salt, iterations, nonce = _HEADER_STRUCT.unpack_from(raw)
        header_size = _HEADER_STRUCT.size

        if magic != _MAGIC:
            raise ValueError("Not a valid Chalie backup file (bad magic)")
        if version > _FORMAT_VERSION:
            raise ValueError(f"Unsupported backup version {version}")

        ciphertext_with_tag = raw[header_size:]

        if iterations == 0:
            plaintext = ciphertext_with_tag
        elif not password:
            raise ValueError("Password required (backup is encrypted)")
        else:
            plaintext = self._decrypt(ciphertext_with_tag, password, salt, iterations, nonce)

        tmp_dir = tempfile.mkdtemp(prefix="chalie-import-")
        zip_path = Path(tmp_dir) / "archive.zip"
        zip_path.write_bytes(plaintext)

        with zipfile.ZipFile(zip_path, "r") as zf:
            self._safe_extractall(zf, Path(tmp_dir))

        export_dirs = [p for p in Path(tmp_dir).iterdir() if p.is_dir() and p.name.startswith("brain-export-")]
        if not export_dirs:
            raise ValueError("Backup archive has no brain-export directory")
        content_dir = export_dirs[0]

        export_json_path = content_dir / "export.json"
        if not export_json_path.exists():
            raise ValueError("Backup archive is missing export.json")

        export_meta = json.loads(export_json_path.read_text(encoding="utf-8"))

        db_snapshot = content_dir / "brain.db"
        if not db_snapshot.exists():
            raise ValueError("Backup archive is missing brain.db")

        if "brain.db" in export_meta.get("checksums", {}):
            actual = self._sha256(db_snapshot)
            expected = export_meta["checksums"]["brain.db"]
            if actual != expected:
                raise ValueError("brain.db checksum mismatch — backup may be corrupted")

        self._backup_current_brain()

        live_db = FileMapperService.get_db_path()
        self._restore_db(db_snapshot, live_db)

        files_restored = 0
        files_src = content_dir / "files"
        if files_src.is_dir():
            docs_dir = FileMapperService.get_documents_path()
            docs_dir.mkdir(parents=True, exist_ok=True)
            files_restored = self._restore_files(files_src, docs_dir)

        result = {
            "success": True,
            "exported_at": export_meta.get("exported_at"),
            "app_version": export_meta.get("app_version"),
            "db_size": db_snapshot.stat().st_size,
            "files_restored": files_restored,
        }

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        return result

    def _safe_extractall(self, zf: zipfile.ZipFile, dest: Path) -> None:
        dest_resolved = dest.resolve()
        for member in zf.infolist():
            member_path = (dest / member.filename).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise ValueError(f"Unsafe zip entry: {member.filename}")
            zf.extract(member, dest)

    def _decrypt(
        self,
        ciphertext_with_tag: bytes,
        password: str,
        salt: bytes,
        iterations: int,
        nonce: bytes,
    ) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        from cryptography.exceptions import InvalidTag

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = kdf.derive(password.encode("utf-8"))

        try:
            return AESGCM(key).decrypt(nonce, ciphertext_with_tag, None)
        except InvalidTag:
            raise ValueError("Wrong passphrase or corrupted backup")

    def _backup_current_brain(self) -> Path | None:
        """Create a timestamped backup of the current brain.db before overwrite."""
        db_path = FileMapperService.get_db_path()
        if not db_path.exists():
            return None

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = db_path.with_name(f"brain-{stamp}.pre-restore.bak")
        shutil.copy2(db_path, backup_path)
        logger.info(f"[BrainImport] Pre-restore backup: {backup_path}")
        return backup_path

    def _restore_db(self, src: Path, dst: Path) -> None:
        import sqlite3
        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()

    def _restore_files(self, src_dir: Path, dest_dir: Path) -> int:
        count = 0
        for item in src_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_dir)
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                count += 1
        return count

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
