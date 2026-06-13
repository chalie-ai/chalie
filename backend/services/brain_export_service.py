"""Brain Export Service — snapshot DB + files into an encrypted archive.

Output format (.chalie-backup):
    Magic:       "CHALIEBACKUP" (12 ASCII bytes)
    Version:     1 (uint16 LE)
    Salt:        16 bytes (PBKDF2)
    Iterations:  uint32 LE (600000)
    Nonce:       12 bytes (AES-256-GCM)
    Ciphertext:  variable (AES-256-GCM encrypted zip bytes)
    Tag:         16 bytes (embedded in ciphertext by AESGCM)

Plaintext inside ciphertext is a .zip archive containing:
    brain-export-<timestamp>/
        export.json    — { version, exported_at, app_version, db_size, file_count, checksums }
        brain.db       — SQLite DB snapshot (via sqlite3 backup API)
        files/         — (if include_files=True) document files from data/documents/
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
_PBKDF2_SALT_SIZE = 16
_PBKDF2_ITERATIONS = 600_000
_NONCE_SIZE = 12
_HEADER_STRUCT = struct.Struct("<12sH16sI12s")


class BrainExportService:

    def export_brain(
        self,
        include_files: bool = True,
        encrypt: bool = True,
        password: str | None = None,
    ) -> Path:
        """Export the brain to a .chalie-backup file.

        Returns:
            Path to the exported .chalie-backup file (in a temp directory).
        """
        if encrypt and not password:
            raise ValueError("Password required when encryption is enabled")

        tmp_dir = tempfile.mkdtemp(prefix="chalie-export-")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        archive_name = f"brain-export-{timestamp}"
        archive_dir = Path(tmp_dir) / archive_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        db_path = FileMapperService.get_db_path()
        docs_dir = FileMapperService.get_documents_path()

        export_meta = {
            "version": _FORMAT_VERSION,
            "exported_at": timestamp,
            "app_version": self._read_version(),
            "include_files": include_files,
            "db_size": 0,
            "file_count": 0,
            "checksums": {},
        }

        db_snapshot = archive_dir / "brain.db"
        self._snapshot_db(db_path, db_snapshot)
        export_meta["db_size"] = db_snapshot.stat().st_size
        export_meta["checksums"]["brain.db"] = self._sha256(db_snapshot)

        if include_files and docs_dir.is_dir():
            files_dir = archive_dir / "files"
            count = self._collect_files(docs_dir, files_dir)
            export_meta["file_count"] = count

        (archive_dir / "export.json").write_text(
            json.dumps(export_meta, indent=2), encoding="utf-8"
        )

        zip_path = Path(tmp_dir) / f"{archive_name}.zip"
        self._build_zip(archive_dir, zip_path)

        if encrypt and password:
            output_path = Path(tmp_dir) / f"{archive_name}.chalie-backup"
            self._encrypt_file(zip_path, output_path, password)
            zip_path.unlink()
            return output_path

        output_path = Path(tmp_dir) / f"{archive_name}.chalie-backup"
        self._write_unencrypted(zip_path, output_path)
        zip_path.unlink()
        return output_path

    def _snapshot_db(self, src: Path, dst: Path) -> None:
        """Create a consistent snapshot of the SQLite database using the backup API."""
        import sqlite3

        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()

    def _collect_files(self, src_dir: Path, dest_dir: Path) -> int:
        """Copy all files from src_dir into dest_dir, preserving structure.

        Returns the number of files copied.
        """
        count = 0
        dest_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_dir)
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                count += 1
        return count

    def _build_zip(self, source_dir: Path, zip_path: Path) -> None:
        """Create a zip archive from source_dir."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in source_dir.rglob("*"):
                if item.is_file():
                    arcname = str(item.relative_to(source_dir.parent))
                    zf.write(item, arcname)

    def _encrypt_file(self, plaintext_path: Path, output_path: Path, password: str) -> None:
        """Encrypt a file with AES-256-GCM using a password-derived key.

        Output format: MAGIC(12) + VERSION(2) + SALT(16) + ITERATIONS(4) + NONCE(12) + CIPHERTEXT+TAG
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        plaintext = plaintext_path.read_bytes()

        salt = os.urandom(_PBKDF2_SALT_SIZE)
        nonce = os.urandom(_NONCE_SIZE)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        key = kdf.derive(password.encode("utf-8"))

        ciphertext_with_tag = AESGCM(key).encrypt(nonce, plaintext, None)

        header = _HEADER_STRUCT.pack(
            _MAGIC, _FORMAT_VERSION, salt, _PBKDF2_ITERATIONS, nonce
        )

        output_path.write_bytes(header + ciphertext_with_tag)

    def _write_unencrypted(self, plaintext_path: Path, output_path: Path) -> None:
        plaintext = plaintext_path.read_bytes()
        header = _HEADER_STRUCT.pack(
            _MAGIC, _FORMAT_VERSION, b"\x00" * _PBKDF2_SALT_SIZE, 0, b"\x00" * _NONCE_SIZE
        )
        output_path.write_bytes(header + plaintext)

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _read_version() -> str:
        version_path = FileMapperService.get_version_path()
        if version_path.exists():
            return version_path.read_text(encoding="utf-8").strip()
        return "unknown"

    def get_brain_info(self) -> dict:
        """Return summary info about the brain for the UI."""
        db_path = FileMapperService.get_db_path()
        docs_dir = FileMapperService.get_documents_path()

        db_size = db_path.stat().st_size if db_path.exists() else 0
        doc_count = 0
        if docs_dir.is_dir():
            doc_count = sum(1 for _ in docs_dir.rglob("*") if _.is_file())

        return {
            "db_size_bytes": db_size,
            "db_size_human": self._human_size(db_size),
            "document_count": doc_count,
        }

    @staticmethod
    def _human_size(size: int | float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
