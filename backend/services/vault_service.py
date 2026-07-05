"""Vault Service — envelope encryption with password-derived master key."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, cast

from services.database import Database

# FileMapperService owns every repository-layout path (CLAUDE.md rule #9). The
# vault key-material backup lives under data/secure/ so it persists on the same
# Docker volume as the database — see FileMapperService.get_secure_dir().
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

# ── Vault-open outcome codes ────────────────────────────────────────────────────
# Shared between VaultService.unlock_or_restore() and the login endpoint so the
# safety-critical "wipe on unrecoverable" branch can never break on a typo.
OUTCOME_UNLOCKED = "unlocked"
OUTCOME_RESTORED = "restored"
OUTCOME_UNRECOVERABLE = "unrecoverable"

# ── Nonce / key constants ──────────────────────────────────────────────────────
_NONCE_SIZE = 12          # AES-GCM recommended nonce length (bytes)
_DEK_SIZE = 32            # AES-256 key length (bytes)
_KDF_SALT_SIZE = 32       # PBKDF2 salt length (bytes)
_KDF_ITERATIONS = 600_000 # PBKDF2-HMAC-SHA256 iteration count
_KDF_ALGORITHM = "pbkdf2_sha256"

# ── Backup filesystem permissions ───────────────────────────────────────────────
_SECURE_DIR_MODE = 0o700  # owner rwx only — the data/secure/ backup directory
_SECURE_FILE_MODE = 0o400 # owner read-only — each backup file
_BACKUP_RETENTION = 6     # keep only the newest N vault_backup_*.json files


# ── Custom exception ───────────────────────────────────────────────────────────

class VaultLockedError(Exception):
    """Raised when an encrypt/decrypt operation is attempted on a sealed vault.

    The vault must be unlocked via :meth:`VaultService.unlock` before any
    cryptographic operations can be performed.
    """


# ── Module-level DEK cache (singleton) ────────────────────────────────────────

@dataclass
class _VaultState:
    """Module-level singleton that caches the plaintext DEK in memory.

    This object is intentionally **not** persisted anywhere.  It holds the
    plaintext DEK only for the lifetime of the current server process and only
    after a successful :meth:`VaultService.unlock` call.

    Attributes:
        dek: The 32-byte plaintext Data Encryption Key, or ``None`` when the
            vault is sealed (not yet unlocked or explicitly locked).
    """
    dek: Optional[bytes] = field(default=None)


# Single shared instance for the whole process.
_vault_state = _VaultState()


@dataclass(frozen=True)
class _VaultKeyMaterial:
    """The password-protected key material for one vault generation.

    Groups the six fields that ``vault_config`` and the filesystem backup both
    store so they always travel together — eliminating long parameter lists and
    the risk of passing them out of order.
    """
    salt: bytes
    kdf_algorithm: str
    kdf_iterations: int
    wrapped_dek: bytes
    nonce: bytes
    created_at: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _derive_kek(password: str, salt: bytes, iterations: int = _KDF_ITERATIONS) -> bytes:
    """Derive a 256-bit Key Encryption Key from *password* using PBKDF2-HMAC-SHA256.

    Args:
        password:   The user's master password (UTF-8 string).
        salt:       Random 32-byte salt read from ``vault_config.kdf_salt``.
        iterations: PBKDF2 iteration count.  Defaults to :data:`_KDF_ITERATIONS`.

    Returns:
        32-byte KEK bytes suitable for use with
        :class:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM`.
    """
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_DEK_SIZE,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode("utf-8"))


def _aesgcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM.

    Args:
        key:       32-byte AES key.
        nonce:     12-byte random nonce.
        plaintext: Arbitrary-length plaintext bytes.

    Returns:
        Ciphertext + 16-byte authentication tag as a single bytes object
        (the format produced by
        :meth:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM.encrypt`).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).encrypt(nonce, plaintext, None)


def _aesgcm_decrypt(key: bytes, nonce: bytes, ciphertext_tag: bytes) -> bytes:
    """Decrypt *ciphertext_tag* with AES-256-GCM.

    Args:
        key:             32-byte AES key.
        nonce:           12-byte nonce that was used during encryption.
        ciphertext_tag:  Ciphertext + 16-byte tag
            (as produced by :func:`_aesgcm_encrypt`).

    Returns:
        Plaintext bytes.

    Raises:
        :exc:`cryptography.exceptions.InvalidTag`: When the tag verification fails
            (wrong key, wrong nonce, or tampered ciphertext).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(nonce, ciphertext_tag, None)


# ── VaultService ───────────────────────────────────────────────────────────────

class VaultService:
    """Envelope-encryption service backed by a password-derived master key.

    The service stores all KDF parameters and the wrapped DEK in the
    ``vault_config`` table (singleton row, id=1).  Consumer services call
    :meth:`encrypt` / :meth:`decrypt` to protect secrets without ever handling
    the raw DEK.

    Lifecycle::

        # First-run (registration)
        vault.initialize(master_password)

        # Every subsequent login
        if vault.unlock(master_password):
            ...  # vault is now open for encrypt/decrypt

        # On logout / shutdown (optional)
        vault.lock()

    Reaches the DB through the static :class:`~services.database.Database`
    gateway — no instance state.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, password: str) -> None:
        """Generate a new DEK, wrap it with a password-derived KEK, and persist.

        Called **once** when the master account is first registered.  If a
        ``vault_config`` row already exists it is deleted and replaced so that
        re-registration always starts with a fresh vault.

        After a successful call the vault is *not* automatically unlocked.
        Call :meth:`unlock` immediately afterwards if the plaintext DEK is
        needed at registration time.

        Args:
            password: The user's chosen master password (UTF-8 string).

        Raises:
            Exception: Propagates any database or cryptography error so the
                caller (registration endpoint) can roll back the account row.
        """
        dek = os.urandom(_DEK_SIZE)
        salt = os.urandom(_KDF_SALT_SIZE)
        nonce = os.urandom(_NONCE_SIZE)

        kek = _derive_kek(password, salt)
        wrapped_dek = _aesgcm_encrypt(kek, nonce, dek)

        from services.time_utils import utc_now
        now_iso = utc_now().isoformat()

        km = _VaultKeyMaterial(
            salt, _KDF_ALGORITHM, _KDF_ITERATIONS, wrapped_dek, nonce, now_iso,
        )
        self._persist_vault_config(km)
        self._write_backup(km)

        logger.info(
            "[Vault] Vault initialised — new DEK wrapped "
            "with password-derived KEK; key material backed up"
        )

    def unlock(self, password: str) -> bool:
        """Derive the KEK from *password*, unwrap the DEK, and cache it in memory.

        On success the cached DEK allows :meth:`encrypt` and :meth:`decrypt`
        to operate.  On failure (wrong password) the vault remains sealed and
        ``False`` is returned without raising.

        Args:
            password: The user's master password to verify.

        Returns:
            ``True`` if the password is correct and the vault is now unlocked,
            ``False`` if the tag verification fails (wrong password).

        Raises:
            RuntimeError: If ``vault_config`` has no row (vault was never
                initialised — call :meth:`initialize` first).
        """
        from cryptography.exceptions import InvalidTag

        row = self._load_vault_config()
        if row is None:
            raise RuntimeError(
                "[Vault] vault_config is empty — call initialize() before unlock()"
            )

        salt = bytes(cast(bytes, row["kdf_salt"]))
        nonce = bytes(cast(bytes, row["dek_nonce"]))
        wrapped_dek = bytes(cast(bytes, row["wrapped_dek"]))
        iterations = cast(int, row["kdf_iterations"])

        kek = _derive_kek(password, salt, iterations)
        try:
            dek = _aesgcm_decrypt(kek, nonce, wrapped_dek)
        except InvalidTag:
            logger.warning(
                "[Vault] unlock() failed — incorrect password "
                "or corrupted vault_config"
            )
            return False

        _vault_state.dek = dek
        logger.info("[Vault] Vault unlocked — DEK cached in memory")
        return True

    def unlock_or_restore(self, password: str) -> str:
        """Open the vault with *password*, recovering from backup if needed.

        Callers pass a password that has ALREADY been verified against the
        master account hash, so any failure to open the vault means the live
        ``vault_config`` row is missing or corrupt — not a wrong password.

        Recovery order:
          1. Try the live ``vault_config`` row.
          2. If it is missing or corrupt, try every filesystem backup key
             (current, then previous). The first key that unwraps is written
             back into ``vault_config`` and promoted to the latest backup.
          3. If nothing opens, the DEK is permanently lost.

        Args:
            password: The already-verified master password.

        Returns:
            ``"unlocked"``      — opened via the live ``vault_config`` row.
            ``"restored"``      — live row was missing/corrupt; a backup matched
                                  and the vault was rebuilt and opened. No data loss.
            ``"unrecoverable"`` — neither the live row nor any backup could be
                                  opened. The caller MUST wipe the account to
                                  force re-onboarding (encrypted data is lost).
        """
        try:
            if self.unlock(password):
                return OUTCOME_UNLOCKED
            logger.error(
                "[Vault] Live vault_config did not unlock with the verified "
                "password — row is corrupt. Attempting backup recovery."
            )
        except RuntimeError:
            logger.error(
                "[Vault] vault_config row is missing at unlock time. "
                "Attempting backup recovery."
            )

        if self._restore_from_backup(password):
            return OUTCOME_RESTORED

        logger.error(
            "[Vault] UNRECOVERABLE — the vault key is corrupted and no valid "
            "backup exists. The DEK is permanently lost; encrypted data cannot "
            "be decrypted. The account must be re-onboarded."
        )
        return OUTCOME_UNRECOVERABLE

    def is_unlocked(self) -> bool:
        """Return ``True`` if the vault has been unlocked and the DEK is in memory."""
        return _vault_state.dek is not None

    def get_state(self) -> str:
        """Return the vault state as a string.

        Returns:
            ``"unlocked"``      — DEK is in memory.
            ``"locked"``        — ``vault_config`` row exists but DEK is not loaded.
            ``"uninitialized"`` — no ``vault_config`` row.
        """
        if self.is_unlocked():
            return "unlocked"
        try:
            row = self._load_vault_config()
            return "locked" if row is not None else "uninitialized"
        except Exception as exc:
            logger.warning("[Vault] get_state() failed: %s", exc)
            return "uninitialized"

    def lock(self) -> None:
        """Seal the vault by clearing the in-memory DEK."""
        _vault_state.dek = None
        logger.info("[Vault] Vault locked — DEK cleared from memory")

    def reset(self) -> None:
        """Tear down all vault state for a re-onboarding.

        Deletes the ``vault_config`` row, clears the in-memory DEK, and removes
        the (now-useless) backup files. Called only after recovery has failed
        and the account is being wiped — the backups have already been tried and
        none matched the verified password, so they protect nothing.
        """
        with Database.transaction() as conn:
            conn.execute("DELETE FROM vault_config WHERE id = 1")
        _vault_state.dek = None
        for path in FileMapperService.list_vault_backups():
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("[Vault] Could not delete backup %s: %s", path, exc)
        logger.warning("[Vault] Vault reset — vault_config and backups removed")

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt *plaintext* with the cached DEK using AES-256-GCM.

        Output wire format: ``nonce (12 B) || AES-GCM ciphertext+tag``

        Raises:
            :exc:`VaultLockedError`: If the vault has not been unlocked.
        """
        if not self.is_unlocked():
            raise VaultLockedError("Vault is locked — call unlock() before encrypting")
        nonce = os.urandom(_NONCE_SIZE)
        ciphertext_tag = _aesgcm_encrypt(cast(bytes, _vault_state.dek), nonce, plaintext)
        return nonce + ciphertext_tag

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt an encrypted blob produced by :meth:`encrypt`.

        Supports ONLY the current wire format: ``nonce (12 B) || AES-GCM ciphertext+tag``.

        Raises:
            :exc:`VaultLockedError`: If the vault has not been unlocked.
            :exc:`ValueError`:       If decryption fails (tampered or wrong DEK).
        """
        if not self.is_unlocked():
            raise VaultLockedError("Vault is locked — call unlock() before decrypting")
        nonce = blob[:_NONCE_SIZE]
        ciphertext_tag = blob[_NONCE_SIZE:]
        try:
            return _aesgcm_decrypt(cast(bytes, _vault_state.dek), nonce, ciphertext_tag)
        except Exception as exc:
            raise ValueError(f"Decryption failed: {exc}") from exc

    def encrypt_str(self, s: str) -> bytes:
        """UTF-8–encode *s* and encrypt it with :meth:`encrypt`."""
        return self.encrypt(s.encode("utf-8"))

    def decrypt_str(self, blob: bytes) -> str:
        """Decrypt *blob* with :meth:`decrypt` and UTF-8–decode the result."""
        return self.decrypt(blob).decode("utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_vault_config(self) -> Optional[dict[str, object]]:
        """Read the singleton ``vault_config`` row (id=1) from the database.

        Returns:
            A dict-like row or ``None`` if the table is empty.
        """
        row = Database.conn().execute(
            "SELECT kdf_salt, kdf_algorithm, kdf_iterations, "
            "wrapped_dek, dek_nonce, created_at, updated_at "
            "FROM vault_config WHERE id = 1"
        ).fetchone()
        return cast("dict[str, object] | None", row)  # sqlite3.Row or None

    def _persist_vault_config(self, km: "_VaultKeyMaterial") -> None:
        """Replace the singleton ``vault_config`` row (id=1) with *km*.
        Shared by :meth:`initialize` and backup restoration."""
        with Database.transaction() as conn:
            conn.execute("DELETE FROM vault_config WHERE id = 1")
            conn.execute(
                """
                INSERT INTO vault_config
                    (id, kdf_salt, kdf_algorithm, kdf_iterations,
                     wrapped_dek, dek_nonce, created_at, updated_at)
                VALUES
                    (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (km.salt, km.kdf_algorithm, km.kdf_iterations, km.wrapped_dek,
                 km.nonce, km.created_at, km.created_at),
            )

    # ------------------------------------------------------------------
    # Filesystem key-material backup
    # ------------------------------------------------------------------

    def _write_backup(self, km: "_VaultKeyMaterial") -> None:
        """Append *km* as a fresh, uniquely-stamped backup under ``data/secure/``.

        Every DEK generation writes a new ``vault_backup_<stamp>.json``; only the
        newest ``_BACKUP_RETENTION`` files are kept so recovery can still fall
        back through recent generations without the directory growing unbounded
        (these files are bundled into every instance snapshot). The backup holds
        the same password-protected material as the ``vault_config`` row, so it
        is no weaker than the database. Written atomically (tmp + rename) and
        locked to owner-read-only inside an owner-only directory.
        """
        from services.time_utils import utc_now

        payload = {
            "kdf_salt": km.salt.hex(),
            "wrapped_dek": km.wrapped_dek.hex(),
            "dek_nonce": km.nonce.hex(),
            "kdf_algorithm": km.kdf_algorithm,
            "kdf_iterations": km.kdf_iterations,
            "created_at": km.created_at,
        }
        secure_dir = FileMapperService.get_secure_dir()
        secure_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(secure_dir, _SECURE_DIR_MODE)

        stamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
        path = FileMapperService.get_vault_backup_path(stamp)
        # Never clobber a retained backup if two writes land in the same
        # microsecond — bump a suffix until the name is free.
        suffix = 1
        while path.exists():
            path = FileMapperService.get_vault_backup_path(f"{stamp}_{suffix}")
            suffix += 1

        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # Never leave a wrapped-key tmp file behind if the atomic rename fails.
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        os.chmod(path, _SECURE_FILE_MODE)
        logger.info("[Vault] Key material backed up to %s", path.name)

        # Prune everything beyond the newest N (list is sorted newest-first).
        for stale in FileMapperService.list_vault_backups()[_BACKUP_RETENTION:]:
            stale.unlink(missing_ok=True)

    def _restore_from_backup(self, password: str) -> bool:
        """Rebuild ``vault_config`` from the first backup that *password* opens.

        Tries every retained backup, newest first. For each, derives the KEK from
        *password* + the backup salt and attempts to unwrap the DEK; iterating all
        generations means a corrupt latest backup is simply skipped. The first
        success is written back into ``vault_config`` and cached in memory — the
        recovered DEK is the original, so no fresh backup is taken. Returns
        ``True`` on success.
        """
        from cryptography.exceptions import InvalidTag
        from services.time_utils import utc_now

        for path in FileMapperService.list_vault_backups():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                salt = bytes.fromhex(data["kdf_salt"])
                nonce = bytes.fromhex(data["dek_nonce"])
                wrapped_dek = bytes.fromhex(data["wrapped_dek"])
                kdf_algorithm = data["kdf_algorithm"]
                kdf_iterations = int(data["kdf_iterations"])
            except (ValueError, KeyError, OSError) as exc:
                logger.warning(
                    "[Vault] Backup %s is unreadable or malformed: %s", path.name, exc
                )
                continue

            kek = _derive_kek(password, salt, kdf_iterations)
            try:
                dek = _aesgcm_decrypt(kek, nonce, wrapped_dek)
            except InvalidTag:
                logger.warning(
                    "[Vault] Backup %s did not match the password — trying next",
                    path.name,
                )
                continue

            # Restoration recovers the original DEK, so no sealed data is
            # orphaned. The backup already exists and is retained — no re-write.
            km = _VaultKeyMaterial(
                salt, kdf_algorithm, kdf_iterations, wrapped_dek, nonce,
                utc_now().isoformat(),
            )
            self._persist_vault_config(km)
            _vault_state.dek = dek
            logger.info(
                "[Vault] Restored vault_config from backup %s — vault unlocked, "
                "no data loss",
                path.name,
            )
            return True

        return False


# ── Module-level factory ───────────────────────────────────────────────────────

_vault_service_instance: Optional[VaultService] = None


def get_vault_service() -> VaultService:
    """Return the process-wide :class:`VaultService` singleton.

    The instance is cached after first creation so every call site shares the
    same object.
    """
    global _vault_service_instance
    if _vault_service_instance is None:
        _vault_service_instance = VaultService()
    return _vault_service_instance
