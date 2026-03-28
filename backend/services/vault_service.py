"""Vault Service — envelope encryption with password-derived master key.

Implements AES-256-GCM envelope encryption:
  - A random 256-bit Data Encryption Key (DEK) is generated once on
    registration and never stored in plaintext.
  - A Key Encryption Key (KEK) is derived from the user's master password
    using PBKDF2-HMAC-SHA256 (600 000 iterations, 32-byte random salt).
  - The DEK is wrapped (encrypted) with the KEK and stored in ``vault_config``
    alongside the KDF parameters.
  - All consumer encryption/decryption operations use the cached plaintext DEK
    which is loaded into memory only after a successful ``unlock()`` call.

Wire format for all secrets (encrypt output / decrypt input):
    nonce (12 bytes) || AES-256-GCM ciphertext+tag

The service is intentionally single-user: the plaintext DEK lives in a
module-level singleton that is safe for a single-process, single-account
architecture.  It is cleared on ``lock()`` or server restart.
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Nonce / key constants ──────────────────────────────────────────────────────
_NONCE_SIZE = 12          # AES-GCM recommended nonce length (bytes)
_DEK_SIZE = 32            # AES-256 key length (bytes)
_KDF_SALT_SIZE = 32       # PBKDF2 salt length (bytes)
_KDF_ITERATIONS = 600_000 # PBKDF2-HMAC-SHA256 iteration count
_KDF_ALGORITHM = "pbkdf2_sha256"


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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _derive_kek(password: str, salt: bytes, iterations: int = _KDF_ITERATIONS) -> bytes:
    """Derive a 256-bit Key Encryption Key from *password* using PBKDF2-HMAC-SHA256.

    Args:
        password:   The user's master password (UTF-8 string).
        salt:       Random 32-byte salt read from ``vault_config.kdf_salt``.
        iterations: PBKDF2 iteration count.  Defaults to :data:`_KDF_ITERATIONS`.

    Returns:
        32-byte KEK bytes suitable for use with :class:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM`.
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
        (the format produced by :meth:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM.encrypt`).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).encrypt(nonce, plaintext, None)


def _aesgcm_decrypt(key: bytes, nonce: bytes, ciphertext_tag: bytes) -> bytes:
    """Decrypt *ciphertext_tag* with AES-256-GCM.

    Args:
        key:             32-byte AES key.
        nonce:           12-byte nonce that was used during encryption.
        ciphertext_tag:  Ciphertext + 16-byte tag (as produced by :func:`_aesgcm_encrypt`).

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

    Args:
        database_service: The shared :class:`~services.database_service.DatabaseService`
            singleton.  Injected to allow deterministic testing.
    """

    def __init__(self, database_service):
        """Initialise the service with an injected database dependency.

        Args:
            database_service: The :class:`~services.database_service.DatabaseService`
                instance to use for all ``vault_config`` reads and writes.
        """
        self._db = database_service

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
        # Generate fresh cryptographic material
        dek = os.urandom(_DEK_SIZE)
        salt = os.urandom(_KDF_SALT_SIZE)
        nonce = os.urandom(_NONCE_SIZE)

        # Derive KEK and wrap the DEK
        kek = _derive_kek(password, salt)
        wrapped_dek = _aesgcm_encrypt(kek, nonce, dek)

        from services.time_utils import utc_now
        now_iso = utc_now().isoformat()

        with self._db.connection() as conn:
            conn.execute("DELETE FROM vault_config WHERE id = 1")
            conn.execute(
                """
                INSERT INTO vault_config
                    (id, kdf_salt, kdf_algorithm, kdf_iterations,
                     wrapped_dek, dek_nonce, created_at, updated_at)
                VALUES
                    (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (salt, _KDF_ALGORITHM, _KDF_ITERATIONS, wrapped_dek, nonce,
                 now_iso, now_iso),
            )

        logger.info("[Vault] Vault initialised — new DEK wrapped with password-derived KEK")

    def unlock(self, password: str) -> bool:
        """Derive the KEK from *password*, unwrap the DEK, and cache it in memory.

        On success the cached DEK allows :meth:`encrypt` and :meth:`decrypt`
        to operate.  On failure (wrong password) the vault remains sealed and
        ``False`` is returned without raising.

        After a successful unlock, if the ``encryption_keys`` table contains a
        legacy Fernet key row (migration window is open), attempts to
        transparently re-encrypt all legacy Fernet-encrypted consumer rows with
        the new AES-256-GCM vault DEK via :meth:`_run_legacy_migration`.
        The migration is best-effort: a failure leaves the fallback decryption
        path active for the affected tables and is retried on the next unlock.

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

        salt = bytes(row["kdf_salt"])
        nonce = bytes(row["dek_nonce"])
        wrapped_dek = bytes(row["wrapped_dek"])
        iterations = row["kdf_iterations"]

        kek = _derive_kek(password, salt, iterations)
        try:
            dek = _aesgcm_decrypt(kek, nonce, wrapped_dek)
        except InvalidTag:
            logger.warning("[Vault] unlock() failed — incorrect password or corrupted vault_config")
            return False

        _vault_state.dek = dek
        logger.info("[Vault] Vault unlocked — DEK cached in memory")

        # Attempt to run the one-time legacy Fernet → AES-GCM migration.
        # _run_legacy_migration() is a no-op when encryption_keys is absent.
        self._run_legacy_migration()

        return True

    def is_unlocked(self) -> bool:
        """Return ``True`` if the vault has been unlocked and the DEK is in memory.

        Returns:
            ``True`` when :meth:`unlock` has been called successfully since the
            last :meth:`lock` call or server restart, ``False`` otherwise.
        """
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
        except Exception:
            return "uninitialized"

    def lock(self) -> None:
        """Seal the vault by clearing the in-memory DEK.

        After this call :meth:`encrypt` and :meth:`decrypt` will raise
        :exc:`VaultLockedError` until :meth:`unlock` is called again.
        """
        _vault_state.dek = None
        logger.info("[Vault] Vault locked — DEK cleared from memory")

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt *plaintext* with the cached DEK using AES-256-GCM.

        A fresh 12-byte random nonce is generated for every call so that
        encrypting the same value twice produces different ciphertexts.

        Output wire format::

            nonce (12 bytes) || AES-GCM ciphertext+tag

        Args:
            plaintext: Arbitrary bytes to encrypt.

        Returns:
            Encrypted blob: ``nonce (12 B) + ciphertext + tag (16 B)``.

        Raises:
            :exc:`VaultLockedError`: If the vault has not been unlocked.
        """
        if not self.is_unlocked():
            raise VaultLockedError("Vault is locked — call unlock() before encrypting")
        nonce = os.urandom(_NONCE_SIZE)
        ciphertext_tag = _aesgcm_encrypt(_vault_state.dek, nonce, plaintext)
        return nonce + ciphertext_tag

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt an encrypted blob produced by :meth:`encrypt`.

        Splits the blob into ``nonce || ciphertext+tag`` and decrypts with the
        cached DEK.  If AES-GCM tag verification fails and a legacy Fernet key
        exists in ``encryption_keys`` (migration window), falls back to Fernet
        decryption so that not-yet-migrated rows remain readable.

        Args:
            blob: Encrypted blob as returned by :meth:`encrypt`, or a legacy
                Fernet token stored as UTF-8–encoded bytes.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            :exc:`VaultLockedError`:  If the vault has not been unlocked.
            :exc:`ValueError`:        If decryption fails with both AES-GCM and
                the Fernet fallback (tampered or unrecognised ciphertext).
        """
        if not self.is_unlocked():
            raise VaultLockedError("Vault is locked — call unlock() before decrypting")

        from cryptography.exceptions import InvalidTag

        # --- Primary path: AES-256-GCM -----------------------------------
        if len(blob) > _NONCE_SIZE:
            nonce = blob[:_NONCE_SIZE]
            ciphertext_tag = blob[_NONCE_SIZE:]
            try:
                return _aesgcm_decrypt(_vault_state.dek, nonce, ciphertext_tag)
            except InvalidTag:
                pass  # Fall through to legacy Fernet path

        # --- Migration-window fallback: legacy Fernet --------------------
        fernet_plaintext = self._try_fernet_fallback(blob)
        if fernet_plaintext is not None:
            return fernet_plaintext

        raise ValueError(
            "[Vault] decrypt() failed — ciphertext is invalid or was not encrypted "
            "with the current DEK and no legacy Fernet key was found"
        )

    def encrypt_str(self, s: str) -> bytes:
        """UTF-8–encode *s* and encrypt it with :meth:`encrypt`.

        Args:
            s: Plaintext string to encrypt.

        Returns:
            Encrypted blob (see :meth:`encrypt` for wire format).

        Raises:
            :exc:`VaultLockedError`: If the vault has not been unlocked.
        """
        return self.encrypt(s.encode("utf-8"))

    def decrypt_str(self, blob: bytes) -> str:
        """Decrypt *blob* with :meth:`decrypt` and UTF-8–decode the result.

        Args:
            blob: Encrypted blob as returned by :meth:`encrypt_str`.

        Returns:
            Decrypted plaintext string.

        Raises:
            :exc:`VaultLockedError`:  If the vault has not been unlocked.
            :exc:`ValueError`:        If decryption fails.
            :exc:`UnicodeDecodeError`: If the decrypted bytes are not valid UTF-8.
        """
        return self.decrypt(blob).decode("utf-8")

    # ------------------------------------------------------------------
    # Password change
    # ------------------------------------------------------------------

    def change_password(self, old_password: str, new_password: str) -> bool:
        """Re-wrap the DEK under a new password without re-encrypting any data.

        Verifies *old_password* by attempting to unwrap the current DEK, then
        derives a fresh KEK from *new_password* with a new random salt and
        re-wraps the **same** DEK.  The ``vault_config`` row is updated
        atomically in a single database transaction.

        Consumer-table ciphertext is **not** touched — the DEK itself does not
        change, so all existing encrypted values remain valid.

        Args:
            old_password: The current master password (used to verify identity).
            new_password: The desired new master password.

        Returns:
            ``True`` if the password was changed successfully,
            ``False`` if *old_password* is incorrect.

        Raises:
            RuntimeError: If ``vault_config`` has no row.
        """
        from cryptography.exceptions import InvalidTag

        row = self._load_vault_config()
        if row is None:
            raise RuntimeError(
                "[Vault] vault_config is empty — call initialize() before change_password()"
            )

        old_salt = bytes(row["kdf_salt"])
        old_nonce = bytes(row["dek_nonce"])
        old_wrapped_dek = bytes(row["wrapped_dek"])
        iterations = row["kdf_iterations"]

        # Verify old password
        old_kek = _derive_kek(old_password, old_salt, iterations)
        try:
            dek = _aesgcm_decrypt(old_kek, old_nonce, old_wrapped_dek)
        except InvalidTag:
            logger.warning("[Vault] change_password() failed — incorrect old password")
            return False

        # Wrap DEK under new KEK
        new_salt = os.urandom(_KDF_SALT_SIZE)
        new_nonce = os.urandom(_NONCE_SIZE)
        new_kek = _derive_kek(new_password, new_salt)
        new_wrapped_dek = _aesgcm_encrypt(new_kek, new_nonce, dek)

        from services.time_utils import utc_now

        with self._db.connection() as conn:
            conn.execute(
                """
                UPDATE vault_config
                SET kdf_salt       = ?,
                    kdf_algorithm  = ?,
                    kdf_iterations = ?,
                    wrapped_dek    = ?,
                    dek_nonce      = ?,
                    updated_at     = ?
                WHERE id = 1
                """,
                (new_salt, _KDF_ALGORITHM, _KDF_ITERATIONS, new_wrapped_dek,
                 new_nonce, utc_now().isoformat()),
            )

        # Update in-memory DEK cache if the vault was already unlocked
        if _vault_state.dek is not None:
            _vault_state.dek = dek

        logger.info("[Vault] Password changed — DEK re-wrapped with new KEK")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_legacy_migration(self) -> bool:
        """Re-encrypt all legacy Fernet consumer rows with the vault DEK (AC15).

        Triggered automatically from :meth:`unlock` when the
        ``encryption_keys`` table still contains a row (``id = 1``).  Each
        consumer table is migrated inside its own transaction so a partial
        failure leaves the already-migrated tables intact.

        Migration targets:
          * ``providers.api_key``  — every non-NULL row.
          * ``settings.encrypted_value`` — rows where ``is_sensitive = 1``
            and ``encrypted_value IS NOT NULL``.
          * ``tool_configs.config_value`` — rows whose value Fernet-decrypts
            successfully (other rows are left untouched).
          * ``browser_credentials.encrypted_data`` — every non-NULL row.

        After all four tables succeed the ``encryption_keys`` singleton row
        is deleted, permanently closing the migration window.

        Returns:
            ``True``  if the migration completed and the ``encryption_keys``
                row was deleted.
            ``False`` if any table failed; the next :meth:`unlock` call will
                retry automatically (Fernet fallback remains active).
        """
        import hashlib
        from cryptography.fernet import Fernet, InvalidToken

        # ── Fetch the legacy key ──────────────────────────────────────────
        try:
            with self._db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key_value FROM encryption_keys WHERE id = 1")
                row = cursor.fetchone()
                cursor.close()
        except Exception:
            return True  # Table absent — nothing to migrate

        if row is None:
            return True  # Already migrated (row was deleted on a previous unlock)

        raw_key: str = row[0]
        key_bytes = hashlib.sha256(raw_key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        f = Fernet(fernet_key)

        def _try_fernet(val) -> Optional[bytes]:
            """Attempt Fernet decryption; return plaintext or None on failure.

            Args:
                val: Raw value from DB — may be ``bytes``, ``memoryview``,
                    or a UTF-8 string Fernet token.

            Returns:
                Plaintext bytes if decryption succeeds, ``None`` otherwise.
            """
            if val is None:
                return None
            token = bytes(val) if isinstance(val, (bytes, memoryview)) else val.encode("utf-8")
            try:
                return f.decrypt(token)
            except (InvalidToken, Exception):
                return None

        # ── Helper: migrate one table ─────────────────────────────────────

        def _migrate_table(
            select_sql: str, update_sql: str, id_col: int = 0, val_col: int = 1
        ) -> tuple:
            """Fetch, re-encrypt, and update all rows for a single table.

            All updates run inside a single transaction so an error causes
            the whole table's changes to be rolled back.

            Args:
                select_sql:  SQL that returns (id, encrypted_value) pairs.
                update_sql:  Parameterised UPDATE accepting ``(new_blob, id)``.
                id_col:      Index of the ID column in the result row.
                val_col:     Index of the encrypted-value column in the row.

            Returns:
                A ``(success: bool, migrated_count: int)`` tuple.  ``success``
                is ``False`` only when an unexpected exception occurs; a result
                with zero rows is still a success.
            """
            try:
                with self._db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(select_sql)
                    rows = cursor.fetchall()
                    cursor.close()

                    count = 0
                    for r in rows:
                        plaintext = _try_fernet(r[val_col])
                        if plaintext is None:
                            continue  # Not Fernet-encrypted; skip
                        new_blob = base64.b64encode(self.encrypt(plaintext)).decode()
                        conn.execute(update_sql, (new_blob, r[id_col]))
                        count += 1

                logger.debug("[Vault] Migration: migrated %d row(s) via %s", count, select_sql[:40])
                return (True, count)
            except Exception as exc:
                logger.warning("[Vault] Migration failed for query '%s': %s", select_sql[:40], exc)
                return (False, 0)

        # ── Run per-table migrations ──────────────────────────────────────

        table_results = [
            _migrate_table(
                "SELECT id, api_key FROM providers WHERE api_key IS NOT NULL",
                "UPDATE providers SET api_key = ? WHERE id = ?",
            ),
            _migrate_table(
                "SELECT id, encrypted_value FROM settings "
                "WHERE is_sensitive = 1 AND encrypted_value IS NOT NULL",
                "UPDATE settings SET encrypted_value = ? WHERE id = ?",
            ),
            _migrate_table(
                "SELECT id, config_value FROM tool_configs WHERE config_value IS NOT NULL",
                "UPDATE tool_configs SET config_value = ? WHERE id = ?",
            ),
            _migrate_table(
                "SELECT id, encrypted_data FROM browser_credentials "
                "WHERE encrypted_data IS NOT NULL",
                "UPDATE browser_credentials SET encrypted_data = ? WHERE id = ?",
            ),
        ]

        all_succeeded = all(ok for ok, _ in table_results)
        total_migrated = sum(cnt for _, cnt in table_results)

        if not all_succeeded:
            logger.warning(
                "[Vault] Legacy migration incomplete — will retry on next unlock. "
                "Per-table results (success, count): %s", table_results,
            )
            return False

        # ── Close the migration window only when rows were actually migrated ──
        # If no Fernet-encrypted rows were found (all tables empty or already
        # using AES-GCM), keep the encryption_keys row so the fallback path
        # remains active until real legacy data arrives and is migrated.
        if total_migrated == 0:
            logger.debug(
                "[Vault] Legacy migration: no Fernet rows found — keeping migration window open"
            )
            return True

        try:
            with self._db.connection() as conn:
                conn.execute("DELETE FROM encryption_keys WHERE id = 1")
            logger.info(
                "[Vault] Legacy Fernet migration complete (%d row(s) re-encrypted) "
                "— encryption_keys row deleted",
                total_migrated,
            )
        except Exception as exc:
            logger.warning("[Vault] Failed to delete encryption_keys row: %s", exc)
            return False

        return True

    def _load_vault_config(self) -> Optional[dict]:
        """Read the singleton ``vault_config`` row (id=1) from the database.

        Returns:
            A dict-like row with columns ``kdf_salt``, ``kdf_algorithm``,
            ``kdf_iterations``, ``wrapped_dek``, ``dek_nonce``,
            ``created_at``, ``updated_at``; or ``None`` if the table is empty.
        """
        with self._db.connection() as conn:
            conn.row_factory = __import__("sqlite3").Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT kdf_salt, kdf_algorithm, kdf_iterations, "
                "wrapped_dek, dek_nonce, created_at, updated_at "
                "FROM vault_config WHERE id = 1"
            )
            row = cursor.fetchone()
            cursor.close()
        return row  # sqlite3.Row or None

    def _try_fernet_fallback(self, blob: bytes) -> Optional[bytes]:
        """Attempt to decrypt *blob* using the legacy Fernet key from the database.

        This method is the *migration-window fallback* (AC16): during the
        period between first unlock (which opens the migration window) and
        full data re-encryption, some rows in consumer tables still contain
        Fernet ciphertexts.  This method allows those rows to be read
        transparently without forcing a synchronous migration on every decrypt.

        The legacy Fernet key derivation mirrors the logic in
        ``provider_db_service.ProviderDbService._decrypt``:
        ``key = base64url(SHA256(raw_key_string))``.

        Args:
            blob: Bytes that may be a UTF-8–encoded Fernet token.

        Returns:
            Decrypted plaintext bytes if Fernet decryption succeeds,
            ``None`` if the legacy key is absent or decryption fails.
        """
        import hashlib

        try:
            with self._db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key_value FROM encryption_keys WHERE id = 1")
                row = cursor.fetchone()
                cursor.close()
        except Exception:
            return None

        if row is None:
            return None

        raw_key: str = row[0]
        try:
            from cryptography.fernet import Fernet, InvalidToken

            key_bytes = hashlib.sha256(raw_key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(key_bytes)
            f = Fernet(fernet_key)

            # blob may be raw bytes of the UTF-8 Fernet token string
            token = blob if isinstance(blob, bytes) else blob.encode("utf-8")
            plaintext = f.decrypt(token)
            logger.debug("[Vault] Fernet fallback decryption succeeded (migration window)")
            return plaintext
        except Exception:
            # Also try base64 decode as a last-resort legacy path
            try:
                return base64.b64decode(blob)
            except Exception:
                return None


# ── Module-level factory ───────────────────────────────────────────────────────

_vault_service_instance: Optional[VaultService] = None


def get_vault_service() -> VaultService:
    """Return the process-wide :class:`VaultService` singleton.

    Uses deferred import of :func:`~services.database_service.get_shared_db_service`
    to avoid circular import issues at module load time.  The instance is
    cached after first creation so every call site shares the same object.

    Returns:
        The cached :class:`VaultService` singleton backed by the process-wide
        :class:`~services.database_service.DatabaseService` singleton.
    """
    global _vault_service_instance
    if _vault_service_instance is None:
        from services.database_service import get_shared_db_service
        _vault_service_instance = VaultService(get_shared_db_service())
    return _vault_service_instance
