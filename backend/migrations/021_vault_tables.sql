-- Envelope-encryption vault tables.
-- Stores the password-derived key-encryption-key (KEK) parameters and the
-- wrapped data-encryption-key (DEK) used by VaultService for AES-256-GCM
-- envelope encryption.  Also provides an optional centralised secret store
-- (vault_secrets) for future use.  Idempotent (IF NOT EXISTS).

-- ────────────────────────────────────────────────────────────────
-- 1. vault_config — singleton row (id MUST equal 1)
--    Holds all KDF parameters and the KEK-wrapped DEK.
--    Only one row is ever allowed (enforced by CHECK (id = 1)).
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vault_config (
    -- Singleton sentinel; the CHECK constraint rejects any INSERT with id != 1.
    id              INTEGER PRIMARY KEY CHECK (id = 1),

    -- 32-byte random salt used when deriving the KEK from the master password.
    kdf_salt        BLOB    NOT NULL,

    -- KDF algorithm identifier; currently only 'pbkdf2_sha256' is supported.
    kdf_algorithm   TEXT    NOT NULL DEFAULT 'pbkdf2_sha256',

    -- PBKDF2 iteration count (≥ 600 000 recommended for PBKDF2-HMAC-SHA256).
    kdf_iterations  INTEGER NOT NULL DEFAULT 600000,

    -- KEK-encrypted DEK blob: nonce (12 bytes) || ciphertext || GCM tag (16 bytes).
    wrapped_dek     BLOB    NOT NULL,

    -- 12-byte GCM nonce used when wrapping the DEK with the KEK.
    dek_nonce       BLOB    NOT NULL,

    created_at      TEXT,
    updated_at      TEXT
);

-- ────────────────────────────────────────────────────────────────
-- 2. vault_secrets — optional centralised secret store
--    Reserved for future use as a single location for all
--    application secrets encrypted with the vault DEK.
--    Consumer services may continue to store ciphertext in their
--    own columns; this table is not required by Phase 1.
-- ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vault_secrets (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Logical namespace for the secret, e.g. 'provider', 'setting',
    -- 'tool_config', 'browser_credential'.
    scope                 TEXT    NOT NULL,

    -- Opaque reference within the scope, e.g. the provider name or setting key.
    scope_ref             TEXT    NOT NULL,

    -- AES-256-GCM ciphertext: nonce (12 bytes) || ciphertext || tag (16 bytes).
    encrypted_value       BLOB    NOT NULL,

    -- 12-byte GCM nonce prepended inside encrypted_value (stored separately
    -- for quick lookup / rotation without re-parsing the blob).
    nonce                 BLOB    NOT NULL,

    -- Set to 1 after a legacy Fernet-encrypted value has been re-encrypted
    -- and written back to this table during the migration window.
    migrated_from_fernet  INTEGER DEFAULT 0,

    created_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_vault_secrets_scope     ON vault_secrets(scope);
CREATE INDEX IF NOT EXISTS idx_vault_secrets_scope_ref ON vault_secrets(scope, scope_ref);
