-- Enable pgcrypto (already enabled by 021, but redundant here is safe)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Change providers.api_key to BYTEA for encrypted storage.
-- Tables are new (019/020 untracked), so no existing data to migrate.
ALTER TABLE providers
    ALTER COLUMN api_key TYPE BYTEA USING NULL;

-- Add sensitivity flag and encrypted value column to settings.
ALTER TABLE settings
    ADD COLUMN IF NOT EXISTS is_sensitive BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE settings
    ADD COLUMN IF NOT EXISTS encrypted_value BYTEA;

-- Mark the REST API key row as sensitive.
UPDATE settings SET is_sensitive = TRUE WHERE key = 'api_key';
