-- 019_create_providers.sql
-- Create providers table and job_provider_assignments for DB-backed provider configuration.
--
-- After migration, populate providers via the REST API:
--   POST /providers {"name": "...", "platform": "...", "model": "...", ...}
--
-- Supported platforms: ollama, anthropic, openai, gemini

CREATE TABLE IF NOT EXISTS providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    platform VARCHAR(50) NOT NULL,
    model VARCHAR(255) NOT NULL,
    host VARCHAR(500),
    api_key VARCHAR(500),
    dimensions INTEGER,
    timeout INTEGER DEFAULT 120,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_provider_assignments (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(255) UNIQUE NOT NULL,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
CREATE INDEX IF NOT EXISTS idx_providers_platform ON providers(platform);
CREATE INDEX IF NOT EXISTS idx_job_assignments_job ON job_provider_assignments(job_name);
