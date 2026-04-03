-- 027: Schema cleanup — composite indexes + drop unused vault_secrets.
--
-- Adds targeted composite indexes on hot query paths identified via
-- codebase audit. Drops vault_secrets (created by 021, never used by
-- any application code).
--
-- All operations are backwards compatible: adding indexes is transparent
-- to older code, and vault_secrets has zero readers/writers.

-- ── Composite indexes ────────────────────────────────────────────

-- knowledge: most queries filter (kind, entity) together with deleted_at IS NULL
CREATE INDEX IF NOT EXISTS idx_knowledge_kind_entity_active
    ON knowledge(kind, entity) WHERE deleted_at IS NULL;

-- topic_transcript: always queried by topic + ordered by created_at
CREATE INDEX IF NOT EXISTS idx_transcript_topic_created_desc
    ON topic_transcript(topic, created_at DESC);

-- goals: goal ecology queries rank by salience within active statuses
CREATE INDEX IF NOT EXISTS idx_goals_active_salience
    ON goals(salience DESC) WHERE status IN ('candidate', 'strengthening', 'actionable');

-- documents: watched-folder scan queries filter folder + status
CREATE INDEX IF NOT EXISTS idx_documents_folder_pending
    ON documents(watched_folder_id, status)
    WHERE deleted_at IS NULL AND watched_folder_id IS NOT NULL;

-- ── Drop unused tables ──────────────────────────────────────────

-- vault_secrets was created by migration 021 "for future use" but
-- no application code reads or writes to it. If needed later, a
-- single migration can recreate it.
DROP INDEX IF EXISTS idx_vault_secrets_scope;
DROP INDEX IF EXISTS idx_vault_secrets_scope_ref;
DROP TABLE IF EXISTS vault_secrets;
