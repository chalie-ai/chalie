-- Migration 042: Create documents and document_chunks tables
-- Document intelligence pipeline — upload, extract, chunk, embed, search

CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    original_name       TEXT NOT NULL,
    mime_type           TEXT NOT NULL,
    file_size_bytes     BIGINT,
    file_path           TEXT NOT NULL,
    file_hash           TEXT,
    page_count          INTEGER,
    status              VARCHAR(20) DEFAULT 'pending',
    error_message       TEXT,
    chunk_count         INTEGER DEFAULT 0,
    source_type         VARCHAR(20) DEFAULT 'upload',
    tags                TEXT[] DEFAULT '{}',
    summary             TEXT,
    summary_embedding   vector(768),
    extracted_metadata  JSONB DEFAULT '{}',
    supersedes_id       TEXT REFERENCES documents(id),
    clean_text          TEXT,
    language            VARCHAR(10),
    fingerprint         TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    deleted_at          TIMESTAMPTZ,
    purge_after         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_deleted ON documents (deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_purge ON documents (purge_after) WHERE purge_after IS NOT NULL;

-- HNSW index on summary_embedding for coarse document-level search
CREATE INDEX IF NOT EXISTS idx_documents_summary_embedding
    ON documents USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    page_number     INTEGER,
    section_title   TEXT,
    token_count     INTEGER,
    embedding       vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks (document_id);

-- HNSW index on chunk embeddings for fine-grained search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
