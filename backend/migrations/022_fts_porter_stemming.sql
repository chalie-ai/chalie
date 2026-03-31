-- Migration 022: Rebuild document_chunks_fts with porter stemming.
--
-- Without stemming, "temperature" doesn't match "temperatures" and
-- "rise" doesn't match "rose" — causing FTS to miss obvious matches.
-- Porter stemmer normalises terms so inflected forms match.

DROP TABLE IF EXISTS document_chunks_fts;

CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content, section_title, content='document_chunks', content_rowid='id',
    tokenize='porter unicode61'
);

-- Repopulate from existing chunks
INSERT INTO document_chunks_fts (rowid, content, section_title)
    SELECT id, content, COALESCE(section_title, '') FROM document_chunks;
