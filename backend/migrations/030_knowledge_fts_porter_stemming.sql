-- Migration 030: Rebuild knowledge_fts with porter stemming and search_queries column.
--
-- Without stemming, "running" doesn't match "run" and "memories" doesn't match
-- "memory" — causing FTS to miss inflected forms. Porter stemmer normalises terms.
-- Also adds search_queries column so doc2query-generated queries are indexed.

DROP TABLE IF EXISTS knowledge_fts;

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    key, value, kind, entity, search_queries, content='knowledge', content_rowid='rowid',
    tokenize='porter unicode61'
);

INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild');
