-- Extracted from the local schema-v2-hybrid analysis database.
-- payload is stored as JSON TEXT; sample.json expands it as an object.

CREATE TABLE documents (
                report_id TEXT PRIMARY KEY, path TEXT NOT NULL, filename TEXT NOT NULL,
                title TEXT NOT NULL, published_date TEXT NOT NULL, category TEXT NOT NULL,
                content_hash TEXT NOT NULL, mtime_ns INTEGER, size INTEGER, available INTEGER NOT NULL,
                status TEXT NOT NULL, error TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL);

CREATE TABLE passages (chunk_id TEXT PRIMARY KEY, report_id TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL);

CREATE VIRTUAL TABLE passage_fts USING fts5(chunk_id UNINDEXED, report_id UNINDEXED, text, tokenize='unicode61');
