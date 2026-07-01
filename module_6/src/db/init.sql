-- init.sql — schema bootstrap for the PostgreSQL service.
--
-- Docker Compose mounts this file into the postgres image's
-- /docker-entrypoint-initdb.d/ directory, so it runs automatically the first
-- time the named volume is initialised (i.e. on a clean machine). The worker
-- also calls db.load_data.create_schema(), so the schema exists even without
-- Docker (local dev / CI). Both paths are idempotent.

CREATE TABLE IF NOT EXISTS applicants (
    p_id                       SERIAL PRIMARY KEY,
    result_id                  BIGINT,
    program                    TEXT,
    comments                   TEXT,
    date_added                 DATE,
    url                        TEXT UNIQUE,
    status                     TEXT,
    term                       TEXT,
    us_or_international         TEXT,
    gpa                        FLOAT,
    gre                        FLOAT,
    gre_v                      FLOAT,
    gre_aw                     FLOAT,
    degree                     TEXT,
    llm_generated_program      TEXT,
    llm_generated_university   TEXT
);

-- High-water mark per source: enables incremental, idempotent ingestion.
CREATE TABLE IF NOT EXISTS ingestion_watermarks (
    source        TEXT PRIMARY KEY,
    last_seen     TEXT,
    updated_at    TIMESTAMPTZ DEFAULT now()
);

-- Cached scalar metrics rendered by the web UI; refreshed by the worker's
-- recompute_analytics task.
CREATE TABLE IF NOT EXISTS analysis_summary (
    metric        TEXT PRIMARY KEY,
    value         DOUBLE PRECISION,
    updated_at    TIMESTAMPTZ DEFAULT now()
);
