CREATE TABLE web_data_scrap (
    id SERIAL PRIMARY KEY,
    p_id INTEGER REFERENCES predictions(p_id),
    statement VARCHAR(1000),
    url TEXT,
    scraped_content TEXT,
    evidence_summary TEXT,
    relevance_score NUMERIC(10, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);