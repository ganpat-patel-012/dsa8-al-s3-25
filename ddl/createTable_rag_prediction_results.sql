CREATE TABLE rag_prediction_results (
    id SERIAL PRIMARY KEY,
    p_id INTEGER REFERENCES predictions(p_id) ON DELETE CASCADE,
    justification TEXT,
    llm_label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
); 