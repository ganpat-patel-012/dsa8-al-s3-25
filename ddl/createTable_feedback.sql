CREATE TABLE feedback (
  f_id SERIAL PRIMARY KEY,
  f_p_id INTEGER NOT NULL
    REFERENCES predictions(p_id)
    ON DELETE CASCADE,
  f_statemnt VARCHAR(500) NOT NULL,
  f_flag VARCHAR(50) NOT NULL,
  f_comment TEXT
);