import psycopg2
from psycopg2 import sql
from configFiles.config import DB_CONFIG

def get_connection():
    """Establishes a PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)

def insert_prediction(data):
    """
    Inserts prediction data into PostgreSQL, supporting both single and batch inserts.
    Expects dict(s) with keys matching the p_ column names (except prediction_time).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                insert_query = sql.SQL("""
                    INSERT INTO predictions (
                        p_statements,
                        p_subjects,
                        p_speakers,
                        p_speakers_job_title,
                        p_locations,
                        p_party,
                        p_context,
                        p_probability_lstm,
                        p_probability_gru,
                        p_probability_textcnn,
                        p_ensemble_probability,
                        p_flag_lstm,
                        p_flag_gru,
                        p_flag_textcnn,
                        p_ensemble_flag
                    ) VALUES ({placeholders})
                """).format(
                    placeholders=sql.SQL(', ').join(sql.Placeholder() * 15)
                )

                if isinstance(data, list):
                    # Batch insert: build a list of tuples
                    values = [
                        (
                            item["p_statements"],
                            item["p_subjects"],
                            item["p_speakers"],
                            item["p_speakers_job_title"],
                            item["p_locations"],
                            item["p_party"],
                            item["p_context"],
                            item["p_probability_lstm"],
                            item["p_probability_gru"],
                            item["p_probability_textcnn"],
                            item["p_ensemble_probability"],
                            item["p_flag_lstm"],
                            item["p_flag_gru"],
                            item["p_flag_textcnn"],
                            item["p_ensemble_flag"]
                        )
                        for item in data
                    ]
                    cursor.executemany(insert_query, values)
                else:
                    # Single insert
                    cursor.execute(insert_query, (
                        data["p_statements"],
                        data["p_subjects"],
                        data["p_speakers"],
                        data["p_speakers_job_title"],
                        data["p_locations"],
                        data["p_party"],
                        data["p_context"],
                        data["p_probability_lstm"],
                        data["p_probability_gru"],
                        data["p_probability_textcnn"],
                        data["p_ensemble_probability"],
                        data["p_flag_lstm"],
                        data["p_flag_gru"],
                        data["p_flag_textcnn"],
                        data["p_ensemble_flag"]
                    ))

            conn.commit()
        return "✅ Prediction(s) saved to database!"
    
    except Exception as e:
        return f"❌ Database Error: {e}"
