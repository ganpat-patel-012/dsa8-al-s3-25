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
    Returns the inserted p_id for single insert, or a list of p_ids for batch insert.
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
                        p_user_id,
                        p_probability_lstm,
                        p_probability_gru,
                        p_probability_textcnn,
                        p_ensemble_probability,
                        p_flag_lstm,
                        p_flag_gru,
                        p_flag_textcnn,
                        p_ensemble_flag
                    ) VALUES ({placeholders}) RETURNING p_id
                """).format(
                    placeholders=sql.SQL(', ').join(sql.Placeholder() * 16)
                )

                if isinstance(data, list):
                    values = [
                        (
                            item["p_statements"],
                            item["p_subjects"],
                            item["p_speakers"],
                            item["p_speakers_job_title"],
                            item["p_locations"],
                            item["p_party"],
                            item["p_context"],
                            item["p_user_id"],
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
                    p_ids = [row[0] for row in cursor.fetchall()]
                    conn.commit()
                    return p_ids
                else:
                    cursor.execute(insert_query, (
                        data["p_statements"],
                        data["p_subjects"],
                        data["p_speakers"],
                        data["p_speakers_job_title"],
                        data["p_locations"],
                        data["p_party"],
                        data["p_context"],
                        data["p_user_id"],
                        data["p_probability_lstm"],
                        data["p_probability_gru"],
                        data["p_probability_textcnn"],
                        data["p_ensemble_probability"],
                        data["p_flag_lstm"],
                        data["p_flag_gru"],
                        data["p_flag_textcnn"],
                        data["p_ensemble_flag"]
                    ))
                    p_id = cursor.fetchone()[0]
                    conn.commit()
                    return p_id
        
    except Exception as e:
        return f"\u274c Database Error: {e}"

# New function to insert feedback
def insert_feedback(f_p_id, f_statemnt, f_flag, f_comment):
    """
    Inserts feedback for a prediction into the feedback table.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                insert_query = """
                    INSERT INTO feedback (f_p_id, f_statemnt, f_flag, f_comment)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (f_p_id, f_statemnt, f_flag, f_comment))
            conn.commit()
        return "\u2705 Feedback saved to database!"
    except Exception as e:
        return f"\u274c Database Error: {e}"

def insert_web_data_scrap(web_data_rows):
    """
    Inserts web evidence data into web_data_scrap table. Expects a list of dicts, each with keys:
    'p_id', 'statement', 'url', 'scraped_content', 'evidence_summary', 'relevance_score'.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                insert_query = """
                    INSERT INTO web_data_scrap (
                        p_id, statement, url, scraped_content, evidence_summary, relevance_score
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = [
                    (
                        row['p_id'],
                        row['statement'],
                        row['url'],
                        row['scraped_content'],
                        row['evidence_summary'],
                        row['relevance_score']
                    ) for row in web_data_rows
                ]
                cursor.executemany(insert_query, values)
            conn.commit()
        return "\u2705 Web evidence data saved to database!"
    except Exception as e:
        return f"\u274c Database Error (web_data_scrap): {e}"
