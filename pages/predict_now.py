import streamlit as st
import pandas as pd
from configFiles.makePrediction import get_prediction_all
from configFiles.dbCode import insert_prediction, insert_feedback
from datetime import datetime
import psycopg2
from configFiles.config import DB_CONFIG
from configFiles.web_data_scrap import scrape_web_evidence
import time
import random
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(
    prefix="dsa8_",
    password="a-very-secret-password"
)
if not cookies.ready():
    st.stop()

# Restore session state from cookies if available
if cookies.get("authenticated") == "True":
    st.session_state["access_token"] = cookies.get("access_token")
    st.session_state["username"] = cookies.get("username")
    st.session_state["authenticated"] = True
    if cookies.get("user_id"):
        st.session_state["user_id"] = int(cookies.get("user_id"))
 
def show():
    # Sidebar login/register button if not authenticated
    if not st.session_state.get("authenticated", False):
        if st.sidebar.button("Login/Register"):
            try:
                st.switch_page("pages/profile.py")
            except Exception:
                st.experimental_set_query_params(page="profile")

    # --- Sidebar: Only show logout if authenticated ---
    if st.session_state.get("authenticated", False):
        st.sidebar.write(f"Logged in as: {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            for key in ["access_token", "username", "authenticated", "user_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear cookies on logout
            for key in ["access_token", "username", "authenticated", "user_id"]:
                cookies[key] = ""
            cookies.save()
            st.rerun()

    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.warning("You must be logged in to access this page.")
        return

    st.title(" Fake News Detection")
    tab2, tab1 = st.tabs(["Statement Only", "Full Input"])

    # --- Tab 1: Statement Only ---
    with tab2:
        st.subheader("🔍 Make your Prediction (Statement Only)")
        statement_only = st.text_input("Statement", placeholder="Please write the statement", key="statement_only")
        if st.button("🚀 Predict (Statement Only)"):
            payload = {
                "statement": statement_only,
                "subject": "",
                "speaker": "",
                "speakers_job_title": "",
                "location": "",
                "party": "",
                "context": ""
            }
            result = get_prediction_all(payload)
            threshold = 0.6
            probability_lstm = result.get("probability_lstm", 0.0) if isinstance(result, dict) else 0.0
            probability_gru = result.get("probability_gru", 0.0) if isinstance(result, dict) else 0.0
            probability_textcnn = result.get("probability_textcnn", 0.0) if isinstance(result, dict) else 0.0

            st.subheader("Model Probabilities")
            st.write("LSTM:", probability_lstm)
            st.write("GRU:", probability_gru)
            st.write("TEXTCNN:", probability_textcnn)

            predictions = {
                "Model": ["LSTM", "GRU", "TEXTCNN"],
                "Probability": [
                    round(probability_lstm, 4),
                    round(probability_gru, 4),
                    round(probability_textcnn, 4)
                ],
                "Prediction (>= 0.6)": [
                    probability_lstm >= threshold,
                    probability_gru >= threshold,
                    probability_textcnn >= threshold
                ]
            }
            df = pd.DataFrame(predictions)
            st.subheader("Prediction Results")
            st.table(df.reset_index(drop=True))

            st.subheader("Ensemble Approach")
            ensemble_prob = (probability_lstm + probability_gru + probability_textcnn) / 3
            ensemble_flag = ensemble_prob >= threshold
            st.write(f"Ensemble Prediction Probability: {probability_lstm:.4f} + {probability_gru:.4f} + {probability_textcnn:.4f} / 3 = {ensemble_prob:.4f}")
            st.write(f"Ensemble Prediction Flag: {ensemble_flag}")
            # Save prediction to DB
            result_data = {"p_statements": statement_only, "p_subjects": "", "p_speakers": "", "p_speakers_job_title": "","p_locations":"","p_party":"","p_context":"","p_probability_lstm":round(probability_lstm, 4),"p_probability_gru":round(probability_gru, 4),"p_probability_textcnn":round(probability_textcnn, 4),"p_ensemble_probability":round(ensemble_prob, 4),"p_flag_lstm":probability_lstm >= threshold,"p_flag_gru":probability_gru >= threshold,"p_flag_textcnn":probability_textcnn >= threshold,"p_ensemble_flag":ensemble_flag}
            user_id = st.session_state.get('user_id')
            if not user_id:
                conn = psycopg2.connect(**DB_CONFIG)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username=%s", (st.session_state['username'],))
                row = cursor.fetchone()
                user_id = row[0] if row else None
                conn.close()
                st.session_state['user_id'] = user_id
            result_data["p_user_id"] = user_id
            p_id = insert_prediction(result_data)
            if isinstance(p_id, int):
                st.session_state['last_p_id'] = p_id
                st.session_state['prediction_made'] = True
                st.success(f"✅ Prediction saved to database! (ID: {p_id})")
            else:
                st.session_state['last_p_id'] = None
                st.session_state['prediction_made'] = False
                st.error(p_id)
            # Web Evidence and RAG logic (shared)
            with st.spinner("Scraping web evidence for the statement... It can take up to 5 minutes... Please wait!"):
                evidence_df = scrape_web_evidence(statement_only)
            st.subheader("Web Evidence Results")
            if not evidence_df.empty:
                st.dataframe(evidence_df.reset_index(drop=True))
                if 'relevance_score' in evidence_df.columns:
                    valid_mask = (
                        evidence_df['evidence_summary'].notna() &
                        (evidence_df['evidence_summary'] != "No relevant evidence summary found")
                    )
                    valid_scores = evidence_df.loc[valid_mask, 'relevance_score'].dropna()
                    if not valid_scores.empty:
                        probability_web = valid_scores.mean()
                        st.write(f"Web Evidence Relevance Score (avg): {probability_web:.4f}")
                        st.write(f"Web Data flag: {probability_web >= threshold}")
                    else:
                        st.info("No relevant web evidence found for this statement.")
                if isinstance(p_id, int):
                    from configFiles.dbCode import insert_web_data_scrap
                    web_data_rows = []
                    for _, row in evidence_df.iterrows():
                        web_data_rows.append({
                            'p_id': p_id,
                            'statement': row.get('statement', ''),
                            'url': row.get('url', ''),
                            'scraped_content': row.get('scraped_content', ''),
                            'evidence_summary': row.get('evidence_summary', ''),
                            'relevance_score': row.get('relevance_score', None)
                        })
                    msg = insert_web_data_scrap(web_data_rows)
                    st.info(msg)
            else:
                st.info("No web evidence found for this statement.")
            if not evidence_df.empty:
                st.subheader("RAG Prediction Results")
                with st.spinner("RAG processing for the statement... It can take up to 5 minutes... Please wait!"):
                    try:
                        rag_df_input = evidence_df.rename(columns={
                            'scraped_content': 'content',
                            'evidence_summary': 'content_summary',
                            'relevance_score': 'probability'
                        })
                        rag_df_input = rag_df_input[[
                            'statement', 'url', 'content', 'content_summary', 'probability'
                        ]]
                        from configFiles.rag_prediction import main as rag_main
                        rag_df = rag_main(rag_df_input, statement_only)
                        st.dataframe(rag_df.reset_index(drop=True))
                        # Save RAG results to DB
                        if not rag_df.empty and isinstance(p_id, int):
                            from configFiles.dbCode import insert_rag_prediction_results
                            rag_rows = []
                            for _, row in rag_df.iterrows():
                                rag_rows.append({
                                    'p_id': p_id,
                                    'justification': row.get('justification', ''),
                                    'llm_label': row.get('llm_label', '')
                                })
                            msg = insert_rag_prediction_results(rag_rows)
                            st.info(msg)
                        st.subheader("RAG Justification and Prediction")
                        st.write("RAG Justification based on Dataset:")
                        st.write(rag_df['justification'].iloc[0] if not rag_df.empty else 'No justification found')
                        st.write("RAG Justification based on Web evidence:")
                        st.write(rag_df['justification'].iloc[1] if not rag_df.empty else 'No justification found')
                        st.write("RAG Prediction based on Dataset:")
                        st.write(rag_df['llm_label'].iloc[0] if not rag_df.empty else 'No prediction found')
                        st.write("RAG Prediction based on Web evidence:")
                        st.write(rag_df['llm_label'].iloc[1] if not rag_df.empty else 'No prediction found')
                    except Exception as e:
                        st.error(f"RAG prediction failed: {e}")

    # --- Tab 2: Full Input ---
    with tab1:
        st.subheader("🔍 Make your Prediction (Full Input)")
        if st.button("🎲 Fill with Random Test Data"):
            test_df = pd.read_csv("liar_dataset/test.tsv", sep="\t", header=None)
            row = test_df.sample(1).iloc[0]
            st.session_state["statement_full"] = row[2]
            st.session_state["subject_full"] = row[3]
            st.session_state["speaker_full"] = row[4]
            st.session_state["speakers_job_title_full"] = row[5] if pd.notnull(row[5]) else ""
            st.session_state["location_full"] = row[6] if pd.notnull(row[6]) else "Unknown"
            st.session_state["party_full"] = row[7] if pd.notnull(row[7]) else "None"
            st.session_state["context_full"] = row[13] if pd.notnull(row[13]) else ""
            st.session_state["_random_label"] = row[1]
            st.rerun()
        if "_random_label" in st.session_state:
            label = str(st.session_state["_random_label"]).strip().lower()
            display_label = "True" if label in ["true", "mostly-true"] else "False"
            st.info(f"Target label for this random sample: {display_label}")
        statement = st.text_input("Statement", placeholder="Please write the statement", key="statement_full")
        col1, col2 = st.columns(2)
        with col1:
            speaker = st.text_input("Speaker", placeholder="Please write the speaker's name", key="speaker_full")
            location = st.selectbox("Location", ["Unknown","Alabama","Alaska","Arizona","Arkansas","California","Colorado","Colorado ","Connecticut","Delaware","District of Columbia","Florida","Georgia","Illinois","Illinois ","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maryland","Massachusetts","Michigan","Minnesota","Missouri","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","Ohio","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","Washington D.C.","Washington, D.C.","Washington, D.C. ","West Virginia","Wisconsin","Wisconsin "], key="location_full")
            context = st.text_input("Context", placeholder="Please write the context", key="context_full")
        with col2:
            subject = st.text_input("Subject", placeholder="Please write the subject", key="subject_full")
            speakers_job_title = st.text_input("Speakers Job Title", placeholder="Please write the speaker's job title", key="speakers_job_title_full")
            party = st.selectbox("Party", ["None","Activist","Business-leader","Columnist","Constitution-party","County-commissioner","Democrat","Government-body","Independent","Journalist","Libertarian","Newsmaker","Organization","Republican","State-official","Talk-show-host"], key="party_full")
        if st.button("🚀 Predict (Full Input)"):
            statement_val = statement if statement else ""
            subject_val = subject if subject else ""
            speaker_val = speaker if speaker else ""
            speakers_job_title_val = speakers_job_title if speakers_job_title else ""
            location_val = location if location else ""
            party_val = party if party else ""
            context_val = context if context else ""
            payload = {
                "statement": statement_val, "subject": subject_val,
                "speaker": speaker_val, "speakers_job_title": speakers_job_title_val,
                "location": location_val, "party": party_val,
                "context": context_val
            }
            result = get_prediction_all(payload)
            threshold = 0.6
            probability_lstm = result.get("probability_lstm", 0.0) if isinstance(result, dict) else 0.0
            probability_gru = result.get("probability_gru", 0.0) if isinstance(result, dict) else 0.0
            probability_textcnn = result.get("probability_textcnn", 0.0) if isinstance(result, dict) else 0.0
            probs = [probability_lstm, probability_gru, probability_textcnn]
            
            st.subheader("Model Probabilities")
            st.write("LSTM:", probability_lstm)
            st.write("GRU:", probability_gru)
            st.write("TEXTCNN:", probability_textcnn)

            predictions = {
                "Model": ["LSTM", "GRU", "TEXTCNN"],
                "Probability": [
                    round(probability_lstm, 4),
                    round(probability_gru, 4),
                    round(probability_textcnn, 4),
                ],
                "Prediction (>= 0.6)": [
                    probability_lstm >= threshold,
                    probability_gru >= threshold,
                    probability_textcnn >= threshold,
                ]
            }
            df = pd.DataFrame(predictions)
            st.subheader("Prediction Results")
            st.table(df.reset_index(drop=True))

            st.subheader("Ensemble Approach")
            ensemble_prob = (probability_lstm + probability_gru + probability_textcnn) / 3
            ensemble_flag = ensemble_prob >= threshold
            st.write(f"Ensemble Prediction Probability: {probability_lstm:.4f} + {probability_gru:.4f} + {probability_textcnn:.4f} / 3 = {ensemble_prob:.4f}")
            st.write(f"Ensemble Prediction Flag: {ensemble_flag}")
            
            # Save prediction to DB
            result_data = {"p_statements": statement_val, "p_subjects": subject_val, "p_speakers": speaker_val ,"p_speakers_job_title": speakers_job_title_val,"p_locations":location_val,"p_party":party_val,"p_context":context_val,"p_probability_lstm":round(probability_lstm, 4),"p_probability_gru":round(probability_gru, 4),"p_probability_textcnn":round(probability_textcnn, 4),"p_ensemble_probability":round(ensemble_prob, 4),"p_flag_lstm":probability_lstm >= threshold,"p_flag_gru":probability_gru >= threshold,"p_flag_textcnn":probability_textcnn >= threshold,"p_ensemble_flag":ensemble_flag}
            user_id = st.session_state.get('user_id')
            if not user_id:
                conn = psycopg2.connect(**DB_CONFIG)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username=%s", (st.session_state['username'],))
                row = cursor.fetchone()
                user_id = row[0] if row else None
                conn.close()
                st.session_state['user_id'] = user_id
            result_data["p_user_id"] = user_id
            p_id = insert_prediction(result_data)
            if isinstance(p_id, int):
                st.session_state['last_p_id'] = p_id
                st.session_state['prediction_made'] = True
                st.success(f"✅ Prediction saved to database! (ID: {p_id})")
            else:
                st.session_state['last_p_id'] = None
                st.session_state['prediction_made'] = False
                st.error(p_id)
            # Web Evidence and RAG logic (shared)
            with st.spinner("Scraping web evidence for the statement... It can take up to 5 minutes... Please wait!"):
                evidence_df = scrape_web_evidence(statement_val)
            st.subheader("Web Evidence Results")
            if not evidence_df.empty:
                st.dataframe(evidence_df.reset_index(drop=True))
                if 'relevance_score' in evidence_df.columns:
                    valid_mask = (
                        evidence_df['evidence_summary'].notna() &
                        (evidence_df['evidence_summary'] != "No relevant evidence summary found")
                    )
                    valid_scores = evidence_df.loc[valid_mask, 'relevance_score'].dropna()
                    if not valid_scores.empty:
                        probability_web = valid_scores.mean()
                        st.write(f"Web Evidence Relevance Score (avg): {probability_web:.4f}")
                        st.write(f"Web Data flag: {probability_web >= threshold}")
                    else:
                        st.info("No relevant web evidence found for this statement.")
                if isinstance(p_id, int):
                    from configFiles.dbCode import insert_web_data_scrap
                    web_data_rows = []
                    for _, row in evidence_df.iterrows():
                        web_data_rows.append({
                            'p_id': p_id,
                            'statement': row.get('statement', ''),
                            'url': row.get('url', ''),
                            'scraped_content': row.get('scraped_content', ''),
                            'evidence_summary': row.get('evidence_summary', ''),
                            'relevance_score': row.get('relevance_score', None)
                        })
                    msg = insert_web_data_scrap(web_data_rows)
                    st.info(msg)
            else:
                st.info("No web evidence found for this statement.")
            if not evidence_df.empty:
                st.subheader("RAG Prediction Results")
                with st.spinner("RAG processing for the statement... It can take up to 5 minutes... Please wait!"):
                    try:
                        rag_df_input = evidence_df.rename(columns={
                            'scraped_content': 'content',
                            'evidence_summary': 'content_summary',
                            'relevance_score': 'probability'
                        })
                        rag_df_input = rag_df_input[[
                            'statement', 'url', 'content', 'content_summary', 'probability'
                        ]]
                        from configFiles.rag_prediction import main as rag_main
                        rag_df = rag_main(rag_df_input, statement_val)
                        st.dataframe(rag_df.reset_index(drop=True))
                        # Save RAG results to DB
                        if not rag_df.empty and isinstance(p_id, int):
                            from configFiles.dbCode import insert_rag_prediction_results
                            rag_rows = []
                            for _, row in rag_df.iterrows():
                                rag_rows.append({
                                    'p_id': p_id,
                                    'justification': row.get('justification', ''),
                                    'llm_label': row.get('llm_label', '')
                                })
                            msg = insert_rag_prediction_results(rag_rows)
                            st.info(msg)
                        st.subheader("RAG Justification and Prediction")
                        st.write("RAG Justification based on Dataset:")
                        st.write(rag_df['justification'].iloc[0] if not rag_df.empty else 'No justification found')
                        st.write("RAG Justification based on Web evidence:")
                        st.write(rag_df['justification'].iloc[1] if not rag_df.empty else 'No justification found')
                        st.write("RAG Prediction based on Dataset:")
                        st.write(rag_df['llm_label'].iloc[0] if not rag_df.empty else 'No prediction found')
                        st.write("RAG Prediction based on Web evidence:")
                        st.write(rag_df['llm_label'].iloc[1] if not rag_df.empty else 'No prediction found')
                    except Exception as e:
                        st.error(f"RAG prediction failed: {e}")

    # Feedback form after prediction (shared)
    if st.session_state.get('prediction_made') and st.session_state.get('last_p_id'):
        st.subheader("📝 Submit Feedback for this Prediction")
        feedback_flag = st.selectbox("Feedback Flag", ["True", "False"], key="feedback_flag")
        feedback_comment = st.text_area("Comment", key="feedback_comment")
        if st.button("Submit Feedback"):
            f_p_id = st.session_state['last_p_id']
            f_statemnt = statement
            f_flag = feedback_flag
            f_comment = feedback_comment
            feedback_msg = insert_feedback(f_p_id, f_statemnt, f_flag, f_comment)
            st.success(feedback_msg)
            st.session_state['prediction_made'] = False
            st.session_state['last_p_id'] = None
 
if __name__ == "__main__":
    show()